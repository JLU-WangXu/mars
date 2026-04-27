from __future__ import annotations

import math
from collections import defaultdict
from operator import itemgetter
from typing import Any

from .decoder import PositionField, ResidueOption
from .mars_score import SAFE_OXIDATION_MAP
from .structure_features import ResidueFeature


HYDRATING_AA_SCORES = {
    "Q": 0.65,
    "N": 0.58,
    "E": 0.42,
    "D": 0.34,
    "S": 0.18,
    "T": 0.18,
}


def _register_option(
    bucket: dict[str, dict[str, Any]],
    aa: str,
    source_label: str,
    score: float,
    proposal_source: str | None = None,
) -> None:
    state = bucket.get(aa)
    if state is None:
        state = {
            "score": 0.0,
            "supporting_sources": set(),
            "evidence_breakdown": {},
        }
        bucket[aa] = state
    # Cache score update to avoid repeated dict lookups
    state["score"] += score
    # Direct assignment avoids get() call - breakdown starts empty
    breakdown = state["evidence_breakdown"]
    breakdown[source_label] = round(breakdown.get(source_label, 0.0) + score, 6)
    if proposal_source:
        state["supporting_sources"].add(proposal_source)


def _proposal_signal(ranking_score: float, mars_score: float, rank_idx: int) -> float:
    rank_discount = 1.0 / math.sqrt(max(1, rank_idx))
    return (0.42 + 0.28 * math.tanh(max(0.0, ranking_score) / 4.0) + 0.18 * math.tanh(max(0.0, mars_score) / 4.0)) * rank_discount


def build_unified_evidence_fields(
    wt_seq: str,
    design_positions: list[int],
    position_to_index: dict[int, int],
    features: list[ResidueFeature],
    oxidation_hotspots: list[int],
    flexible_positions: list[int],
    manual_bias: dict[int, dict[str, float]],
    profile: list[dict[str, float]] | None,
    family_recommendations: dict[int, dict[str, float]] | None,
    ancestral_field: dict[int, dict[str, Any]] | None,
    retrieval_recommendations: dict[int, dict[str, float]] | None,
    proposal_rows: list[dict[str, object]],
    top_k_per_position: int = 4,
    max_rows_per_source: int = 12,
) -> list[PositionField]:
    # Pre-compute feature map once
    feat_map = {feat.num: feat for feat in features}
    oxidation_set = set(oxidation_hotspots)
    flexible_set = set(flexible_positions)

    # Pre-compute wt residues for all positions (avoids repeated wt_seq lookups)
    position_wt: dict[int, str] = {}
    for pos in design_positions:
        pos_int = int(pos)
        position_wt[pos_int] = wt_seq[position_to_index[pos_int]]

    # Initialize per_position with wt entries
    per_position: dict[int, dict[str, dict[str, Any]]] = {
        pos_int: {
            position_wt[pos_int]: {
                "score": 0.0,
                "supporting_sources": set(),
                "evidence_breakdown": {"wt_anchor": 0.0},
            }
        }
        for pos_int in position_wt
    }

    # Cache lookups as local variables for faster access
    p2i = position_to_index
    manual_bias_get = manual_bias.get
    family_get = (family_recommendations or {}).get
    ancestral_get = (ancestral_field or {}).get
    retrieval_get = (retrieval_recommendations or {}).get
    safe_ox = SAFE_OXIDATION_MAP
    profile_len = len(profile) if profile is not None else 0

    for position in design_positions:
        pos_int = int(position)
        idx = p2i[pos_int]
        wt = position_wt[pos_int]
        bucket = per_position[pos_int]

        # Process manual_bias
        bias_data = manual_bias_get(position, {})
        for aa, bias in bias_data.items():
            _register_option(bucket, aa, "manual_prior", float(bias))

        # Process oxidation hotspots
        if pos_int in oxidation_set:
            for aa, val in safe_ox.get(wt, ()).items():
                _register_option(bucket, aa, "structure_hotspot", 0.75 + 0.35 * float(val))

        # Process flexible positions (excluding oxidation)
        if pos_int in flexible_set and pos_int not in oxidation_set:
            feat = feat_map.get(pos_int)
            if feat is None:
                sasa_scale = 1.0
            else:
                sasa_scale = min(1.25, max(0.65, float(feat.sasa) / 80.0))
            for aa, base in HYDRATING_AA_SCORES.items():
                if aa != wt:
                    _register_option(bucket, aa, "structure_surface", float(base) * sasa_scale)

        # Process evolution profile
        if profile is not None and idx < profile_len:
            profile_item = profile[idx]
            ranked = [(aa, prob) for aa, prob in profile_item.items() if aa != "-" and aa != wt]
            ranked.sort(key=lambda item: (-item[1], item[0]))
            del ranked[4:]  # Keep top 4 in-place
            for aa, prob in ranked:
                if prob >= 0.04:
                    _register_option(bucket, aa, "evolution_profile", 1.35 * prob)

        # Process family recommendations
        family_data = family_get(position, {})
        for aa, delta in family_data.items():
            if aa != wt:
                _register_option(bucket, aa, "family_differential", 1.8 * float(delta))

        # Process ancestral field
        ancestral_data = ancestral_get(position, {})
        ancestral_recs = ancestral_data.get("recommendations", {}) if ancestral_data else {}
        for aa, prob in ancestral_recs.items():
            if aa != wt:
                _register_option(bucket, aa, "ancestral_field", 1.6 * float(prob))

        # Process retrieval recommendations
        retrieval_data = retrieval_get(position, {})
        if pos_int in oxidation_set:
            allowed_hotspot = set(safe_ox.get(wt, ())) | set(bias_data)
            for aa, score in retrieval_data.items():
                if aa != wt and aa in allowed_hotspot:
                    _register_option(bucket, aa, "retrieval_memory", 0.9 * float(score))
        else:
            for aa, score in retrieval_data.items():
                if aa != wt:
                    _register_option(bucket, aa, "retrieval_memory", 0.9 * float(score))

    # Build source buckets using defaultdict pattern
    source_buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in proposal_rows:
        source_buckets[str(row.get("source", ""))].append(row)

    # Pre-compute design_positions count for loop
    num_positions = len(design_positions)
    sorted_positions = design_positions
    pos_indices = [p2i[int(p)] for p in design_positions]

    for source in sorted(source_buckets):
        ranked_rows = sorted(
            source_buckets[source],
            key=lambda item: (
                -float(item.get("ranking_score", item.get("mars_score", 0.0))),
                -float(item.get("mars_score", 0.0)),
                str(item.get("candidate_id", "")),
            ),
        )[:max_rows_per_source]
        for rank_idx, row in enumerate(ranked_rows, start=1):
            sequence = str(row.get("sequence", ""))
            ranking_score = float(row.get("ranking_score", row.get("mars_score", 0.0)))
            mars_score = float(row.get("mars_score", 0.0))
            contribution = _proposal_signal(ranking_score, mars_score, rank_idx)
            source_group = str(row.get("source_group", ""))
            if source_group == "learned":
                contribution *= 0.92
            elif source_group == "heuristic_local":
                contribution *= 0.82
            elif source == "manual":
                contribution *= 0.75

            for i in range(num_positions):
                pos_int = int(sorted_positions[i])
                idx = pos_indices[i]
                aa = sequence[idx]
                _register_option(
                    per_position[pos_int],
                    aa,
                    "generator_evidence",
                    contribution,
                    proposal_source=source,
                )

    # Build final fields
    fields: list[PositionField] = []
    # Pre-compute sorted key for consistent use
    for position in sorted(per_position):
        idx = p2i[position]
        wt = position_wt[position]
        position_data = per_position[position]
        ranked = sorted(
            position_data.items(),
            key=lambda item: (-item[1]["score"], -len(item[1]["supporting_sources"]), item[0]),
        )[:top_k_per_position]
        if wt not in {aa for aa, _ in ranked}:
            ranked.append((wt, position_data[wt]))
            ranked.sort(key=lambda item: (-item[1]["score"], -len(item[1]["supporting_sources"]), item[0]))
            del ranked[top_k_per_position:]

        fields.append(
            PositionField(
                position=position,
                wt_residue=wt,
                options=[
                    ResidueOption(
                        residue=aa,
                        score=round(option_state["score"], 6),
                        supporting_sources=sorted(option_state["supporting_sources"]),
                        support_strength=round(option_state["score"], 6),
                        evidence_breakdown={k: round(v, 6) for k, v in option_state["evidence_breakdown"].items()},
                    )
                    for aa, option_state in ranked
                ],
            )
        )
    return fields


def serialize_evidence_fields(fields: list[PositionField]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for field in fields:
        payload.append(
            {
                "position": field.position,
                "wt_residue": field.wt_residue,
                "options": [
                    {
                        "residue": option.residue,
                        "score": option.score,
                        "supporting_sources": option.supporting_sources or [],
                        "support_strength": option.support_strength,
                        "evidence_breakdown": option.evidence_breakdown or {},
                    }
                    for option in field.options
                ],
            }
        )
    return payload
