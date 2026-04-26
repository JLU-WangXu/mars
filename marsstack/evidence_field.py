from __future__ import annotations

import math
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
    state = bucket.setdefault(
        aa,
        {
            "score": 0.0,
            "supporting_sources": set(),
            "evidence_breakdown": {},
        },
    )
    state["score"] = float(state["score"]) + float(score)
    state["evidence_breakdown"][source_label] = round(
        float(state["evidence_breakdown"].get(source_label, 0.0)) + float(score),
        6,
    )
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
    feat_map = {feat.num: feat for feat in features}
    oxidation_set = set(int(x) for x in oxidation_hotspots)
    flexible_set = set(int(x) for x in flexible_positions)

    per_position: dict[int, dict[str, dict[str, Any]]] = {}
    for position in design_positions:
        idx = position_to_index[int(position)]
        wt = wt_seq[idx]
        per_position[int(position)] = {
            wt: {
                "score": 0.0,
                "supporting_sources": set(),
                "evidence_breakdown": {"wt_anchor": 0.0},
            }
        }

    for position in design_positions:
        position = int(position)
        idx = position_to_index[position]
        wt = wt_seq[idx]
        bucket = per_position[position]

        for aa, bias in manual_bias.get(position, {}).items():
            _register_option(bucket, aa, "manual_prior", float(bias))

        if position in oxidation_set:
            for aa, val in SAFE_OXIDATION_MAP.get(wt, {}).items():
                _register_option(bucket, aa, "structure_hotspot", 0.75 + 0.35 * float(val))

        if position in flexible_set and position not in oxidation_set:
            feat = feat_map.get(position)
            sasa_scale = 1.0 if feat is None else min(1.25, max(0.65, float(feat.sasa) / 80.0))
            for aa, base in HYDRATING_AA_SCORES.items():
                if aa != wt:
                    _register_option(bucket, aa, "structure_surface", float(base) * sasa_scale)

        if profile is not None and idx < len(profile):
            ranked = sorted(
                [(aa, float(prob)) for aa, prob in profile[idx].items() if aa not in {"-", wt}],
                key=lambda item: (-item[1], item[0]),
            )[:4]
            for aa, prob in ranked:
                if prob >= 0.04:
                    _register_option(bucket, aa, "evolution_profile", 1.35 * prob)

        for aa, delta in (family_recommendations or {}).get(position, {}).items():
            if aa != wt:
                _register_option(bucket, aa, "family_differential", 1.8 * float(delta))

        for aa, prob in ((ancestral_field or {}).get(position, {}).get("recommendations", {}) or {}).items():
            if aa != wt:
                _register_option(bucket, aa, "ancestral_field", 1.6 * float(prob))

        for aa, score in (retrieval_recommendations or {}).get(position, {}).items():
            if aa == wt:
                continue
            if position in oxidation_set:
                allowed_hotspot = set(SAFE_OXIDATION_MAP.get(wt, {})) | set(manual_bias.get(position, {}))
                if aa not in allowed_hotspot:
                    continue
            _register_option(bucket, aa, "retrieval_memory", 0.9 * float(score))

    source_buckets: dict[str, list[dict[str, object]]] = {}
    for row in proposal_rows:
        source = str(row.get("source", ""))
        source_buckets.setdefault(source, []).append(row)

    for source in sorted(source_buckets):
        ranked_rows = sorted(
            source_buckets[source],
            key=lambda item: (
                -float(item.get("ranking_score", item.get("mars_score", 0.0))),
                -float(item.get("mars_score", 0.0)),
                str(item.get("candidate_id", "")),
            ),
        )[: int(max_rows_per_source)]
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

            for position in design_positions:
                idx = position_to_index[int(position)]
                aa = sequence[idx]
                _register_option(
                    per_position[int(position)],
                    aa,
                    "generator_evidence",
                    contribution,
                    proposal_source=source,
                )

    fields: list[PositionField] = []
    for position in sorted(per_position):
        idx = position_to_index[position]
        wt = wt_seq[idx]
        ranked = sorted(
            per_position[position].items(),
            key=lambda item: (
                -float(item[1]["score"]),
                -len(item[1]["supporting_sources"]),
                item[0],
            ),
        )[: int(top_k_per_position)]
        if wt not in {aa for aa, _ in ranked}:
            ranked.append((wt, per_position[position][wt]))
            ranked.sort(
                key=lambda item: (
                    -float(item[1]["score"]),
                    -len(item[1]["supporting_sources"]),
                    item[0],
                )
            )
            ranked = ranked[: int(top_k_per_position)]

        fields.append(
            PositionField(
                position=position,
                wt_residue=wt,
                options=[
                    ResidueOption(
                        residue=aa,
                        score=round(float(option_state["score"]), 6),
                        supporting_sources=sorted(option_state["supporting_sources"]),
                        support_strength=round(float(option_state["score"]), 6),
                        evidence_breakdown={k: round(float(v), 6) for k, v in option_state["evidence_breakdown"].items()},
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
