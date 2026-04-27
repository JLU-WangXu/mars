from __future__ import annotations

import math
from dataclasses import dataclass

from .decoder import PositionField, ResidueOption


@dataclass
class ProposalRecord:
    candidate_id: str
    source: str
    source_group: str
    sequence: str
    ranking_score: float
    mars_score: float


def source_weight(source_group: str, source: str) -> float:
    if source_group == "learned":
        return 1.15
    if source_group == "heuristic_local":
        return 1.0
    if source == "manual":
        return 0.9
    return 1.0


def build_position_fields_from_proposals(
    wt_seq: str,
    proposal_rows: list[dict[str, object]],
    design_positions: list[int],
    position_to_index: dict[int, int],
    top_k_per_position: int = 4,
    max_rows_per_source: int = 16,
) -> list[PositionField]:
    # Pre-compute rank discounts to avoid repeated sqrt calculations
    _rank_discount_cache: dict[int, float] = {i: 1.0 / math.sqrt(i) for i in range(1, max_rows_per_source + 1)}
    # Cache tanh(x/4.0) with clamped input for common score ranges [-20, 20]
    _tanh_cache: dict[float, float] = {x: math.tanh(x / 4.0) for x in range(-20, 21)}

    # Pre-allocate position data structure with cached index lookups
    per_position: dict[int, dict[str, dict[str, object]]] = {}
    position_to_index_get = position_to_index.get
    for pos in design_positions:
        pos_int = int(pos)
        idx = position_to_index_get(pos_int, position_to_index_get(int(pos), 0))
        wt_aa = wt_seq[idx]
        per_position[pos_int] = {
            wt_aa: {
                "score": 0.0,
                "supporting_sources": set(),
            }
        }

    source_buckets: dict[str, list[ProposalRecord]] = {}
    for row in proposal_rows:
        record = ProposalRecord(
            candidate_id=str(row.get("candidate_id", "")),
            source=str(row.get("source", "")),
            source_group=str(row.get("source_group", "")),
            sequence=str(row.get("sequence", "")),
            ranking_score=float(row.get("ranking_score", row.get("mars_score", 0.0))),
            mars_score=float(row.get("mars_score", 0.0)),
        )
        source_buckets.setdefault(record.source, []).append(record)

    for source in sorted(source_buckets):
        bucket_rows = sorted(
            source_buckets[source],
            key=lambda item: (-item.ranking_score, -item.mars_score, item.candidate_id),
        )[: int(max_rows_per_source)]
        for rank_idx, record in enumerate(bucket_rows, start=1):
            proposal_weight = source_weight(record.source_group, record.source)
            rank_discount = _rank_discount_cache[rank_idx]
            # Use cached tanh values with fallback for out-of-range scores
            rs_clamped = max(-20.0, min(20.0, record.ranking_score))
            ms_clamped = max(-20.0, min(20.0, record.mars_score))
            tanh_rs = _tanh_cache.get(rs_clamped, math.tanh(rs_clamped / 4.0))
            tanh_ms = _tanh_cache.get(ms_clamped, math.tanh(ms_clamped / 4.0))
            score_signal = 0.55 * tanh_rs + 0.35 * tanh_ms
            contribution = proposal_weight * (0.8 + score_signal) * rank_discount
            seq = record.sequence
            source_name = record.source
            for position in design_positions:
                pos_int = int(position)
                idx = position_to_index[pos_int]
                aa = seq[idx]
                bucket = per_position[pos_int]
                option_state = bucket.get(aa)
                if option_state is None:
                    option_state = {"score": 0.0, "supporting_sources": set()}
                    bucket[aa] = option_state
                option_state["score"] = float(option_state["score"]) + contribution
                option_state["supporting_sources"].add(source_name)

    fields: list[PositionField] = []
    for position in sorted(per_position):
        idx = position_to_index[position]
        wt_residue = wt_seq[idx]
        position_data = per_position[position]
        ranked = sorted(
            position_data.items(),
            key=lambda item: (-float(item[1]["score"]), -len(item[1]["supporting_sources"]), item[0]),
        )[: int(top_k_per_position)]
        # Check if wt_residue is in top-k using list comprehension instead of set
        ranked_residues = [aa for aa, _ in ranked]
        if wt_residue not in ranked_residues:
            ranked.append((wt_residue, position_data[wt_residue]))
            ranked.sort(key=lambda item: (-float(item[1]["score"]), -len(item[1]["supporting_sources"]), item[0]))
            ranked = ranked[: int(top_k_per_position)]
        fields.append(
            PositionField(
                position=position,
                wt_residue=wt_residue,
                options=[
                    ResidueOption(
                        residue=aa,
                        score=round(float(option_state["score"]), 6),
                        supporting_sources=sorted(option_state["supporting_sources"]),
                        support_strength=round(float(option_state["score"]), 6),
                    )
                    for aa, option_state in ranked
                ],
            )
        )
    return fields


def serialize_position_fields(fields: list[PositionField]) -> list[dict[str, object]]:
    return [
        {
            "position": field.position,
            "wt_residue": field.wt_residue,
            "options": [
                {
                    "residue": option.residue,
                    "score": option.score,
                    "supporting_sources": option.supporting_sources or [],
                    "support_strength": option.support_strength,
                }
                for option in field.options
            ],
        }
        for field in fields
    ]
