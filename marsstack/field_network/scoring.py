from __future__ import annotations

from dataclasses import dataclass

from ..mars_score import mutation_list, score_candidate
from ..structure_features import ResidueFeature


@dataclass
class ScoringInputs:
    wt_seq: str
    features: list[ResidueFeature]
    oxidation_hotspots: list[int]
    flexible_positions: list[int]
    profile: list[dict[str, float]] | None
    asr_profile: list[dict[str, float]] | None
    family_positive_profile: list[dict[str, float]] | None
    family_negative_profile: list[dict[str, float]] | None
    manual_preferred: dict[int, dict[str, float]]
    design_positions: list[int]
    term_weights: dict[str, float]
    position_to_index: dict[int, int]
    evolution_position_weights: dict[int, float]
    residue_numbers: list[int]
    profile_prior_scale: float
    asr_prior_scale: float
    family_prior_scale: float
    topic_name: str | None
    topic_cfg: dict[str, object] | None


def score_candidate_rows(
    candidates: list[dict[str, object]],
    scoring: ScoringInputs,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in candidates:
        sequence = str(entry["sequence"])
        result = score_candidate(
            wt_seq=scoring.wt_seq,
            seq=sequence,
            features=scoring.features,
            oxidation_hotspots=scoring.oxidation_hotspots,
            flexible_positions=scoring.flexible_positions,
            profile=scoring.profile,
            asr_profile=scoring.asr_profile,
            family_positive_profile=scoring.family_positive_profile,
            family_negative_profile=scoring.family_negative_profile,
            manual_preferred=scoring.manual_preferred,
            evolution_positions=scoring.design_positions,
            mutable_positions=scoring.design_positions,
            term_weights=scoring.term_weights,
            position_to_index=scoring.position_to_index,
            evolution_position_weights=scoring.evolution_position_weights,
            residue_numbers=scoring.residue_numbers,
            profile_prior_scale=scoring.profile_prior_scale,
            asr_prior_scale=scoring.asr_prior_scale,
            family_prior_scale=scoring.family_prior_scale,
            topic_name=scoring.topic_name,
            topic_cfg=scoring.topic_cfg,
        )
        rows.append(
            {
                "candidate_id": entry["candidate_id"],
                "source": entry["source"],
                "source_group": entry.get("source_group", ""),
                "supporting_sources": ";".join(entry.get("supporting_sources", [str(entry["source"])])),
                "mutations": ";".join(mutation_list(scoring.wt_seq, sequence, residue_numbers=scoring.residue_numbers)) or "WT",
                "mars_score": result.total,
                "notes": ";".join(result.notes),
                "sequence": sequence,
                "header": entry.get("header", ""),
                "score_oxidation": result.components["oxidation"],
                "score_surface": result.components["surface"],
                "score_manual": result.components["manual"],
                "score_evolution": result.components["evolution"],
                "score_burden": result.components["burden"],
                "score_topic_sequence": result.components["topic_sequence"],
                "score_topic_structure": result.components["topic_structure"],
                "score_topic_evolution": result.components["topic_evolution"],
            }
        )
    return rows
