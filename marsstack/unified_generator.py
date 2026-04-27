from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Any

from .decoder import PositionField, ResidueOption
from .multi_objective import (
    ObjectiveVector,
    MultiObjectiveCandidate,
    compute_multi_objective_scores,
    rank_candidates_multi_objective,
    WeightsConfig,
    create_default_csp_config,
    compute_pareto_front,
    crowding_distance,
)
from .ensemble_ranker import (
    EnsembleRanker,
    RankerConfig,
    CandidateFeatures,
    build_candidate_features,
    rank_candidates,
    analyze_feature_importance,
    FEATURE_NAMES,
)


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


@dataclass
class RankingContext:
    """Context for ensemble ranking of candidates."""

    wt_sequence: str = ""
    design_positions: list[int] = field(default_factory=list)
    oxidation_hotspots: list[int] = field(default_factory=list)
    flexible_positions: list[int] = field(default_factory=list)
    position_to_index: dict[int, int] = field(default_factory=dict)
    features: list[Any] = field(default_factory=list)  # ResidueFeature list
    profile: list[dict[str, float]] | None = None
    asr_profile: list[dict[str, float]] | None = None
    positive_profile: list[dict[str, float]] | None = None
    negative_profile: list[dict[str, float]] | None = None
    ranker_model_path: Path | None = None


def build_candidates_from_proposals(
    proposals: list[dict[str, Any]],
    context: RankingContext,
) -> list[CandidateFeatures]:
    """Build CandidateFeatures from proposal rows."""
    candidates = []
    for i, row in enumerate(proposals):
        sequence = str(row.get("sequence", ""))
        mutations_str = str(row.get("mutations", ""))
        mutations = mutations_str.split(";") if mutations_str and mutations_str != "WT" else []

        source = str(row.get("source", ""))
        source_group = str(row.get("source_group", ""))
        supporting_str = str(row.get("supporting_sources", ""))
        supporting_sources = supporting_str.split(";") if supporting_str else []

        header_metrics = {}
        header = str(row.get("header", ""))
        if header:
            for key in ["mpnn_score", "esm_recovery", "decoder_score"]:
                if key in header.lower():
                    try:
                        header_metrics[key] = float(header.lower().split(key)[1].split()[0])
                    except (IndexError, ValueError):
                        pass

        candidate = build_candidate_features(
            sequence=sequence,
            wt_sequence=context.wt_sequence,
            mutations=mutations,
            features=context.features,
            profile=context.profile,
            asr_profile=context.asr_profile,
            positive_profile=context.positive_profile,
            negative_profile=context.negative_profile,
            design_positions=context.design_positions,
            oxidation_hotspots=context.oxidation_hotspots,
            flexible_positions=context.flexible_positions,
            position_to_index=context.position_to_index,
            decoder_score=float(row.get("decoder_score", 0.0)),
            header_metrics=header_metrics,
            supporting_sources=supporting_sources,
            source=source,
            source_group=source_group,
            candidate_id=str(row.get("candidate_id", f"proposal_{i}")),
            fusion_score=float(row.get("fusion_score", 0.0)) if "fusion_score" in row else None,
        )

        label = None
        if "label" in row:
            label = float(row["label"])
        elif row.get("is_selected"):
            label = 1.0
        candidate.label = label

        candidates.append(candidate)

    return candidates


def rank_proposals_with_ensemble(
    proposals: list[dict[str, Any]],
    context: RankingContext,
    model_path: Path | None = None,
    config: RankerConfig | None = None,
    use_fallback: bool = True,
) -> list[tuple[dict[str, Any], float]]:
    """Rank proposals using ensemble learning ranker.

    Args:
        proposals: List of proposal dictionaries
        context: Ranking context with features and profiles
        model_path: Path to pre-trained model (optional)
        config: Ranker configuration (used if model_path is None)
        use_fallback: Use fallback scoring if no model available

    Returns:
        List of (proposal_dict, score) tuples sorted by score descending
    """
    candidates = build_candidates_from_proposals(proposals, context)

    ranker = None
    if model_path is not None and Path(model_path).exists():
        try:
            ranker = EnsembleRanker.load(model_path)
        except Exception:
            pass

    if ranker is None and config is not None:
        ranker = EnsembleRanker(config)

    ranked_candidates = rank_candidates(candidates, model=ranker, config=config, use_fallback=use_fallback)

    proposal_map = {str(row.get("candidate_id", f"proposal_{i}")): row for i, row in enumerate(proposals)}

    ranked_proposals = []
    for candidate, score in ranked_candidates:
        proposal = proposal_map.get(candidate.candidate_id, {})
        if not proposal:
            proposal = {
                "candidate_id": candidate.candidate_id,
                "sequence": candidate.sequence,
                "mutations": ";".join(candidate.mutations),
                "source": candidate.source,
                "source_group": candidate.source_group,
                "supporting_sources": ";".join(candidate.supporting_sources),
            }
        ranked_proposals.append((proposal, float(score)))

    return ranked_proposals


def analyze_proposal_features(
    proposals: list[dict[str, Any]],
    context: RankingContext,
    config: RankerConfig | None = None,
) -> dict[str, Any]:
    """Analyze feature importance for proposals.

    Returns feature importance analysis from training a temporary model.
    """
    candidates = build_candidates_from_proposals(proposals, context)

    has_labels = any(c.label is not None for c in candidates)
    if not has_labels:
        return {"error": "No labels available for feature importance analysis"}

    return analyze_feature_importance(candidates, config=config)


def train_ensemble_ranker(
    training_data: list[dict[str, Any]],
    context: RankingContext,
    config: RankerConfig | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Train an ensemble ranker on labeled proposal data.

    Args:
        training_data: List of proposal dicts with 'label' field (1.0 = selected, 0.0 = rejected)
        context: Ranking context with features and profiles
        config: Ranker configuration
        output_path: Path to save the trained model

    Returns:
        Dictionary with training metrics and model path
    """
    config = config or RankerConfig()
    candidates = build_candidates_from_proposals(training_data, context)

    has_labels = [c for c in candidates if c.label is not None]
    if len(has_labels) < 10:
        return {"error": "Need at least 10 labeled samples for training"}

    ranker = EnsembleRanker(config)

    cv_metrics = ranker.cross_validate(candidates)
    ranker.fit(candidates)

    model_path = output_path or config.model_dir / f"ranker_{config.model_type}.pkl"
    saved_path = ranker.save(model_path)

    importance = ranker.get_feature_importance(top_k=10)

    return {
        "model_path": str(saved_path),
        "cv_metrics": cv_metrics,
        "top_features": [{"feature": f, "importance": float(imp)} for f, imp in importance],
        "n_training_samples": len(candidates),
        "model_type": config.model_type,
    }
