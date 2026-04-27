from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .calibration import apply_target_score_calibration
from .config import FusionRankerConfig, OutputContext
from .constants import LINEAR_GROUP_RULES
from .corpus import build_corpus, load_training_tables
from .features import build_feature_matrix, feature_group
from .model import explain_rows, score_feature_matrix, train_factor_ranker


def rank_rows_with_model(
    rows: list[dict[str, Any]],
    protein_name: str,
    feature_summary: dict[str, Any],
    profile_summary: dict[str, Any],
    model_payload: dict[str, Any],
    cfg: FusionRankerConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = cfg or FusionRankerConfig()
    context = OutputContext(
        protein=protein_name,
        feature_summary=feature_summary,
        profile_summary=profile_summary,
    )
    raw_features = build_feature_matrix(rows, context, feature_names=list(model_payload["feature_names"]), cfg=cfg)
    means = np.array(model_payload["means"], dtype=float)
    stds = np.array(model_payload["stds"], dtype=float)
    standardized = (raw_features - means) / stds
    fusion_scores = score_feature_matrix(model_payload, standardized)
    calibrated_scores, diagnostics = apply_target_score_calibration(rows, fusion_scores, cfg)
    explanations = explain_rows(model_payload, raw_features=raw_features, standardized_features=standardized)

    scored_rows: list[dict[str, Any]] = []
    for row, raw_score, calibrated_score, explanation, diagnostic in zip(rows, fusion_scores.tolist(), calibrated_scores.tolist(), explanations, diagnostics):
        updated = dict(row)
        updated["ranking_score"] = round(float(calibrated_score), 6)
        updated["fusion_score"] = round(float(raw_score), 6)
        updated["ranking_model"] = "learned_fusion_v2_0"
        updated.update(explanation)
        updated.update(diagnostic)
        scored_rows.append(updated)

    scored_rows.sort(
        key=lambda item: (
            -float(item["ranking_score"]),
            -float(item.get("mars_score", 0.0)),
            str(item.get("mutations", "")),
            str(item.get("source", "")),
        )
    )
    return scored_rows


def apply_learned_fusion_ranking(
    rows: list[dict[str, Any]],
    protein_name: str,
    feature_summary: dict[str, Any],
    profile_summary: dict[str, Any],
    outputs_root: Path,
    cfg_raw: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    cfg = FusionRankerConfig.from_dict(cfg_raw)
    default_rows = [dict(row) for row in rows]
    for row in default_rows:
        row["ranking_score"] = float(row.get("mars_score", 0.0))
        row["ranking_model"] = "mars_score_v0"
        row["fusion_score"] = float(row.get("mars_score", 0.0))
        row["ranking_score_raw"] = float(row.get("mars_score", 0.0))
        row["ranking_score_z"] = 0.0
        row["ranking_score_bounded"] = float(row.get("mars_score", 0.0))
        row["ranking_score_mars_calibrated"] = 0.0
        row["ranking_penalty"] = 0.0
        row["ranking_penalty_reasons"] = ""
        row["fusion_linear_generator"] = 0.0
        row["fusion_linear_structure"] = 0.0
        row["fusion_linear_evolution"] = 0.0
        row["fusion_linear_consensus"] = 0.0
        row["fusion_linear_topic"] = 0.0
        row["fusion_linear_context"] = 0.0
        row["fusion_linear_misc"] = 0.0
        row["fusion_interaction"] = 0.0
        row["fusion_raw_feature_norm"] = 0.0
    default_rows.sort(
        key=lambda item: (
            -float(item["ranking_score"]),
            -float(item.get("mars_score", 0.0)),
            str(item.get("mutations", "")),
            str(item.get("source", "")),
        )
    )

    if not cfg.enabled:
        return default_rows, {"ranking_model": "mars_score_v0", "reason": "disabled"}, None

    training_tables = load_training_tables(outputs_root, exclude_protein=protein_name)
    if len(training_tables) < int(cfg.min_training_targets):
        return (
            default_rows,
            {
                "ranking_model": "mars_score_v0",
                "reason": "insufficient_training_targets",
                "training_target_count": len(training_tables),
            },
            None,
        )

    corpus = build_corpus(training_tables, cfg)
    if corpus is None:
        return (
            default_rows,
            {
                "ranking_model": "mars_score_v0",
                "reason": "insufficient_training_candidates",
            },
            None,
        )

    model_payload, training_summary = train_factor_ranker(corpus, cfg)
    scored_rows = rank_rows_with_model(
        rows=rows,
        protein_name=protein_name,
        feature_summary=feature_summary,
        profile_summary=profile_summary,
        model_payload=model_payload,
        cfg=cfg,
    )
    training_summary.update(
        {
            "ranking_model": "learned_fusion_v2_0",
            "feature_groups": {group: 0 for group in LINEAR_GROUP_RULES},
        }
    )
    for feature_name in model_payload["feature_names"]:
        group = feature_group(feature_name)
        training_summary["feature_groups"][group] = training_summary["feature_groups"].get(group, 0) + 1
    return scored_rows, training_summary, model_payload
