from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._utils import split_semicolon
from .config import FusionRankerConfig


def mutation_tokens(text: Any) -> list[str]:
    if text is None:
        return []
    raw = str(text)
    if raw == "WT":
        return []
    return split_semicolon(raw)


def mutation_overlap_ratio(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    return float(len(set_a & set_b) / max(1, len(set_a | set_b)))


def apply_target_score_calibration(
    rows: list[dict[str, Any]],
    raw_scores: np.ndarray,
    cfg: FusionRankerConfig,
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    mean = float(raw_scores.mean()) if len(raw_scores) else 0.0
    std = float(raw_scores.std()) if len(raw_scores) else 1.0
    if std < 1e-6:
        std = 1.0
    z_scores = (raw_scores - mean) / std
    bounded_scores = float(cfg.target_score_scale) * np.tanh(z_scores / max(1e-6, float(cfg.target_z_temperature)))

    engineered_prior_rows = [
        row for row in rows
        if str(row.get("source", "")) != "fusion_decoder" and str(row.get("mutations", "WT")) != "WT"
    ]
    if not engineered_prior_rows:
        engineered_prior_rows = [row for row in rows if str(row.get("source", "")) != "fusion_decoder"]
    best_prior_row = max(
        engineered_prior_rows,
        key=lambda item: (float(item.get("mars_score", 0.0)), float(item.get("score_manual", 0.0)), -len(mutation_tokens(item.get("mutations", "WT")))),
    ) if engineered_prior_rows else None
    best_prior_mars = float(best_prior_row.get("mars_score", 0.0)) if best_prior_row else 0.0
    best_prior_tokens = mutation_tokens(best_prior_row.get("mutations", "WT")) if best_prior_row else []

    calibrated_scores = bounded_scores.copy()
    diagnostics: list[dict[str, float | str]] = []

    for idx, row in enumerate(rows):
        penalty = 0.0
        reasons: list[str] = []
        support_count = float(len(split_semicolon(row.get("supporting_sources", ""))))
        mars_score = float(row.get("mars_score", 0.0))
        source = str(row.get("source", ""))
        tokens = mutation_tokens(row.get("mutations", "WT"))
        calibrated_bonus = float(cfg.mars_calibration_weight) * math.tanh(mars_score / 4.0)

        if mars_score < 0.0:
            penalty += float(cfg.negative_mars_penalty_base) + float(cfg.negative_mars_penalty_scale) * abs(mars_score)
            reasons.append("negative_mars_penalty")

        if source == "fusion_decoder":
            over_support = max(0.0, support_count - float(cfg.decoder_support_cap))
            if over_support > 0:
                penalty += float(cfg.decoder_consensus_penalty) * over_support
                reasons.append("decoder_consensus_cap")
            prior_gap = best_prior_mars - mars_score
            if prior_gap > float(cfg.decoder_prior_gap_tolerance):
                penalty += float(cfg.decoder_prior_gap_penalty) * (prior_gap - float(cfg.decoder_prior_gap_tolerance))
                reasons.append("decoder_prior_gap")
            overlap = mutation_overlap_ratio(tokens, best_prior_tokens)
            if best_prior_tokens and overlap < 0.34:
                penalty += 0.9 * (0.34 - overlap)
                reasons.append("decoder_prior_divergence")

        if source != "fusion_decoder":
            prior_gap = best_prior_mars - mars_score
            if prior_gap > float(cfg.engineering_prior_gap_tolerance) and best_prior_tokens:
                overlap = mutation_overlap_ratio(tokens, best_prior_tokens)
                penalty += float(cfg.engineering_prior_gap_penalty) * max(0.0, prior_gap - float(cfg.engineering_prior_gap_tolerance)) * (1.0 - overlap)
                reasons.append("engineering_prior_gap")

        if str(row.get("mutations", "WT")) == "WT":
            prior_gap = best_prior_mars - mars_score
            if prior_gap > float(cfg.wt_prior_gap_tolerance):
                penalty += float(cfg.wt_static_penalty) + float(cfg.wt_prior_gap_penalty) * (prior_gap - float(cfg.wt_prior_gap_tolerance))
                reasons.append("wt_prior_gap")

        calibrated_scores[idx] = bounded_scores[idx] + calibrated_bonus - penalty
        diagnostics.append(
            {
                "ranking_score_raw": round(float(raw_scores[idx]), 6),
                "ranking_score_z": round(float(z_scores[idx]), 6),
                "ranking_score_bounded": round(float(bounded_scores[idx]), 6),
                "ranking_score_mars_calibrated": round(float(calibrated_bonus), 6),
                "ranking_penalty": round(float(penalty), 6),
                "ranking_penalty_reasons": ";".join(reasons),
            }
        )

    return calibrated_scores, diagnostics
