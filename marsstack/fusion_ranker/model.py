from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np

from .config import FusionRankerConfig, TrainingCorpus
from .constants import LINEAR_GROUP_RULES
from .features import feature_group


def fm_score(features: np.ndarray, bias: float, linear: np.ndarray, factors: np.ndarray) -> np.ndarray:
    linear_term = bias + features @ linear
    summed = features @ factors
    squared_sum = summed * summed
    square_then_sum = (features * features) @ (factors * factors)
    interaction = 0.5 * np.sum(squared_sum - square_then_sum, axis=1)
    return linear_term + interaction


def fm_gradients(
    features: np.ndarray,
    score_grad: np.ndarray,
    linear: np.ndarray,
    factors: np.ndarray,
    reg_lambda: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    grad_bias = float(score_grad.sum())
    grad_linear = features.T @ score_grad + reg_lambda * linear
    squared_features = features * features
    factor_sum = features @ factors
    factor_base = squared_features.T @ score_grad
    grad_factors = np.empty_like(factors)
    for k in range(factors.shape[1]):
        grad_factors[:, k] = features.T @ (score_grad * factor_sum[:, k]) - factors[:, k] * factor_base
    grad_factors += reg_lambda * factors
    return grad_bias, grad_linear, grad_factors


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def train_factor_ranker(corpus: TrainingCorpus, cfg: FusionRankerConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(int(cfg.seed))
    num_features = corpus.features.shape[1]
    linear = np.zeros(num_features, dtype=float)
    factors = rng.normal(0.0, 0.05, size=(num_features, int(cfg.latent_dim)))
    bias = 0.0

    m_bias = 0.0
    v_bias = 0.0
    m_linear = np.zeros_like(linear)
    v_linear = np.zeros_like(linear)
    m_factors = np.zeros_like(factors)
    v_factors = np.zeros_like(factors)

    history: list[dict[str, float]] = []
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    for epoch in range(1, int(cfg.epochs) + 1):
        pred = fm_score(corpus.features, bias, linear, factors)
        grad = np.zeros_like(pred)

        residual = pred - corpus.regression_target
        grad += (2.0 * float(cfg.regression_weight) / max(1, len(pred))) * residual

        losses = {
            "regression": float(np.mean(residual * residual)),
            "primary_pair": 0.0,
            "native_pair": 0.0,
            "consensus_pair": 0.0,
        }

        def apply_pairs(
            pairs: list[tuple[int, int, float]],
            pair_weight: float,
            loss_key: str,
        ) -> None:
            if not pairs or pair_weight <= 0:
                return
            idx_i = np.array([i for i, _, _ in pairs], dtype=int)
            idx_j = np.array([j for _, j, _ in pairs], dtype=int)
            pair_w = np.array([w for _, _, w in pairs], dtype=float)
            diff = pred[idx_i] - pred[idx_j]
            sig = _sigmoid(-diff)
            norm = float(pair_weight / max(1, len(pairs)))
            grad_delta = -norm * pair_w * sig
            np.add.at(grad, idx_i, grad_delta)
            np.add.at(grad, idx_j, -grad_delta)
            losses[loss_key] = float(np.mean(np.logaddexp(0.0, -diff) * pair_w))

        apply_pairs(corpus.rank_pairs, float(cfg.primary_pair_weight), "primary_pair")
        apply_pairs(corpus.native_pairs, float(cfg.native_pair_weight), "native_pair")
        apply_pairs(corpus.consensus_pairs, float(cfg.consensus_pair_weight), "consensus_pair")

        grad_bias, grad_linear, grad_factors = fm_gradients(
            corpus.features,
            grad,
            linear=linear,
            factors=factors,
            reg_lambda=float(cfg.reg_lambda),
        )

        m_bias = beta1 * m_bias + (1.0 - beta1) * grad_bias
        v_bias = beta2 * v_bias + (1.0 - beta2) * (grad_bias * grad_bias)
        bias -= float(cfg.learning_rate) * (m_bias / (1.0 - beta1 ** epoch)) / (math.sqrt(v_bias / (1.0 - beta2 ** epoch)) + epsilon)

        m_linear = beta1 * m_linear + (1.0 - beta1) * grad_linear
        v_linear = beta2 * v_linear + (1.0 - beta2) * (grad_linear * grad_linear)
        linear -= float(cfg.learning_rate) * (m_linear / (1.0 - beta1 ** epoch)) / (np.sqrt(v_linear / (1.0 - beta2 ** epoch)) + epsilon)

        m_factors = beta1 * m_factors + (1.0 - beta1) * grad_factors
        v_factors = beta2 * v_factors + (1.0 - beta2) * (grad_factors * grad_factors)
        factors -= float(cfg.learning_rate) * (m_factors / (1.0 - beta1 ** epoch)) / (np.sqrt(v_factors / (1.0 - beta2 ** epoch)) + epsilon)

        if epoch == 1 or epoch == int(cfg.epochs) or epoch % 25 == 0:
            history.append(
                {
                    "epoch": float(epoch),
                    "regression_loss": round(losses["regression"], 6),
                    "primary_pair_loss": round(losses["primary_pair"], 6),
                    "native_pair_loss": round(losses["native_pair"], 6),
                    "consensus_pair_loss": round(losses["consensus_pair"], 6),
                }
            )

    model_payload = {
        "bias": bias,
        "linear": linear.tolist(),
        "factors": factors.tolist(),
        "feature_names": corpus.feature_names,
        "means": corpus.means.tolist(),
        "stds": corpus.stds.tolist(),
        "config": asdict(cfg),
    }
    training_summary = {
        "training_targets": sorted(set(corpus.sample_targets)),
        "training_target_count": len(set(corpus.sample_targets)),
        "training_example_count": int(corpus.features.shape[0]),
        "feature_count": len(corpus.feature_names),
        "primary_pair_count": len(corpus.rank_pairs),
        "native_pair_count": len(corpus.native_pairs),
        "consensus_pair_count": len(corpus.consensus_pairs),
        "history": history,
    }
    return model_payload, training_summary


def score_feature_matrix(model_payload: dict[str, Any], features: np.ndarray) -> np.ndarray:
    return fm_score(
        features,
        bias=float(model_payload["bias"]),
        linear=np.array(model_payload["linear"], dtype=float),
        factors=np.array(model_payload["factors"], dtype=float),
    )


def explain_rows(
    model_payload: dict[str, Any],
    raw_features: np.ndarray,
    standardized_features: np.ndarray,
) -> list[dict[str, float]]:
    linear = np.array(model_payload["linear"], dtype=float)
    factors = np.array(model_payload["factors"], dtype=float)
    feature_names = list(model_payload["feature_names"])
    grouped_linear = [dict.fromkeys(LINEAR_GROUP_RULES.keys(), 0.0) for _ in range(len(standardized_features))]

    linear_contrib = standardized_features * linear
    for col_idx, feature_name in enumerate(feature_names):
        group = feature_group(feature_name)
        for row_idx in range(len(standardized_features)):
            grouped_linear[row_idx][group] = grouped_linear[row_idx].get(group, 0.0) + float(linear_contrib[row_idx, col_idx])

    interaction_total = 0.5 * np.sum(
        (standardized_features @ factors) ** 2 - (standardized_features * standardized_features) @ (factors * factors),
        axis=1,
    )
    explanations: list[dict[str, float]] = []
    for row_idx, group_scores in enumerate(grouped_linear):
        explanation = {f"fusion_linear_{group}": round(score, 6) for group, score in group_scores.items()}
        explanation["fusion_interaction"] = round(float(interaction_total[row_idx]), 6)
        explanation["fusion_raw_feature_norm"] = round(float(np.linalg.norm(raw_features[row_idx])), 6)
        explanations.append(explanation)
    return explanations
