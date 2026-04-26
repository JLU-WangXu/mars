from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_ORDER = ["manual", "baseline_mpnn", "mars_mpnn", "esm_if", "local_proposal", "fusion_decoder", "neural_decoder"]
SOURCE_GROUP_ORDER = ["manual_control", "learned", "heuristic_local"]
NUMERIC_ROW_COLUMNS = [
    "mars_score",
    "score_oxidation",
    "score_surface",
    "score_manual",
    "score_evolution",
    "score_burden",
    "score_topic_sequence",
    "score_topic_structure",
    "score_topic_evolution",
]
NOTE_PREFIXES = [
    "hardens_hotspot_",
    "keeps_hotspot_",
    "bad_hotspot_choice_",
    "surface_hydration_",
    "sticky_surface_",
    "manual_bias_",
    "topic_seq_",
    "topic_struct_",
    "topic_evo_",
]
LINEAR_GROUP_RULES = {
    "generator": (
        "source__",
        "source_group__",
        "header__",
        "native__",
    ),
    "structure": (
        "score_oxidation",
        "score_surface",
        "score_burden",
        "mutation_",
        "hotspot_",
        "flex_",
        "note__hardens_hotspot",
        "note__keeps_hotspot",
        "note__bad_hotspot_choice",
        "note__surface_hydration",
        "note__sticky_surface",
        "ctx__num_design_positions",
        "ctx__num_oxidation_hotspots",
        "ctx__num_flexible_positions",
    ),
    "evolution": (
        "score_evolution",
        "note__evolution_prior",
        "note__asr_prior",
        "note__family_evolution_prior",
        "note__template_weighted_evolution",
        "ctx__accepted_homologs",
        "ctx__accepted_asr",
        "ctx__accepted_positive",
        "ctx__accepted_negative",
        "ctx__mean_coverage",
        "ctx__mean_asr_coverage",
        "ctx__mean_positive_coverage",
        "ctx__mean_negative_coverage",
        "ctx__family_prior_enabled",
        "ctx__asr_prior_enabled",
        "ctx__template_weighting_enabled",
    ),
    "consensus": (
        "support_",
        "consensus_",
    ),
    "topic": (
        "score_topic_",
        "note__topic_",
        "ctx__topic_enabled",
    ),
    "context": (
        "ctx__",
    ),
}
HEADER_FLOAT_RE = re.compile(r"([A-Za-z_]+)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


@dataclass
class FusionRankerConfig:
    enabled: bool = True
    latent_dim: int = 8
    epochs: int = 180
    learning_rate: float = 0.035
    reg_lambda: float = 0.0005
    regression_weight: float = 1.0
    primary_pair_weight: float = 1.4
    native_pair_weight: float = 0.30
    consensus_pair_weight: float = 0.18
    pair_margin: float = 0.15
    min_training_targets: int = 4
    min_training_candidates: int = 80
    rank_pair_offsets: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    native_pair_offsets: tuple[int, ...] = (1, 2, 4)
    consensus_pair_offsets: tuple[int, ...] = (1, 2, 4)
    max_pairs_per_target: int = 240
    seed: int = 13
    decoder_generator_feature_scale: float = 0.18
    decoder_consensus_feature_scale: float = 0.20
    target_score_scale: float = 4.0
    target_z_temperature: float = 1.0
    decoder_support_cap: float = 3.0
    decoder_consensus_penalty: float = 0.45
    decoder_prior_gap_tolerance: float = 0.8
    decoder_prior_gap_penalty: float = 1.25
    wt_prior_gap_tolerance: float = 0.5
    wt_prior_gap_penalty: float = 1.2
    wt_static_penalty: float = 1.5
    mars_calibration_weight: float = 1.5
    negative_mars_penalty_base: float = 1.0
    negative_mars_penalty_scale: float = 0.35
    engineering_prior_gap_penalty: float = 0.45
    engineering_prior_gap_tolerance: float = 1.5

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "FusionRankerConfig":
        if not raw:
            return cls()
        cfg = cls()
        for key, value in raw.items():
            if not hasattr(cfg, key):
                continue
            if key.endswith("_offsets"):
                setattr(cfg, key, tuple(int(x) for x in value))
            else:
                setattr(cfg, key, value)
        return cfg


@dataclass
class OutputContext:
    protein: str
    feature_summary: dict[str, Any]
    profile_summary: dict[str, Any]


@dataclass
class TrainingCorpus:
    features: np.ndarray
    feature_names: list[str]
    means: np.ndarray
    stds: np.ndarray
    regression_target: np.ndarray
    sample_targets: list[str]
    rank_pairs: list[tuple[int, int, float]]
    native_pairs: list[tuple[int, int, float]]
    consensus_pairs: list[tuple[int, int, float]]


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def split_semicolon(text: Any) -> list[str]:
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []
    return [item for item in raw.split(";") if item]


def parse_header_metrics(source: str, header: Any) -> dict[str, float]:
    metrics = {
        "header__temperature": 0.0,
        "header__sample": 0.0,
        "header__mpnn_score": 0.0,
        "header__mpnn_global_score": 0.0,
        "header__mpnn_seq_recovery": 0.0,
        "header__esm_recovery": 0.0,
        "header__decoder_score": 0.0,
        "header__decoder_mutation_count": 0.0,
        "native__confidence": 0.0,
        "native__available": 0.0,
    }
    if header is None:
        return metrics
    text = str(header).strip()
    if not text:
        return metrics

    parsed: dict[str, float] = {}
    for key, raw_value in HEADER_FLOAT_RE.findall(text):
        parsed[key.lower()] = safe_float(raw_value)

    if "temperature" in parsed:
        metrics["header__temperature"] = parsed["temperature"]
    if "t" in parsed:
        metrics["header__temperature"] = parsed["t"]
    if "sample" in parsed:
        metrics["header__sample"] = parsed["sample"]
    if "score" in parsed:
        metrics["header__mpnn_score"] = parsed["score"]
    if "global_score" in parsed:
        metrics["header__mpnn_global_score"] = parsed["global_score"]
    if "seq_recovery" in parsed:
        metrics["header__mpnn_seq_recovery"] = parsed["seq_recovery"]
    if "recovery" in parsed:
        metrics["header__esm_recovery"] = parsed["recovery"]
    if "decoder_score" in parsed:
        metrics["header__decoder_score"] = parsed["decoder_score"]
    if "mutation_count" in parsed:
        metrics["header__decoder_mutation_count"] = parsed["mutation_count"]

    native_confidence = None
    if source in {"baseline_mpnn", "mars_mpnn"}:
        if metrics["header__mpnn_seq_recovery"] > 0:
            native_confidence = metrics["header__mpnn_seq_recovery"]
        elif metrics["header__mpnn_global_score"] != 0:
            native_confidence = -metrics["header__mpnn_global_score"]
        elif metrics["header__mpnn_score"] != 0:
            native_confidence = -metrics["header__mpnn_score"]
    elif source == "esm_if" and metrics["header__esm_recovery"] > 0:
        native_confidence = metrics["header__esm_recovery"]

    if native_confidence is not None:
        metrics["native__confidence"] = float(native_confidence)
        metrics["native__available"] = 1.0
    return metrics


def note_features(notes: Any) -> dict[str, float]:
    items = split_semicolon(notes)
    features: dict[str, float] = {
        "note__evolution_prior": 0.0,
        "note__asr_prior": 0.0,
        "note__family_evolution_prior": 0.0,
        "note__template_weighted_evolution": 0.0,
        "note__low_burden": 0.0,
    }
    for prefix in NOTE_PREFIXES:
        tag = prefix.rstrip("_")
        features[f"note__{tag}_count"] = 0.0

    for item in items:
        if item in features:
            features[item] = 1.0
            continue
        for prefix in NOTE_PREFIXES:
            if item.startswith(prefix):
                tag = prefix.rstrip("_")
                features[f"note__{tag}_count"] += 1.0
                break
    return features


def source_features(source: str, source_group: str) -> dict[str, float]:
    out = {}
    for item in SOURCE_ORDER:
        out[f"source__{item}"] = 1.0 if source == item else 0.0
    for item in SOURCE_GROUP_ORDER:
        out[f"source_group__{item}"] = 1.0 if source_group == item else 0.0
    return out


def support_features(supporting_sources: Any) -> dict[str, float]:
    sources = split_semicolon(supporting_sources)
    unique_sources = list(dict.fromkeys(sources))
    learned = {"baseline_mpnn", "mars_mpnn", "esm_if"}
    heuristic = {"local_proposal"}
    manual = {"manual"}
    learned_count = sum(1 for src in unique_sources if src in learned)
    heuristic_count = sum(1 for src in unique_sources if src in heuristic)
    manual_count = sum(1 for src in unique_sources if src in manual)
    return {
        "support__count": float(len(unique_sources)),
        "support__learned_count": float(learned_count),
        "support__heuristic_count": float(heuristic_count),
        "support__manual_count": float(manual_count),
        "consensus__multi_source": 1.0 if len(unique_sources) > 1 else 0.0,
        "consensus__learned_multi_source": 1.0 if learned_count > 1 else 0.0,
    }


def mutation_features(mutations: Any, context: OutputContext) -> dict[str, float]:
    mut_items = [] if str(mutations or "WT") == "WT" else split_semicolon(mutations)
    design_positions = [int(x) for x in context.feature_summary.get("design_positions", [])]
    num_design_positions = max(1, len(design_positions))
    oxidation_hotspots = set(int(x) for x in context.feature_summary.get("oxidation_hotspots", []))
    flexible_positions = set(int(x) for x in context.feature_summary.get("flexible_surface_positions", []))

    mutated_positions: list[int] = []
    for item in mut_items:
        digits = "".join(ch for ch in item if ch.isdigit())
        if digits:
            mutated_positions.append(int(digits))

    hotspot_hits = sum(1 for pos in mutated_positions if pos in oxidation_hotspots)
    flexible_hits = sum(1 for pos in mutated_positions if pos in flexible_positions)
    return {
        "mutation__count": float(len(mut_items)),
        "mutation__fraction_design": float(len(mut_items) / num_design_positions),
        "hotspot__mutated_count": float(hotspot_hits),
        "hotspot__mutated_fraction": float(hotspot_hits / num_design_positions),
        "flex__mutated_count": float(flexible_hits),
        "flex__mutated_fraction": float(flexible_hits / num_design_positions),
        "mutation__is_wt": 1.0 if len(mut_items) == 0 else 0.0,
    }


def context_features(context: OutputContext) -> dict[str, float]:
    feature_summary = context.feature_summary
    profile_summary = context.profile_summary
    return {
        "ctx__num_design_positions": float(len(feature_summary.get("design_positions", []))),
        "ctx__num_oxidation_hotspots": float(len(feature_summary.get("oxidation_hotspots", []))),
        "ctx__num_flexible_positions": float(len(feature_summary.get("flexible_surface_positions", []))),
        "ctx__accepted_homologs": safe_float(profile_summary.get("accepted_homologs")),
        "ctx__accepted_asr": safe_float(profile_summary.get("accepted_asr")),
        "ctx__accepted_positive": safe_float(profile_summary.get("accepted_positive")),
        "ctx__accepted_negative": safe_float(profile_summary.get("accepted_negative")),
        "ctx__mean_coverage": safe_float(profile_summary.get("mean_coverage")),
        "ctx__mean_asr_coverage": safe_float(profile_summary.get("mean_asr_coverage")),
        "ctx__mean_positive_coverage": safe_float(profile_summary.get("mean_positive_coverage")),
        "ctx__mean_negative_coverage": safe_float(profile_summary.get("mean_negative_coverage")),
        "ctx__family_prior_enabled": 1.0 if profile_summary.get("family_prior_enabled", False) else 0.0,
        "ctx__asr_prior_enabled": 1.0 if profile_summary.get("asr_prior_enabled", False) else 0.0,
        "ctx__template_weighting_enabled": 1.0 if profile_summary.get("template_weighting_enabled", False) else 0.0,
        "ctx__topic_enabled": 1.0
        if any(safe_float(profile_summary.get(key)) != 0.0 for key in ("topic_score_sequence", "topic_score_structure", "topic_score_evolution"))
        else 0.0,
    }


def build_feature_dict(row: dict[str, Any], context: OutputContext, cfg: FusionRankerConfig | None = None) -> dict[str, float]:
    features: dict[str, float] = {}
    source = str(row.get("source", ""))
    source_group = str(row.get("source_group", ""))
    cfg = cfg or FusionRankerConfig()
    for column in NUMERIC_ROW_COLUMNS:
        features[column] = safe_float(row.get(column))
    features.update(source_features(source, source_group))
    features.update(note_features(row.get("notes", "")))
    features.update(parse_header_metrics(source, row.get("header", "")))
    features.update(support_features(row.get("supporting_sources", "")))
    features.update(mutation_features(row.get("mutations", "WT"), context))
    features.update(context_features(context))
    if source == "fusion_decoder":
        generator_scale = float(cfg.decoder_generator_feature_scale)
        consensus_scale = float(cfg.decoder_consensus_feature_scale)
        for key in list(features):
            if key.startswith("source__fusion_decoder") or key.startswith("header__decoder_"):
                features[key] *= generator_scale
            elif key.startswith("support__") or key.startswith("consensus__"):
                features[key] *= consensus_scale
    return features


def feature_group(feature_name: str) -> str:
    for group, prefixes in LINEAR_GROUP_RULES.items():
        if any(feature_name.startswith(prefix) for prefix in prefixes):
            return group
    return "misc"


def load_context(output_dir: Path) -> OutputContext | None:
    ranked_path = output_dir / "combined_ranked_candidates.csv"
    feature_path = output_dir / "feature_summary.json"
    profile_path = output_dir / "profile_summary.json"
    if not (ranked_path.exists() and feature_path.exists() and profile_path.exists()):
        return None
    feature_summary = json.loads(feature_path.read_text(encoding="utf-8"))
    profile_summary = json.loads(profile_path.read_text(encoding="utf-8"))
    protein = str(feature_summary.get("protein", output_dir.name.replace("_pipeline", "")))
    return OutputContext(protein=protein, feature_summary=feature_summary, profile_summary=profile_summary)


def load_training_tables(outputs_root: Path, exclude_protein: str) -> list[tuple[pd.DataFrame, OutputContext]]:
    datasets: list[tuple[pd.DataFrame, OutputContext]] = []
    for output_dir in sorted(outputs_root.glob("*_pipeline")):
        context = load_context(output_dir)
        if context is None or context.protein == exclude_protein:
            continue
        ranked_path = output_dir / "combined_ranked_candidates.csv"
        df = pd.read_csv(ranked_path)
        if df.empty or "mars_score" not in df.columns:
            continue
        datasets.append((df, context))
    return datasets


def standardize_target(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        return values - mean
    return (values - mean) / std


def sorted_pair_indices(order: np.ndarray, scores: np.ndarray, offsets: tuple[int, ...], max_pairs: int, min_margin: float) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    for rank_idx, idx_a in enumerate(order):
        for offset in offsets:
            idx_b_pos = rank_idx + offset
            if idx_b_pos >= len(order):
                continue
            idx_b = int(order[idx_b_pos])
            gap = float(scores[idx_a] - scores[idx_b])
            if gap <= min_margin:
                continue
            pairs.append((int(idx_a), idx_b, min(3.0, gap)))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def build_corpus(datasets: list[tuple[pd.DataFrame, OutputContext]], cfg: FusionRankerConfig) -> TrainingCorpus | None:
    feature_rows: list[dict[str, float]] = []
    regression_targets: list[float] = []
    sample_targets: list[str] = []
    rank_pairs: list[tuple[int, int, float]] = []
    native_pairs: list[tuple[int, int, float]] = []
    consensus_pairs: list[tuple[int, int, float]] = []
    running_offset = 0

    for df, context in datasets:
        row_features = [build_feature_dict(row, context, cfg=cfg) for row in df.to_dict(orient="records")]
        y = standardize_target(df["mars_score"].to_numpy(dtype=float))
        feature_rows.extend(row_features)
        regression_targets.extend(float(v) for v in y.tolist())
        sample_targets.extend([context.protein] * len(df))

        primary_order = np.argsort(-y)
        rank_pairs.extend(
            [
                (running_offset + i, running_offset + j, weight)
                for i, j, weight in sorted_pair_indices(
                    primary_order,
                    y,
                    offsets=cfg.rank_pair_offsets,
                    max_pairs=cfg.max_pairs_per_target,
                    min_margin=cfg.pair_margin,
                )
            ]
        )

        source_groups = df["source"].astype(str).tolist()
        native_conf = np.array(
            [build_feature_dict(row, context, cfg=cfg)["native__confidence"] for row in df.to_dict(orient="records")],
            dtype=float,
        )
        native_available = np.array(
            [build_feature_dict(row, context, cfg=cfg)["native__available"] for row in df.to_dict(orient="records")],
            dtype=float,
        )
        for source in SOURCE_ORDER:
            local_idx = np.where((np.array(source_groups) == source) & (native_available > 0))[0]
            if len(local_idx) < 2:
                continue
            local_values = native_conf[local_idx]
            local_order = local_idx[np.argsort(-local_values)]
            native_pairs.extend(
                [
                    (running_offset + i, running_offset + j, weight)
                    for i, j, weight in sorted_pair_indices(
                        local_order,
                        native_conf,
                        offsets=cfg.native_pair_offsets,
                        max_pairs=max(8, cfg.max_pairs_per_target // 4),
                        min_margin=0.02,
                    )
                ]
            )

        support_count = np.array(
            [build_feature_dict(row, context, cfg=cfg)["support__count"] for row in df.to_dict(orient="records")],
            dtype=float,
        )
        if support_count.max() > support_count.min():
            consensus_order = np.argsort(-support_count)
            consensus_pairs.extend(
                [
                    (running_offset + i, running_offset + j, weight)
                    for i, j, weight in sorted_pair_indices(
                        consensus_order,
                        support_count,
                        offsets=cfg.consensus_pair_offsets,
                        max_pairs=max(8, cfg.max_pairs_per_target // 4),
                        min_margin=0.5,
                    )
                ]
            )

        running_offset += len(df)

    if len(feature_rows) < cfg.min_training_candidates:
        return None

    feature_names = sorted({name for row in feature_rows for name in row})
    raw_matrix = np.array(
        [[row.get(name, 0.0) for name in feature_names] for row in feature_rows],
        dtype=float,
    )
    means = raw_matrix.mean(axis=0)
    stds = raw_matrix.std(axis=0)
    stds[stds < 1e-6] = 1.0
    features = (raw_matrix - means) / stds
    return TrainingCorpus(
        features=features,
        feature_names=feature_names,
        means=means,
        stds=stds,
        regression_target=np.array(regression_targets, dtype=float),
        sample_targets=sample_targets,
        rank_pairs=rank_pairs,
        native_pairs=native_pairs,
        consensus_pairs=consensus_pairs,
    )


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
            sig = sigmoid(-diff)
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


def build_feature_matrix(rows: list[dict[str, Any]], context: OutputContext, feature_names: list[str], cfg: FusionRankerConfig | None = None) -> np.ndarray:
    feature_dicts = [build_feature_dict(row, context, cfg=cfg) for row in rows]
    raw = np.array([[feat.get(name, 0.0) for name in feature_names] for feat in feature_dicts], dtype=float)
    return raw


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
