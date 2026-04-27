from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FusionRankerConfig, OutputContext, TrainingCorpus
from .constants import SOURCE_ORDER
from .features import build_feature_dict


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


def sorted_pair_indices(
    order: np.ndarray,
    scores: np.ndarray,
    offsets: tuple[int, ...],
    max_pairs: int,
    min_margin: float,
) -> list[tuple[int, int, float]]:
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


def build_corpus(
    datasets: list[tuple[pd.DataFrame, OutputContext]],
    cfg: FusionRankerConfig,
) -> TrainingCorpus | None:
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
