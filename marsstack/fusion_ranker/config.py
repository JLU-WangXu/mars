from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
