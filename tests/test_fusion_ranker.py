"""Fusion-ranker FM math + config plumbing."""

from __future__ import annotations

import numpy as np
import pytest

from marsstack.fusion_ranker import (
    FusionRankerConfig,
    OutputContext,
    build_feature_dict,
    feature_group,
    fm_score,
    sigmoid,
)


def test_config_from_dict_coerces_offsets_to_tuples_of_int():
    cfg = FusionRankerConfig.from_dict({"rank_pair_offsets": [3, 7, 11]})
    assert cfg.rank_pair_offsets == (3, 7, 11)
    assert all(isinstance(x, int) for x in cfg.rank_pair_offsets)


def test_config_from_dict_ignores_unknown_keys():
    cfg = FusionRankerConfig.from_dict({"definitely_not_a_field": 1.0})
    assert not hasattr(cfg, "definitely_not_a_field")


def test_sigmoid_extreme_inputs_stay_in_unit_interval():
    extremes = np.array([-1e3, -10.0, 0.0, 10.0, 1e3])
    out = sigmoid(extremes)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
    assert out[2] == pytest.approx(0.5)


def test_fm_score_matches_manual_formula_for_small_input():
    features = np.array([[1.0, 2.0]], dtype=float)
    linear = np.array([0.5, -0.25], dtype=float)
    factors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    bias = 0.1

    score = fm_score(features, bias=bias, linear=linear, factors=factors)
    # Manual: linear = 0.1 + 0.5 - 0.5 = 0.1
    # interaction = 0.5 * sum_k ((sum_i f_i v_ik)^2 - sum_i f_i^2 v_ik^2)
    # k=0: (1*1 + 2*0)^2 - (1*1 + 4*0) = 1 - 1 = 0
    # k=1: (1*0 + 2*1)^2 - (1*0 + 4*1) = 4 - 4 = 0
    # interaction = 0
    assert score[0] == pytest.approx(0.1)


def test_build_feature_dict_handles_missing_columns():
    ctx = OutputContext(
        protein="x",
        feature_summary={"design_positions": [1, 2, 3], "oxidation_hotspots": [], "flexible_surface_positions": []},
        profile_summary={},
    )
    feats = build_feature_dict({"source": "manual", "source_group": "manual_control"}, ctx)
    assert feats["mars_score"] == 0.0
    assert feats["source__manual"] == 1.0
    assert feats["source_group__manual_control"] == 1.0
    assert feats["mutation__count"] == 0.0


def test_feature_group_falls_back_to_misc_for_unknown_prefix():
    assert feature_group("totally_unknown_feature") == "misc"
    assert feature_group("source__manual") == "generator"
    assert feature_group("score_evolution") == "evolution"
    assert feature_group("score_topic_sequence") == "topic"
