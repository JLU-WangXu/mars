"""Evolution profile primitives."""

from __future__ import annotations

import math

import pytest

from marsstack.evolution import (
    build_profile,
    differential_profile_score,
    profile_log_score,
)


def test_build_profile_collapses_columns():
    seqs = ["AAAA", "ACAA", "ACGA"]
    profile = build_profile(seqs, "AAAA")
    assert profile is not None
    # column 0: all A
    assert profile[0] == {"A": pytest.approx(1.0)}
    # column 1: 1 A, 2 C
    assert profile[1] == {"A": pytest.approx(1 / 3), "C": pytest.approx(2 / 3)}


def test_build_profile_returns_none_when_no_matching_lengths():
    assert build_profile([], "AAAA") is None
    assert build_profile(["AAAAA"], "AAAA") is None


def test_profile_log_score_handles_missing_letter_with_floor():
    profile = [{"A": 1.0}]
    score = profile_log_score("C", profile, [1])
    # missing letter falls back to 1e-6, then `+1e-6` floor inside the log
    expected = math.log(2e-6)
    assert score == pytest.approx(expected, rel=1e-6)


def test_profile_log_score_position_weights_average_correctly():
    profile = [{"A": 1.0, "C": 0.0001}, {"A": 1.0, "C": 0.0001}]
    seq = "AC"
    weighted = profile_log_score(
        seq,
        profile,
        positions=[1, 2],
        position_weights={1: 0.0, 2: 1.0},
    )
    only_pos2 = profile_log_score(seq, profile, positions=[2])
    assert weighted == pytest.approx(only_pos2)


def test_differential_profile_score_swaps_sign_when_profiles_swap():
    pos_profile = [{"A": 0.9, "C": 0.1}]
    neg_profile = [{"A": 0.1, "C": 0.9}]
    score_for_a = differential_profile_score("A", pos_profile, neg_profile, [1])
    score_for_a_swapped = differential_profile_score("A", neg_profile, pos_profile, [1])
    assert score_for_a > 0
    assert score_for_a == pytest.approx(-score_for_a_swapped)
