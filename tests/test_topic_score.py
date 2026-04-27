"""Topic-score behavioural tests.

These reuse the four-topic fixtures previously baked into
``scripts/validate_topic_scoring.py``, but pinned as pytest assertions so
refactors can't silently shift component scores.
"""

from __future__ import annotations

import pytest

from marsstack.mars_score import score_candidate
from marsstack.structure_features import ResidueFeature
from marsstack.topic_score import (
    register_topic,
    registered_topics,
    score_topic_candidate,
)


def make_profile(length: int, supported: dict[int, dict[str, float]]) -> list[dict[str, float]]:
    profile = [{} for _ in range(length)]
    for pos, mapping in supported.items():
        profile[pos - 1] = dict(mapping)
    return profile


def set_seq_chars(length: int, updates: dict[int, str], fill: str = "A") -> str:
    chars = [fill] * length
    for pos, aa in updates.items():
        chars[pos - 1] = aa
    return "".join(chars)


def build_feature(num: int, aa: str, sasa: float, b: float = 25.0) -> ResidueFeature:
    return ResidueFeature(
        num=num,
        name="ALA",
        aa=aa,
        sasa=sasa,
        mean_b=b,
        min_dist_protected=12.0,
        in_disulfide=False,
        glyco_motif=False,
    )


def test_registry_contains_default_topics():
    assert sorted(registered_topics()) == ["aresg", "cld", "drwh", "microgravity"]


def test_unknown_topic_is_safe():
    res = score_topic_candidate(
        topic_name="not_a_real_topic",
        wt_seq="A" * 4,
        seq="A" * 4,
        features=[],
        mutable_positions=None,
        position_to_index=None,
        profile=None,
        asr_profile=None,
        family_positive_profile=None,
        family_negative_profile=None,
    )
    assert res.total == 0.0
    assert res.notes == ["unknown_topic_not_a_real_topic"]


def test_cld_case_components_in_expected_ranges():
    length = 300
    wt = set_seq_chars(length, {189: "W", 190: "W", 204: "H", 217: "R", 254: "E", 261: "W"})
    seq = set_seq_chars(length, {189: "F", 190: "F", 204: "Q", 217: "R", 254: "E", 261: "F"})
    features = [
        build_feature(189, "W", 18.0),
        build_feature(190, "W", 18.0),
        build_feature(204, "H", 12.0),
        build_feature(217, "R", 26.0),
        build_feature(254, "E", 14.0),
        build_feature(261, "W", 19.0),
    ]
    profile = make_profile(
        length,
        {189: {"F": 0.6}, 190: {"F": 0.6}, 204: {"Q": 0.7}, 261: {"F": 0.65}},
    )
    res = score_candidate(
        wt_seq=wt,
        seq=seq,
        features=features,
        oxidation_hotspots=[],
        flexible_positions=[],
        profile=profile,
        asr_profile=profile,
        family_positive_profile=profile,
        family_negative_profile=make_profile(length, {189: {"W": 0.6}, 190: {"W": 0.6}}),
        manual_preferred={},
        evolution_positions=[189, 190, 204, 217, 254, 261],
        mutable_positions=[189, 190, 204, 261],
        position_to_index={i: i - 1 for i in range(1, length + 1)},
        topic_name="cld",
        topic_cfg={
            "enabled": True,
            "name": "cld",
            "cld": {
                "functional_shell_positions": [189, 190, 204, 217, 254, 261],
                "oxidation_guard_positions": [189, 190, 261],
                "distal_gate_positions": [217],
                "proximal_network_positions": [204, 254],
            },
        },
    )
    assert res.components["topic_structure"] > 0
    assert "cld_topic_evolution" in res.notes


def test_drwh_case_emits_topic_evolution_note():
    motif = "VLANQDGPAA"
    wt = motif * 12
    seq_list = list(wt)
    for pos, aa in {12: "M", 28: "Q", 39: "L", 58: "N", 72: "E"}.items():
        seq_list[pos - 1] = aa
    seq = "".join(seq_list)
    features = [
        build_feature(12, wt[11], 12.0),
        build_feature(28, wt[27], 44.0),
        build_feature(39, wt[38], 15.0),
        build_feature(58, wt[57], 42.0),
        build_feature(72, wt[71], 46.0),
    ]
    profile = make_profile(
        len(wt),
        {12: {"M": 0.5}, 28: {"Q": 0.6}, 39: {"L": 0.55}, 58: {"N": 0.65}, 72: {"E": 0.6}},
    )
    res = score_candidate(
        wt_seq=wt,
        seq=seq,
        features=features,
        oxidation_hotspots=[],
        flexible_positions=[],
        profile=profile,
        asr_profile=profile,
        family_positive_profile=profile,
        family_negative_profile=make_profile(len(wt), {12: {"P": 0.6}}),
        manual_preferred={},
        evolution_positions=[12, 28, 39, 58, 72],
        mutable_positions=[12, 28, 39, 58, 72],
        position_to_index={i: i - 1 for i in range(1, len(wt) + 1)},
        topic_name="drwh",
        topic_cfg={"enabled": True, "name": "drwh", "drwh": {}},
    )
    assert "drwh_topic_evolution" in res.notes


def test_register_topic_adds_a_handler(monkeypatch):
    """Custom topics can be registered without touching dispatcher code."""

    captured: dict[str, object] = {}

    def fake_score(**kwargs):
        captured.update(kwargs)
        from marsstack.topic_score import TopicScoreResult

        return TopicScoreResult(
            total=1.23,
            components={"topic_sequence": 1.0, "topic_structure": 0.0, "topic_evolution": 0.23},
            notes=["fake_note"],
        )

    def fake_recs(**kwargs):
        return {1: {"A": 0.5}}

    register_topic("__test_dummy", score=fake_score, recommendations=fake_recs)
    try:
        from marsstack.topic_score import build_topic_local_recommendations

        res = score_topic_candidate(
            topic_name="__test_dummy",
            wt_seq="ACDE",
            seq="ACDE",
            features=[],
            mutable_positions=None,
            position_to_index={1: 0, 2: 1, 3: 2, 4: 3},
            profile=None,
            asr_profile=None,
            family_positive_profile=None,
            family_negative_profile=None,
            topic_cfg={"__test_dummy": {"some": "config"}},
        )
        assert res.total == pytest.approx(1.23)
        assert captured["cfg"] == {"some": "config"}
        recs = build_topic_local_recommendations(
            topic_name="__test_dummy",
            wt_seq="ACDE",
            features=[],
            design_positions=[1],
            position_to_index={1: 0, 2: 1, 3: 2, 4: 3},
            topic_cfg={"__test_dummy": {}},
        )
        assert recs == {1: {"A": 0.5}}
    finally:
        from marsstack.topic_score.registry import _REGISTRY

        _REGISTRY.pop("__test_dummy", None)
