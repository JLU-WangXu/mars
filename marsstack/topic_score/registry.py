"""Topic registry for MARS-FIELD topic scoring.

A "topic" is a domain-specific scorer (Cld, DrwH, AresG, Microgravity, ...).
Each topic module exposes ``score(...)`` and ``recommendations(...)`` and is
registered here so the dispatcher can route by name. Adding a new topic means
dropping a new module in this package and calling :func:`register_topic`.

Example::

    # marsstack/topic_score/my_topic.py
    from . import _common
    def score(...): ...
    def recommendations(...): ...

    # then in marsstack/topic_score/__init__.py
    from . import my_topic
    register_topic("my_topic", score=my_topic.score,
                                  recommendations=my_topic.recommendations)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..structure_features import ResidueFeature
from ._common import TopicScoreResult


ScoreFn = Callable[..., TopicScoreResult]
RecommendFn = Callable[..., dict[int, dict[str, float]]]


@dataclass(frozen=True)
class TopicHandlers:
    score: ScoreFn
    recommendations: RecommendFn


_REGISTRY: dict[str, TopicHandlers] = {}


def register_topic(name: str, *, score: ScoreFn, recommendations: RecommendFn) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("topic name must be non-empty")
    _REGISTRY[key] = TopicHandlers(score=score, recommendations=recommendations)


def get_topic(name: str) -> TopicHandlers | None:
    return _REGISTRY.get(name.strip().lower())


def registered_topics() -> list[str]:
    return sorted(_REGISTRY)


def score_topic_candidate(
    topic_name: str | None,
    wt_seq: str,
    seq: str,
    features: list[ResidueFeature],
    mutable_positions: list[int] | None,
    position_to_index: dict[int, int] | None,
    profile: list[dict[str, float]] | None,
    asr_profile: list[dict[str, float]] | None,
    family_positive_profile: list[dict[str, float]] | None,
    family_negative_profile: list[dict[str, float]] | None,
    topic_cfg: dict[str, object] | None = None,
) -> TopicScoreResult:
    empty_components = {"topic_sequence": 0.0, "topic_structure": 0.0, "topic_evolution": 0.0}
    if not topic_name:
        return TopicScoreResult(total=0.0, components=empty_components, notes=[])

    name = topic_name.strip().lower()
    handlers = _REGISTRY.get(name)
    if handlers is None:
        return TopicScoreResult(total=0.0, components=empty_components, notes=[f"unknown_topic_{name}"])

    cfg = (topic_cfg or {}).get(name, {})
    return handlers.score(
        wt_seq=wt_seq,
        seq=seq,
        features=features,
        mutable_positions=mutable_positions,
        position_to_index=position_to_index,
        profile=profile,
        asr_profile=asr_profile,
        family_positive_profile=family_positive_profile,
        family_negative_profile=family_negative_profile,
        cfg=cfg,
    )


def build_topic_local_recommendations(
    topic_name: str | None,
    wt_seq: str,
    features: list[ResidueFeature],
    design_positions: list[int],
    position_to_index: dict[int, int],
    topic_cfg: dict[str, object] | None = None,
) -> dict[int, dict[str, float]]:
    if not topic_name:
        return {}

    name = topic_name.strip().lower()
    handlers = _REGISTRY.get(name)
    if handlers is None:
        return {}

    cfg = (topic_cfg or {}).get(name, {})
    return handlers.recommendations(
        wt_seq=wt_seq,
        features=features,
        design_positions=design_positions,
        position_to_index=position_to_index,
        cfg=cfg,
    )
