"""MARS-FIELD topic scoring.

Public API (preserved for backwards compatibility)::

    from marsstack.topic_score import (
        TopicScoreResult,
        score_topic_candidate,
        build_topic_local_recommendations,
    )

Adding a new topic
------------------
1. Create ``marsstack/topic_score/my_topic.py`` exporting ``score`` and
   ``recommendations`` callables.
2. Import the module here and call :func:`register_topic`.
"""

from . import aresg, cld, drwh, microgravity
from ._common import (
    BURIED_BREAKERS,
    CHARGED,
    HYDROPHOBIC,
    LOW_COMPLEXITY_CORE,
    MICROGRAVITY_FLEX_RISK,
    MICROGRAVITY_STICKY,
    MICROGRAVITY_SURFACE_FAVORABLE,
    OXIDATION_PRONE,
    POLAR,
    SAFE_OXIDATION_TOPIC_MAP,
    TopicScoreResult,
)
from .registry import (
    TopicHandlers,
    build_topic_local_recommendations,
    get_topic,
    register_topic,
    registered_topics,
    score_topic_candidate,
)


register_topic("cld", score=cld.score, recommendations=cld.recommendations)
register_topic("drwh", score=drwh.score, recommendations=drwh.recommendations)
register_topic("aresg", score=aresg.score, recommendations=aresg.recommendations)
register_topic(
    "microgravity",
    score=microgravity.score,
    recommendations=microgravity.recommendations,
)


__all__ = [
    "TopicScoreResult",
    "TopicHandlers",
    "score_topic_candidate",
    "build_topic_local_recommendations",
    "register_topic",
    "registered_topics",
    "get_topic",
    "SAFE_OXIDATION_TOPIC_MAP",
    "HYDROPHOBIC",
    "POLAR",
    "CHARGED",
    "LOW_COMPLEXITY_CORE",
    "OXIDATION_PRONE",
    "BURIED_BREAKERS",
    "MICROGRAVITY_SURFACE_FAVORABLE",
    "MICROGRAVITY_STICKY",
    "MICROGRAVITY_FLEX_RISK",
]
