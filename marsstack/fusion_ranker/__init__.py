"""Learned fusion ranker (FM-style) for MARS-FIELD candidate ordering.

Public surface is preserved at the package root for backwards compatibility:
``from marsstack.fusion_ranker import apply_learned_fusion_ranking, rank_rows_with_model``.
"""

from ._utils import safe_float, sigmoid, split_semicolon
from .calibration import (
    apply_target_score_calibration,
    mutation_overlap_ratio,
    mutation_tokens,
)
from .config import FusionRankerConfig, OutputContext, TrainingCorpus
from .constants import (
    HEADER_FLOAT_RE,
    LINEAR_GROUP_RULES,
    NOTE_PREFIXES,
    NUMERIC_ROW_COLUMNS,
    SOURCE_GROUP_ORDER,
    SOURCE_ORDER,
)
from .corpus import (
    build_corpus,
    load_context,
    load_training_tables,
    sorted_pair_indices,
    standardize_target,
)
from .features import (
    build_feature_dict,
    build_feature_matrix,
    context_features,
    feature_group,
    mutation_features,
    note_features,
    parse_header_metrics,
    source_features,
    support_features,
)
from .model import (
    explain_rows,
    fm_gradients,
    fm_score,
    score_feature_matrix,
    train_factor_ranker,
)
from .ranking import apply_learned_fusion_ranking, rank_rows_with_model

__all__ = [
    "FusionRankerConfig",
    "OutputContext",
    "TrainingCorpus",
    "SOURCE_ORDER",
    "SOURCE_GROUP_ORDER",
    "NUMERIC_ROW_COLUMNS",
    "NOTE_PREFIXES",
    "LINEAR_GROUP_RULES",
    "HEADER_FLOAT_RE",
    "safe_float",
    "sigmoid",
    "split_semicolon",
    "parse_header_metrics",
    "note_features",
    "source_features",
    "support_features",
    "mutation_features",
    "context_features",
    "build_feature_dict",
    "feature_group",
    "build_feature_matrix",
    "load_context",
    "load_training_tables",
    "standardize_target",
    "sorted_pair_indices",
    "build_corpus",
    "fm_score",
    "fm_gradients",
    "train_factor_ranker",
    "score_feature_matrix",
    "explain_rows",
    "mutation_tokens",
    "mutation_overlap_ratio",
    "apply_target_score_calibration",
    "rank_rows_with_model",
    "apply_learned_fusion_ranking",
]
