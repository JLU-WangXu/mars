from __future__ import annotations

import re


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
