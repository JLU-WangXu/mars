from __future__ import annotations

from typing import Any

from ._utils import safe_float, split_semicolon
from .config import FusionRankerConfig, OutputContext
from .constants import (
    HEADER_FLOAT_RE,
    LINEAR_GROUP_RULES,
    NOTE_PREFIXES,
    NUMERIC_ROW_COLUMNS,
    SOURCE_GROUP_ORDER,
    SOURCE_ORDER,
)


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


def build_feature_matrix(
    rows: list[dict[str, Any]],
    context: OutputContext,
    feature_names: list[str],
    cfg: FusionRankerConfig | None = None,
):
    import numpy as np

    feature_dicts = [build_feature_dict(row, context, cfg=cfg) for row in rows]
    raw = np.array([[feat.get(name, 0.0) for name in feature_names] for feat in feature_dicts], dtype=float)
    return raw
