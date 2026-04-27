from __future__ import annotations

from ..structure_features import ResidueFeature
from ._common import (
    BURIED_BREAKERS,
    HYDROPHOBIC,
    LOW_COMPLEXITY_CORE,
    OXIDATION_PRONE,
    POLAR,
    TopicScoreResult,
    finalize_result,
    fraction,
    merge_cfg,
    mutated_positions,
    net_charge,
    score_profile_bundle,
)


ARESG_SCORE_DEFAULTS = {
    "buried_sasa_max": 20.0,
    "surface_sasa_min": 35.0,
    "target_low_complexity_min": 0.40,
    "target_low_complexity_max": 0.82,
    "target_polar_fraction_min": 0.20,
    "target_polar_fraction_max": 0.42,
    "target_hydrophobic_fraction_min": 0.22,
    "target_hydrophobic_fraction_max": 0.42,
    "target_net_charge_abs_max": 10.0,
    "profile_prior_scale": 0.25,
    "asr_prior_scale": 0.45,
    "family_prior_scale": 0.25,
}

ARESG_RECOMMEND_DEFAULTS = {
    "buried_sasa_max": 20.0,
    "surface_sasa_min": 35.0,
}


def score(
    wt_seq: str,
    seq: str,
    features: list[ResidueFeature],
    mutable_positions: list[int] | None,
    position_to_index: dict[int, int] | None,
    profile: list[dict[str, float]] | None,
    asr_profile: list[dict[str, float]] | None,
    family_positive_profile: list[dict[str, float]] | None,
    family_negative_profile: list[dict[str, float]] | None,
    cfg: dict[str, object],
) -> TopicScoreResult:
    cfg = merge_cfg(ARESG_SCORE_DEFAULTS, cfg)
    feat_map = {f.num: f for f in features}
    mutated = mutated_positions(wt_seq, seq, mutable_positions, position_to_index)
    components = {"topic_sequence": 0.0, "topic_structure": 0.0, "topic_evolution": 0.0}
    notes: list[str] = []

    low_complexity_fraction = fraction(seq, LOW_COMPLEXITY_CORE)
    polar_fraction = fraction(seq, POLAR | {"D", "E", "K"})
    hydrophobic_fraction = fraction(seq, HYDROPHOBIC)
    net_charge_abs = abs(net_charge(seq))

    if float(cfg["target_low_complexity_min"]) <= low_complexity_fraction <= float(cfg["target_low_complexity_max"]):
        components["topic_sequence"] += 0.8
        notes.append("aresg_low_complexity_window")
    else:
        components["topic_sequence"] -= 0.5

    if float(cfg["target_polar_fraction_min"]) <= polar_fraction <= float(cfg["target_polar_fraction_max"]):
        components["topic_sequence"] += 0.55
        notes.append("aresg_polar_window")
    else:
        components["topic_sequence"] -= 0.25

    if float(cfg["target_hydrophobic_fraction_min"]) <= hydrophobic_fraction <= float(cfg["target_hydrophobic_fraction_max"]):
        components["topic_sequence"] += 0.45
        notes.append("aresg_hydrophobic_window")
    else:
        components["topic_sequence"] -= 0.2

    if net_charge_abs <= float(cfg["target_net_charge_abs_max"]):
        components["topic_sequence"] += 0.35
        notes.append("aresg_charge_window")
    else:
        components["topic_sequence"] -= 0.4

    for pos in mutated:
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        aa = seq[idx]
        feat = feat_map.get(pos)
        if feat is None:
            continue
        if feat.sasa >= float(cfg["surface_sasa_min"]):
            if aa in {"Q", "N", "S", "T", "E", "D", "A", "K"}:
                components["topic_structure"] += 0.35
            if aa in OXIDATION_PRONE:
                components["topic_structure"] -= 0.5
                notes.append(f"aresg_surface_oxidation_risk_{pos}")
        elif feat.sasa <= float(cfg["buried_sasa_max"]):
            if aa in HYDROPHOBIC:
                components["topic_structure"] += 0.25
            elif aa in BURIED_BREAKERS:
                components["topic_structure"] -= 0.75
                notes.append(f"aresg_core_break_{pos}")

    evo_positions = mutated if mutated else (mutable_positions or [])
    if evo_positions:
        components["topic_evolution"] += score_profile_bundle(
            seq=seq,
            positions=evo_positions,
            position_to_index=position_to_index,
            profile=profile,
            asr_profile=asr_profile,
            family_positive_profile=family_positive_profile,
            family_negative_profile=family_negative_profile,
            profile_scale=float(cfg["profile_prior_scale"]),
            asr_scale=float(cfg["asr_prior_scale"]),
            family_scale=float(cfg["family_prior_scale"]),
        )
        notes.append("aresg_topic_evolution")

    return finalize_result(components, notes)


def recommendations(
    wt_seq: str,
    features: list[ResidueFeature],
    design_positions: list[int],
    position_to_index: dict[int, int],
    cfg: dict[str, object] | None,
) -> dict[int, dict[str, float]]:
    cfg = merge_cfg(ARESG_RECOMMEND_DEFAULTS, cfg)
    feat_map = {f.num: f for f in features}
    recs: dict[int, dict[str, float]] = {}
    for pos in design_positions:
        feat = feat_map.get(pos)
        if feat is None:
            continue
        aa_scores: dict[str, float] = {}
        if feat.sasa >= float(cfg["surface_sasa_min"]):
            for aa in {"Q", "N", "S", "T", "E", "D", "A", "K"}:
                aa_scores[aa] = 0.5
        elif feat.sasa <= float(cfg["buried_sasa_max"]):
            for aa in {"A", "L", "V", "I", wt_seq[position_to_index[pos]]}:
                aa_scores[aa] = 0.35
        if aa_scores:
            recs[pos] = aa_scores
    return recs
