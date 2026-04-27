from __future__ import annotations

from ..structure_features import ResidueFeature
from ._common import (
    BURIED_BREAKERS,
    HYDROPHOBIC,
    MICROGRAVITY_FLEX_RISK,
    MICROGRAVITY_STICKY,
    MICROGRAVITY_SURFACE_FAVORABLE,
    POLAR,
    TopicScoreResult,
    finalize_result,
    fraction,
    merge_cfg,
    mutated_positions,
    net_charge,
    score_profile_bundle,
)


MICROGRAVITY_SCORE_DEFAULTS = {
    "surface_sasa_min": 38.0,
    "buried_sasa_max": 20.0,
    "high_flex_b_min": 32.0,
    "target_polar_fraction_min": 0.18,
    "target_polar_fraction_max": 0.42,
    "target_net_charge_abs_min": 2.0,
    "target_net_charge_abs_max": 14.0,
    "surface_sticky_fraction_max": 0.33,
    "surface_charge_fraction_min": 0.14,
    "surface_charge_fraction_max": 0.42,
    "profile_prior_scale": 0.25,
    "asr_prior_scale": 0.35,
    "family_prior_scale": 0.35,
}

MICROGRAVITY_RECOMMEND_DEFAULTS = {
    "buried_sasa_max": 20.0,
    "surface_sasa_min": 38.0,
    "high_flex_b_min": 32.0,
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
    cfg = merge_cfg(MICROGRAVITY_SCORE_DEFAULTS, cfg)
    feat_map = {f.num: f for f in features}
    mutated = mutated_positions(wt_seq, seq, mutable_positions, position_to_index)
    components = {"topic_sequence": 0.0, "topic_structure": 0.0, "topic_evolution": 0.0}
    notes: list[str] = []

    polar_fraction = fraction(seq, POLAR | {"D", "E", "K", "R", "H"})
    net_charge_abs = abs(net_charge(seq))

    if float(cfg["target_polar_fraction_min"]) <= polar_fraction <= float(cfg["target_polar_fraction_max"]):
        components["topic_sequence"] += 0.55
        notes.append("microgravity_polar_window")
    else:
        components["topic_sequence"] -= 0.25

    if float(cfg["target_net_charge_abs_min"]) <= net_charge_abs <= float(cfg["target_net_charge_abs_max"]):
        components["topic_sequence"] += 0.35
        notes.append("microgravity_charge_window")
    else:
        components["topic_sequence"] -= 0.25

    surface_positions = [feat for feat in features if feat.sasa >= float(cfg["surface_sasa_min"])]
    if surface_positions:
        sticky_count = 0
        charged_count = 0
        for feat in surface_positions:
            idx = position_to_index[feat.num] if position_to_index is not None else feat.num - 1
            aa = seq[idx]
            if aa in MICROGRAVITY_STICKY:
                sticky_count += 1
            if aa in {"D", "E", "K", "R", "H"}:
                charged_count += 1

        sticky_fraction = sticky_count / len(surface_positions)
        charge_fraction = charged_count / len(surface_positions)

        if sticky_fraction <= float(cfg["surface_sticky_fraction_max"]):
            components["topic_structure"] += 0.65
            notes.append("microgravity_low_surface_stickiness")
        else:
            components["topic_structure"] -= 0.75
            notes.append("microgravity_surface_stickiness")

        if float(cfg["surface_charge_fraction_min"]) <= charge_fraction <= float(cfg["surface_charge_fraction_max"]):
            components["topic_structure"] += 0.45
            notes.append("microgravity_surface_charge_window")
        else:
            components["topic_structure"] -= 0.25

    for pos in mutated:
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        aa = seq[idx]
        feat = feat_map.get(pos)
        if feat is None:
            continue
        if feat.sasa >= float(cfg["surface_sasa_min"]):
            if aa in MICROGRAVITY_SURFACE_FAVORABLE:
                components["topic_structure"] += 0.4
            if aa in MICROGRAVITY_STICKY:
                components["topic_structure"] -= 0.7
                notes.append(f"microgravity_surface_sticky_{pos}")
            if feat.mean_b >= float(cfg["high_flex_b_min"]):
                if aa in {"Q", "N", "S", "T", "E", "D", "K"}:
                    components["topic_structure"] += 0.3
                    notes.append(f"microgravity_flexible_surface_buffer_{pos}")
                elif aa in MICROGRAVITY_FLEX_RISK:
                    components["topic_structure"] -= 0.45
                    notes.append(f"microgravity_flexible_surface_risk_{pos}")
        elif feat.sasa <= float(cfg["buried_sasa_max"]):
            if aa in HYDROPHOBIC:
                components["topic_structure"] += 0.25
            elif aa in BURIED_BREAKERS:
                components["topic_structure"] -= 0.8
                notes.append(f"microgravity_core_break_{pos}")

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
        notes.append("microgravity_topic_evolution")

    return finalize_result(components, notes)


def recommendations(
    wt_seq: str,
    features: list[ResidueFeature],
    design_positions: list[int],
    position_to_index: dict[int, int],
    cfg: dict[str, object] | None,
) -> dict[int, dict[str, float]]:
    cfg = merge_cfg(MICROGRAVITY_RECOMMEND_DEFAULTS, cfg)
    feat_map = {f.num: f for f in features}
    recs: dict[int, dict[str, float]] = {}
    for pos in design_positions:
        feat = feat_map.get(pos)
        if feat is None:
            continue
        aa_scores: dict[str, float] = {}
        if feat.sasa >= float(cfg["surface_sasa_min"]):
            preferred = {"Q", "N", "S", "T", "E", "D", "K", "R", "A"}
            bonus = 0.55 if feat.mean_b >= float(cfg["high_flex_b_min"]) else 0.45
            for aa in preferred:
                aa_scores[aa] = bonus
        elif feat.sasa <= float(cfg["buried_sasa_max"]):
            for aa in {"A", "L", "V", "I", wt_seq[position_to_index[pos]]}:
                aa_scores[aa] = 0.35
        if aa_scores:
            recs[pos] = aa_scores
    return recs
