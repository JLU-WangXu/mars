from __future__ import annotations

from ..structure_features import ResidueFeature
from ._common import (
    BURIED_BREAKERS,
    HYDROPHOBIC,
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


DRWH_SCORE_DEFAULTS = {
    "buried_sasa_max": 20.0,
    "surface_sasa_min": 35.0,
    "target_hydrophobic_fraction_min": 0.28,
    "target_hydrophobic_fraction_max": 0.48,
    "target_polar_fraction_min": 0.16,
    "target_polar_fraction_max": 0.35,
    "target_net_charge_min": -8.0,
    "target_net_charge_max": 2.0,
    "profile_prior_scale": 0.30,
    "asr_prior_scale": 0.70,
    "family_prior_scale": 0.35,
}

DRWH_RECOMMEND_DEFAULTS = {
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
    cfg = merge_cfg(DRWH_SCORE_DEFAULTS, cfg)
    feat_map = {f.num: f for f in features}
    mutated = mutated_positions(wt_seq, seq, mutable_positions, position_to_index)
    components = {"topic_sequence": 0.0, "topic_structure": 0.0, "topic_evolution": 0.0}
    notes: list[str] = []

    hydrophobic_fraction = fraction(seq, HYDROPHOBIC)
    polar_fraction = fraction(seq, POLAR)
    seq_net_charge = net_charge(seq)

    if float(cfg["target_hydrophobic_fraction_min"]) <= hydrophobic_fraction <= float(cfg["target_hydrophobic_fraction_max"]):
        components["topic_sequence"] += 0.6
        notes.append("drwh_hydrophobic_balance")
    else:
        components["topic_sequence"] -= 0.4

    if float(cfg["target_polar_fraction_min"]) <= polar_fraction <= float(cfg["target_polar_fraction_max"]):
        components["topic_sequence"] += 0.5
        notes.append("drwh_polar_balance")
    else:
        components["topic_sequence"] -= 0.25

    if float(cfg["target_net_charge_min"]) <= seq_net_charge <= float(cfg["target_net_charge_max"]):
        components["topic_sequence"] += 0.45
        notes.append("drwh_charge_window")
    else:
        components["topic_sequence"] -= 0.35

    for pos in mutated:
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        aa = seq[idx]
        feat = feat_map.get(pos)
        if feat is None:
            continue
        if feat.sasa <= float(cfg["buried_sasa_max"]):
            if aa in HYDROPHOBIC:
                components["topic_structure"] += 0.35
            elif aa in BURIED_BREAKERS:
                components["topic_structure"] -= 0.85
                notes.append(f"drwh_core_break_{pos}")
        elif feat.sasa >= float(cfg["surface_sasa_min"]):
            if aa in POLAR | {"D", "E", "K"}:
                components["topic_structure"] += 0.35
            if aa in OXIDATION_PRONE:
                components["topic_structure"] -= 0.45
                notes.append(f"drwh_surface_oxidation_risk_{pos}")

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
        notes.append("drwh_topic_evolution")

    return finalize_result(components, notes)


def recommendations(
    wt_seq: str,
    features: list[ResidueFeature],
    design_positions: list[int],
    position_to_index: dict[int, int],
    cfg: dict[str, object] | None,
) -> dict[int, dict[str, float]]:
    cfg = merge_cfg(DRWH_RECOMMEND_DEFAULTS, cfg)
    feat_map = {f.num: f for f in features}
    recs: dict[int, dict[str, float]] = {}
    for pos in design_positions:
        feat = feat_map.get(pos)
        if feat is None:
            continue
        aa_scores: dict[str, float] = {}
        if feat.sasa <= float(cfg["buried_sasa_max"]):
            for aa in {"L", "I", "V", "F", "A", wt_seq[position_to_index[pos]]}:
                aa_scores[aa] = 0.55
        elif feat.sasa >= float(cfg["surface_sasa_min"]):
            for aa in {"Q", "N", "E", "D", "K", "S", "T"}:
                aa_scores[aa] = 0.45
        if aa_scores:
            recs[pos] = aa_scores
    return recs
