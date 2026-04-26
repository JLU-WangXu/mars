from __future__ import annotations

from dataclasses import dataclass

from .evolution import differential_profile_score, profile_log_score
from .structure_features import ResidueFeature


HYDROPHOBIC = set("AILMFWVY")
POLAR = set("NQST")
CHARGED = set("DEKRH")
LOW_COMPLEXITY_CORE = set("QSGNAEKDTP")
OXIDATION_PRONE = set("MWYCH")
BURIED_BREAKERS = set("PGDEKR")
MICROGRAVITY_SURFACE_FAVORABLE = set("NQSTDEKRHA")
MICROGRAVITY_STICKY = set("FWYMLIVC")
MICROGRAVITY_FLEX_RISK = set("PGFWYMC")
SAFE_OXIDATION_TOPIC_MAP = {
    "M": {"L": 3.0, "I": 2.6, "V": 1.8},
    "C": {"S": 2.0, "A": 1.4},
    "W": {"F": 1.6, "Y": 0.8},
    "Y": {"F": 1.2},
    "H": {"Q": 1.1, "N": 0.8},
}


@dataclass
class TopicScoreResult:
    total: float
    components: dict[str, float]
    notes: list[str]


def _merge_cfg(defaults: dict[str, object], overrides: dict[str, object] | None) -> dict[str, object]:
    merged = dict(defaults)
    if overrides:
        merged.update(overrides)
    return merged


def _fraction(seq: str, alphabet: set[str]) -> float:
    if not seq:
        return 0.0
    return sum(1 for aa in seq if aa in alphabet) / len(seq)


def _net_charge(seq: str) -> float:
    charge_map = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}
    return sum(charge_map.get(aa, 0.0) for aa in seq)


def _mutated_positions(
    wt_seq: str,
    seq: str,
    mutable_positions: list[int] | None,
    position_to_index: dict[int, int] | None,
) -> list[int]:
    if mutable_positions is None:
        positions = list(range(1, len(wt_seq) + 1))
    else:
        positions = list(mutable_positions)
    changed = []
    for pos in positions:
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        if wt_seq[idx] != seq[idx]:
            changed.append(pos)
    return changed


def _score_profile_bundle(
    seq: str,
    positions: list[int],
    position_to_index: dict[int, int] | None,
    profile: list[dict[str, float]] | None,
    asr_profile: list[dict[str, float]] | None,
    family_positive_profile: list[dict[str, float]] | None,
    family_negative_profile: list[dict[str, float]] | None,
    profile_scale: float,
    asr_scale: float,
    family_scale: float,
) -> float:
    total = 0.0
    if profile is not None:
        total += profile_scale * profile_log_score(seq, profile, positions, position_to_index=position_to_index)
    if asr_profile is not None:
        total += asr_scale * profile_log_score(seq, asr_profile, positions, position_to_index=position_to_index)
    if family_positive_profile is not None and family_negative_profile is not None:
        total += family_scale * differential_profile_score(
            seq,
            family_positive_profile,
            family_negative_profile,
            positions,
            position_to_index=position_to_index,
        )
    return total


def _score_cld(
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
    defaults = {
        "functional_shell_positions": [],
        "oxidation_guard_positions": [],
        "distal_gate_positions": [],
        "proximal_network_positions": [],
        "buried_sasa_max": 20.0,
        "profile_prior_scale": 0.35,
        "asr_prior_scale": 0.55,
        "family_prior_scale": 0.55,
    }
    cfg = _merge_cfg(defaults, cfg)
    feat_map = {f.num: f for f in features}
    mutated = _mutated_positions(wt_seq, seq, mutable_positions, position_to_index)
    components = {"topic_sequence": 0.0, "topic_structure": 0.0, "topic_evolution": 0.0}
    notes: list[str] = []

    shell_positions = sorted(
        set(cfg["functional_shell_positions"])
        | set(cfg["oxidation_guard_positions"])
        | set(cfg["distal_gate_positions"])
        | set(cfg["proximal_network_positions"])
    )
    shell_mutations = 0
    for pos in mutated:
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        wt = wt_seq[idx]
        aa = seq[idx]
        feat = feat_map.get(pos)
        if pos in shell_positions:
            shell_mutations += 1
            if aa in {"P", "G"}:
                components["topic_sequence"] -= 1.4
                notes.append(f"cld_shell_break_{pos}")
            elif wt in HYDROPHOBIC and aa not in HYDROPHOBIC and feat and feat.sasa <= float(cfg["buried_sasa_max"]):
                components["topic_structure"] -= 0.9
                notes.append(f"cld_buried_pocket_soften_{pos}")
            elif wt in CHARGED and aa in CHARGED:
                components["topic_sequence"] += 0.25
            else:
                components["topic_sequence"] += 0.1

        if pos in set(cfg["oxidation_guard_positions"]):
            if aa == wt:
                components["topic_structure"] -= 0.35
                notes.append(f"cld_guard_kept_{pos}")
            elif aa in SAFE_OXIDATION_TOPIC_MAP.get(wt, {}):
                components["topic_structure"] += 1.2
                notes.append(f"cld_guard_hardened_{pos}")
            elif aa in OXIDATION_PRONE:
                components["topic_structure"] -= 1.1
                notes.append(f"cld_guard_bad_choice_{pos}")

        if pos in set(cfg["distal_gate_positions"]) | set(cfg["proximal_network_positions"]):
            if aa in {"P", "G"}:
                components["topic_structure"] -= 1.2
                notes.append(f"cld_network_break_{pos}")
            elif aa in CHARGED or aa in POLAR:
                components["topic_structure"] += 0.2

    if shell_mutations <= 1:
        components["topic_sequence"] += 0.5
        notes.append("cld_low_shell_burden")
    elif shell_mutations >= 3:
        components["topic_sequence"] -= 0.6
        notes.append("cld_high_shell_burden")

    if shell_positions:
        components["topic_evolution"] += _score_profile_bundle(
            seq=seq,
            positions=shell_positions,
            position_to_index=position_to_index,
            profile=profile,
            asr_profile=asr_profile,
            family_positive_profile=family_positive_profile,
            family_negative_profile=family_negative_profile,
            profile_scale=float(cfg["profile_prior_scale"]),
            asr_scale=float(cfg["asr_prior_scale"]),
            family_scale=float(cfg["family_prior_scale"]),
        )
        notes.append("cld_topic_evolution")

    total = round(sum(components.values()), 3)
    return TopicScoreResult(total=total, components={k: round(v, 3) for k, v in components.items()}, notes=notes)


def _score_drwh(
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
    defaults = {
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
    cfg = _merge_cfg(defaults, cfg)
    feat_map = {f.num: f for f in features}
    mutated = _mutated_positions(wt_seq, seq, mutable_positions, position_to_index)
    components = {"topic_sequence": 0.0, "topic_structure": 0.0, "topic_evolution": 0.0}
    notes: list[str] = []

    hydrophobic_fraction = _fraction(seq, HYDROPHOBIC)
    polar_fraction = _fraction(seq, POLAR)
    net_charge = _net_charge(seq)

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

    if float(cfg["target_net_charge_min"]) <= net_charge <= float(cfg["target_net_charge_max"]):
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
        components["topic_evolution"] += _score_profile_bundle(
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

    total = round(sum(components.values()), 3)
    return TopicScoreResult(total=total, components={k: round(v, 3) for k, v in components.items()}, notes=notes)


def _score_aresg(
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
    defaults = {
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
    cfg = _merge_cfg(defaults, cfg)
    feat_map = {f.num: f for f in features}
    mutated = _mutated_positions(wt_seq, seq, mutable_positions, position_to_index)
    components = {"topic_sequence": 0.0, "topic_structure": 0.0, "topic_evolution": 0.0}
    notes: list[str] = []

    low_complexity_fraction = _fraction(seq, LOW_COMPLEXITY_CORE)
    polar_fraction = _fraction(seq, POLAR | {"D", "E", "K"})
    hydrophobic_fraction = _fraction(seq, HYDROPHOBIC)
    net_charge_abs = abs(_net_charge(seq))

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
        components["topic_evolution"] += _score_profile_bundle(
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

    total = round(sum(components.values()), 3)
    return TopicScoreResult(total=total, components={k: round(v, 3) for k, v in components.items()}, notes=notes)


def _score_microgravity(
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
    defaults = {
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
    cfg = _merge_cfg(defaults, cfg)
    feat_map = {f.num: f for f in features}
    mutated = _mutated_positions(wt_seq, seq, mutable_positions, position_to_index)
    components = {"topic_sequence": 0.0, "topic_structure": 0.0, "topic_evolution": 0.0}
    notes: list[str] = []

    polar_fraction = _fraction(seq, POLAR | {"D", "E", "K", "R", "H"})
    net_charge_abs = abs(_net_charge(seq))

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
        components["topic_evolution"] += _score_profile_bundle(
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

    total = round(sum(components.values()), 3)
    return TopicScoreResult(total=total, components={k: round(v, 3) for k, v in components.items()}, notes=notes)


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
    if not topic_name:
        return TopicScoreResult(
            total=0.0,
            components={"topic_sequence": 0.0, "topic_structure": 0.0, "topic_evolution": 0.0},
            notes=[],
        )

    name = topic_name.strip().lower()
    topic_cfg = topic_cfg or {}
    if name == "cld":
        return _score_cld(
            wt_seq=wt_seq,
            seq=seq,
            features=features,
            mutable_positions=mutable_positions,
            position_to_index=position_to_index,
            profile=profile,
            asr_profile=asr_profile,
            family_positive_profile=family_positive_profile,
            family_negative_profile=family_negative_profile,
            cfg=topic_cfg.get("cld", {}),
        )
    if name == "drwh":
        return _score_drwh(
            wt_seq=wt_seq,
            seq=seq,
            features=features,
            mutable_positions=mutable_positions,
            position_to_index=position_to_index,
            profile=profile,
            asr_profile=asr_profile,
            family_positive_profile=family_positive_profile,
            family_negative_profile=family_negative_profile,
            cfg=topic_cfg.get("drwh", {}),
        )
    if name == "aresg":
        return _score_aresg(
            wt_seq=wt_seq,
            seq=seq,
            features=features,
            mutable_positions=mutable_positions,
            position_to_index=position_to_index,
            profile=profile,
            asr_profile=asr_profile,
            family_positive_profile=family_positive_profile,
            family_negative_profile=family_negative_profile,
            cfg=topic_cfg.get("aresg", {}),
        )
    if name == "microgravity":
        return _score_microgravity(
            wt_seq=wt_seq,
            seq=seq,
            features=features,
            mutable_positions=mutable_positions,
            position_to_index=position_to_index,
            profile=profile,
            asr_profile=asr_profile,
            family_positive_profile=family_positive_profile,
            family_negative_profile=family_negative_profile,
            cfg=topic_cfg.get("microgravity", {}),
        )

    return TopicScoreResult(
        total=0.0,
        components={"topic_sequence": 0.0, "topic_structure": 0.0, "topic_evolution": 0.0},
        notes=[f"unknown_topic_{name}"],
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
    topic_cfg = topic_cfg or {}
    feat_map = {f.num: f for f in features}
    recs: dict[int, dict[str, float]] = {}

    if name == "cld":
        cfg = _merge_cfg(
            {
                "functional_shell_positions": [],
                "oxidation_guard_positions": [],
                "distal_gate_positions": [],
                "proximal_network_positions": [],
            },
            topic_cfg.get("cld", {}),
        )
        oxidation_guard = set(cfg["oxidation_guard_positions"])
        gate_network = set(cfg["distal_gate_positions"]) | set(cfg["proximal_network_positions"])
        shell = set(cfg["functional_shell_positions"])
        for pos in design_positions:
            idx = position_to_index[pos]
            wt = wt_seq[idx]
            aa_scores: dict[str, float] = {}
            if pos in oxidation_guard:
                for aa, score in SAFE_OXIDATION_TOPIC_MAP.get(wt, {}).items():
                    aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), 1.2 + 0.2 * score)
            if pos in gate_network:
                for aa in {"Q", "N", "E", "D", "Y", wt}:
                    aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), 0.35)
            if pos in shell and wt in HYDROPHOBIC:
                for aa in {wt, "L", "I", "V", "F"}:
                    aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), 0.25)
            if aa_scores:
                recs[pos] = aa_scores
        return recs

    if name == "drwh":
        cfg = _merge_cfg({"buried_sasa_max": 20.0, "surface_sasa_min": 35.0}, topic_cfg.get("drwh", {}))
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

    if name == "aresg":
        cfg = _merge_cfg({"buried_sasa_max": 20.0, "surface_sasa_min": 35.0}, topic_cfg.get("aresg", {}))
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

    if name == "microgravity":
        cfg = _merge_cfg(
            {"buried_sasa_max": 20.0, "surface_sasa_min": 38.0, "high_flex_b_min": 32.0},
            topic_cfg.get("microgravity", {}),
        )
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

    return {}
