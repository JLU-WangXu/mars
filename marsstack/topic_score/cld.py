from __future__ import annotations

from ..structure_features import ResidueFeature
from ._common import (
    CHARGED,
    HYDROPHOBIC,
    OXIDATION_PRONE,
    SAFE_OXIDATION_TOPIC_MAP,
    TopicScoreResult,
    finalize_result,
    merge_cfg,
    mutated_positions,
    score_profile_bundle,
)


CLD_SCORE_DEFAULTS = {
    "functional_shell_positions": [],
    "oxidation_guard_positions": [],
    "distal_gate_positions": [],
    "proximal_network_positions": [],
    "buried_sasa_max": 20.0,
    "profile_prior_scale": 0.35,
    "asr_prior_scale": 0.55,
    "family_prior_scale": 0.55,
}

CLD_RECOMMEND_DEFAULTS = {
    "functional_shell_positions": [],
    "oxidation_guard_positions": [],
    "distal_gate_positions": [],
    "proximal_network_positions": [],
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
    cfg = merge_cfg(CLD_SCORE_DEFAULTS, cfg)
    feat_map = {f.num: f for f in features}
    mutated = mutated_positions(wt_seq, seq, mutable_positions, position_to_index)
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
            elif aa in CHARGED or aa in {"N", "Q", "S", "T"}:
                components["topic_structure"] += 0.2

    if shell_mutations <= 1:
        components["topic_sequence"] += 0.5
        notes.append("cld_low_shell_burden")
    elif shell_mutations >= 3:
        components["topic_sequence"] -= 0.6
        notes.append("cld_high_shell_burden")

    if shell_positions:
        components["topic_evolution"] += score_profile_bundle(
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

    return finalize_result(components, notes)


def recommendations(
    wt_seq: str,
    features: list[ResidueFeature],
    design_positions: list[int],
    position_to_index: dict[int, int],
    cfg: dict[str, object] | None,
) -> dict[int, dict[str, float]]:
    cfg = merge_cfg(CLD_RECOMMEND_DEFAULTS, cfg)
    oxidation_guard = set(cfg["oxidation_guard_positions"])
    gate_network = set(cfg["distal_gate_positions"]) | set(cfg["proximal_network_positions"])
    shell = set(cfg["functional_shell_positions"])
    recs: dict[int, dict[str, float]] = {}
    for pos in design_positions:
        idx = position_to_index[pos]
        wt = wt_seq[idx]
        aa_scores: dict[str, float] = {}
        if pos in oxidation_guard:
            for aa, val in SAFE_OXIDATION_TOPIC_MAP.get(wt, {}).items():
                aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), 1.2 + 0.2 * val)
        if pos in gate_network:
            for aa in {"Q", "N", "E", "D", "Y", wt}:
                aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), 0.35)
        if pos in shell and wt in HYDROPHOBIC:
            for aa in {wt, "L", "I", "V", "F"}:
                aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), 0.25)
        if aa_scores:
            recs[pos] = aa_scores
    return recs
