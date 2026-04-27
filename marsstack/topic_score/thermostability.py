"""Thermostability topic scoring module.

Scoring for protein thermostability based on:
- Amino acid composition (thermophilic AA enrichment)
- Structural features (salt bridges, disulfide bonds, charge density)
- Evolutionary conservation profiles
"""
from __future__ import annotations

from ..structure_features import ResidueFeature
from ._common import (
    CHARGED,
    HYDROPHOBIC,
    TopicScoreResult,
    finalize_result,
    merge_cfg,
    mutated_positions,
    score_profile_bundle,
)


# Thermostable amino acids (enriched in thermophilic proteins)
THERMOSTABLE_AA = set("ILVMAFCW")
# Thermolabile amino acids (enriched in mesophilic proteins)
THERMOLABILE_AA = set("DEKRNSY")
# Proline/glycine helix breakers (penalize in core)
HELIX_BREAKERS = set("PG")
# Aromatic residues for hydrophobic core
AROMATIC_AA = set("FWY")


THERMOSTABILITY_SCORE_DEFAULTS = {
    "core_positions": [],
    "surface_positions": [],
    "salt_bridge_pairs": [],
    "disulfide_positions": [],
    "buried_sasa_max": 20.0,
    "surface_sasa_min": 40.0,
    "profile_prior_scale": 0.35,
    "asr_prior_scale": 0.55,
    "family_prior_scale": 0.55,
    "thermophilic_aa_bonus": 0.5,
    "helix_breaker_penalty": 0.8,
    "salt_bridge_bonus": 1.0,
    "disulfide_bonus": 0.9,
    "charge_mismatch_penalty": 0.6,
}

THERMOSTABILITY_RECOMMEND_DEFAULTS = {
    "core_positions": [],
    "surface_positions": [],
    "salt_bridge_pairs": [],
    "disulfide_positions": [],
}


def _score_aa_composition(seq: str, mutated: list[int], position_to_index: dict[int, int] | None) -> tuple[float, list[str]]:
    """Score amino acid composition for thermostability."""
    score = 0.0
    notes: list[str] = []

    thermostable_count = sum(1 for aa in seq if aa in THERMOSTABLE_AA)
    thermolabile_count = sum(1 for aa in seq if aa in THERMOLABILE_AA)

    if len(seq) > 0:
        thermostable_ratio = thermostable_count / len(seq)
        thermolabile_ratio = thermolabile_count / len(seq)

        # Bonus for high thermophilic AA content
        if thermostable_ratio >= 0.45:
            score += 1.2
            notes.append("high_thermophilic_composition")
        elif thermostable_ratio >= 0.35:
            score += 0.5
            notes.append("moderate_thermophilic_composition")

        # Penalty for high thermolabile content
        if thermolabile_ratio >= 0.35:
            score -= 0.8
            notes.append("high_thermolabile_composition")

    return score, notes


def _score_structural_features(
    seq: str,
    mutated: list[int],
    features: list[ResidueFeature],
    position_to_index: dict[int, int] | None,
    cfg: dict[str, object],
) -> tuple[float, list[str]]:
    """Score structural features for thermostability."""
    score = 0.0
    notes: list[str] = []
    feat_map = {f.num: f for f in features}

    core_positions = set(cfg.get("core_positions", []))
    surface_positions = set(cfg.get("surface_positions", []))
    salt_bridge_pairs = cfg.get("salt_bridge_pairs", [])
    disulfide_positions = set(cfg.get("disulfide_positions", []))
    buried_sasa_max = float(cfg.get("buried_sasa_max", 20.0))
    helix_breaker_penalty = float(cfg.get("helix_breaker_penalty", 0.8))
    salt_bridge_bonus = float(cfg.get("salt_bridge_bonus", 1.0))
    disulfide_bonus = float(cfg.get("disulfide_bonus", 0.9))
    charge_mismatch_penalty = float(cfg.get("charge_mismatch_penalty", 0.6))

    # Evaluate each mutation
    for pos in mutated:
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        wt = seq[:len([f for f in features])]  # placeholder
        aa = seq[idx]
        feat = feat_map.get(pos)

        if feat is None:
            continue

        # Proline/Glycine in core is destabilizing
        if pos in core_positions and aa in HELIX_BREAKERS:
            score -= helix_breaker_penalty
            notes.append(f"thermo_core_helix_breaker_{pos}")

        # Thermostable AA in core is stabilizing
        if pos in core_positions and aa in THERMOSTABLE_AA:
            score += 0.4
            notes.append(f"thermo_core_stabilizing_{pos}")

        # Surface position analysis
        if pos in surface_positions:
            # Charged residues on surface can form stabilizing interactions
            if aa in CHARGED:
                score += 0.3
                notes.append(f"thermo_surface_charged_{pos}")
            # Small polar residues on surface are acceptable
            elif aa in "NQST":
                score += 0.2
                notes.append(f"thermo_surface_polar_{pos}")

        # Disulfide bond preservation
        if pos in disulfide_positions and aa == "C":
            score += disulfide_bonus
            notes.append(f"thermo_disulfide_preserved_{pos}")
        elif pos in disulfide_positions and aa != "C":
            score -= disulfide_bonus * 0.5
            notes.append(f"thermo_disulfide_broken_{pos}")

    # Evaluate salt bridge pairs
    for pair in salt_bridge_pairs:
        if len(pair) >= 2:
            pos1, pos2 = pair[0], pair[1]
            feat1 = feat_map.get(pos1)
            feat2 = feat_map.get(pos2)

            if feat1 and feat2:
                aa1, aa2 = feat1.aa, feat2.aa
                idx1 = position_to_index.get(pos1, pos1 - 1)
                idx2 = position_to_index.get(pos2, pos2 - 1)

                # Check if either position is mutated
                mutated_pos1 = idx1 < len(seq) and (
                    position_to_index is None or
                    any(mutated_positions := mutated)
                )
                mutated_pos2 = idx2 < len(seq)

                # Formation or preservation of salt bridge
                is_positive = lambda a: a in "KRH"
                is_negative = lambda a: a in "DE"

                if is_positive(aa1) and is_negative(aa2):
                    if mutated_pos1 or mutated_pos2:
                        score += salt_bridge_bonus
                        notes.append(f"thermo_salt_bridge_{pos1}_{pos2}")
                elif is_negative(aa1) and is_positive(aa2):
                    if mutated_pos1 or mutated_pos2:
                        score += salt_bridge_bonus
                        notes.append(f"thermo_salt_bridge_{pos1}_{pos2}")

    return score, notes


def _score_charge_interactions(
    seq: str,
    mutated: list[int],
    features: list[ResidueFeature],
    position_to_index: dict[int, int] | None,
) -> tuple[float, list[str]]:
    """Score charge interactions and charge density."""
    score = 0.0
    notes: list[str] = []
    feat_map = {f.num: f for f in features}

    # Calculate charge balance
    total_charge = 0.0
    charge_positions = []

    for feat in features:
        idx = position_to_index.get(feat.num, feat.num - 1)
        if idx < len(seq):
            aa = seq[idx]
            charge_map = {"K": 1, "R": 1, "H": 0.5, "D": -1, "E": -1}
            charge = charge_map.get(aa, 0)
            total_charge += charge
            if charge != 0:
                charge_positions.append((feat.num, charge, feat.sasa))

    # Penalize extreme net charge
    if abs(total_charge) >= 5:
        score -= 0.6
        notes.append("thermo_extreme_net_charge")

    # Penalize charge clusters on surface (can be destabilizing)
    charge_clusters = 0
    for i, (pos1, _, sasa1) in enumerate(charge_positions):
        cluster_count = 1
        for j in range(i + 1, len(charge_positions)):
            pos2, _, sasa2 = charge_positions[j]
            if abs(pos2 - pos1) <= 3 and sasa1 >= 40 and sasa2 >= 40:
                cluster_count += 1
        if cluster_count >= 3:
            charge_clusters += 1

    if charge_clusters >= 2:
        score -= 0.5
        notes.append("thermo_charge_clustering")

    return score, notes


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
    cfg = merge_cfg(THERMOSTABILITY_SCORE_DEFAULTS, cfg)
    mutated = mutated_positions(wt_seq, seq, mutable_positions, position_to_index)
    components = {"topic_sequence": 0.0, "topic_structure": 0.0, "topic_evolution": 0.0}
    notes: list[str] = []

    # Amino acid composition scoring (topic_sequence)
    aa_score, aa_notes = _score_aa_composition(seq, mutated, position_to_index)
    components["topic_sequence"] += aa_score
    notes.extend(aa_notes)

    # Structural features scoring (topic_structure)
    struct_score, struct_notes = _score_structural_features(
        seq, mutated, features, position_to_index, cfg
    )
    components["topic_structure"] += struct_score
    notes.extend(struct_notes)

    # Charge interactions scoring (topic_structure)
    charge_score, charge_notes = _score_charge_interactions(
        seq, mutated, features, position_to_index
    )
    components["topic_structure"] += charge_score
    notes.extend(charge_notes)

    # Evolutionary profile scoring (topic_evolution)
    core_positions = set(cfg.get("core_positions", []))
    if core_positions:
        components["topic_evolution"] += score_profile_bundle(
            seq=seq,
            positions=sorted(core_positions),
            position_to_index=position_to_index,
            profile=profile,
            asr_profile=asr_profile,
            family_positive_profile=family_positive_profile,
            family_negative_profile=family_negative_profile,
            profile_scale=float(cfg["profile_prior_scale"]),
            asr_scale=float(cfg["asr_prior_scale"]),
            family_scale=float(cfg["family_prior_scale"]),
        )
        notes.append("thermo_topic_evolution")

    return finalize_result(components, notes)


def recommendations(
    wt_seq: str,
    features: list[ResidueFeature],
    design_positions: list[int],
    position_to_index: dict[int, int],
    cfg: dict[str, object] | None,
) -> dict[int, dict[str, float]]:
    cfg = merge_cfg(THERMOSTABILITY_RECOMMEND_DEFAULTS, cfg)
    core_positions = set(cfg.get("core_positions", []))
    surface_positions = set(cfg.get("surface_positions", []))
    disulfide_positions = set(cfg.get("disulfide_positions", []))

    recs: dict[int, dict[str, float]] = {}

    for pos in design_positions:
        idx = position_to_index[pos]
        wt = wt_seq[idx]
        aa_scores: dict[str, float] = {}

        if pos in core_positions:
            # Core positions: favor thermostable AAs
            for aa in THERMOSTABLE_AA:
                if aa != wt:
                    aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), 0.8)
            # Penalize helix breakers in core
            for aa in HELIX_BREAKERS:
                aa_scores[aa] = min(aa_scores.get(aa, float("inf")), -0.5)

        if pos in surface_positions:
            # Surface positions: allow charged and polar
            for aa in CHARGED | {"N", "Q", "S", "T"}:
                if aa != wt:
                    aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), 0.5)

        if pos in disulfide_positions:
            # Disulfide positions: prefer cysteine
            aa_scores["C"] = max(aa_scores.get("C", float("-inf")), 1.0)

        # General recommendations for all positions
        if wt not in THERMOSTABLE_AA and wt not in HELIX_BREAKERS:
            for aa in THERMOSTABLE_AA:
                aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), 0.3)

        if aa_scores:
            recs[pos] = aa_scores

    return recs
