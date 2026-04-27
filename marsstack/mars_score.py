from __future__ import annotations

import math
from dataclasses import dataclass

from .evolution import differential_profile_score, profile_log_score
from .structure_features import ResidueFeature
from .topic_score import score_topic_candidate


SAFE_OXIDATION_MAP = {
    "M": {"L": 3.0, "I": 2.6, "V": 1.8},
    "C": {"S": 2.0, "A": 1.4},
    "W": {"F": 1.6, "Y": 0.8},
    "Y": {"F": 1.2},
    "H": {"Q": 1.1, "N": 0.8},
}

# Thermostability contribution scores for each amino acid (based on experimental data)
# Positive values indicate contribution to thermal stability
# Reference: Thermodynamic scales derived from protein engineering experiments
AA_THERMOSTABILITY = {
    "A": 0.0,   # Alanine - neutral, helix former
    "R": 0.5,   # Arginine - stabilizing, forms H-bonds and salt bridges
    "N": -0.2,  # Asparagine - moderate destabilizer in buried positions
    "D": -0.3,  # Aspartate - destabilizing due to desolvation penalty
    "C": 0.3,   # Cysteine - stabilizing in disulfide bonds, destabilizing otherwise
    "E": 0.2,   # Glutamate - stabilizing when buried, destabilizing when exposed
    "Q": 0.1,   # Glutamine - slightly stabilizing
    "G": -0.5,  # Glycine - strong destabilizer, increases flexibility
    "H": 0.0,   # Histidine - pH-sensitive, context-dependent
    "I": 0.8,   # Isoleucine - strong stabilizer, beta-branched
    "L": 0.7,   # Leucine - strong stabilizer, hydrophobic core
    "K": 0.4,   # Lysine - stabilizing, charged
    "M": 0.6,   # Methionine - moderate stabilizer, flexible
    "F": 0.9,   # Phenylalanine - strong stabilizer, hydrophobic
    "P": -0.8,  # Proline - strong destabilizer, reduces flexibility
    "S": -0.1,  # Serine - slight destabilizer
    "T": 0.1,   # Threonine - slight stabilizer, beta-branched
    "W": 1.0,   # Tryptophan - strongest stabilizer, large hydrophobic
    "Y": 0.4,   # Tyrosine - stabilizer, aromatic
    "V": 0.5,   # Valine - stabilizer, beta-branched
}

# Amino acid volume (Angstroms^3) - for steric compatibility scoring
AA_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "E": 138.4, "Q": 143.8, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 164.0, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

# Amino acid hydrophobicity scale (Kyte-Doolittle, normalized to -1 to +1)
AA_HYDROPHOBICITY = {
    "A": 0.25, "R": -0.59, "N": -0.32, "D": -0.54, "C": 0.21,
    "E": -0.44, "Q": -0.31, "G": 0.00, "H": -0.14, "I": 0.62,
    "L": 0.61, "K": -0.61, "M": 0.30, "F": 0.61, "P": -0.07,
    "S": -0.04, "T": -0.05, "W": 0.37, "Y": 0.02, "V": 0.46,
}

# Charge at neutral pH
AA_CHARGE = {
    "A": 0.0, "R": 1.0, "N": 0.0, "D": -1.0, "C": 0.0,
    "E": -1.0, "Q": 0.0, "G": 0.0, "H": 0.1, "I": 0.0,
    "L": 0.0, "K": 1.0, "M": 0.0, "F": 0.0, "P": 0.0,
    "S": 0.0, "T": 0.0, "W": 0.0, "Y": 0.0, "V": 0.0,
}

# Beta-branched amino acids (affect backbone rigidity)
BETA_BRANCHED = {"I", "V", "T"}

# Aromatic amino acids
AROMATIC = {"F", "W", "Y"}

# Small amino acids
SMALL = {"A", "G", "S", "T"}

# Large hydrophobic amino acids
LARGE_HYDROPHOBIC = {"F", "W", "Y", "L", "I", "M"}

# Mutation stability impact predictions (simplified model)
# Format: {(wt, mut): (delta_delta_G, confidence)}
# Positive delta_delta_G = destabilizing, negative = stabilizing
# Based on statistical potential and experimental data
MUTATION_STABILITY_ESTIMATE = {
    # Alanine mutations
    ("G", "A"): (-0.5, 0.7),  # Gly->Ala often stabilizing (reduces flexibility)
    ("P", "A"): (-0.8, 0.6),  # Pro->Ala stabilizing (increases flexibility)
    ("D", "A"): (-0.3, 0.5),  # Asp->Ala often stabilizing
    ("E", "A"): (-0.2, 0.5),  # Glu->Ala often stabilizing

    # Core hydrophobic mutations
    ("L", "I"): (-0.2, 0.8),  # Conservative hydrophobic substitution
    ("L", "V"): (0.0, 0.7),   # Similar size hydrophobic
    ("I", "L"): (-0.1, 0.8),  # Conservative
    ("V", "I"): (0.1, 0.7),   # Conservative
    ("F", "Y"): (-0.1, 0.9),  # Aromatic substitution, very conservative
    ("F", "W"): (0.2, 0.6),   # Larger aromatic
    ("Y", "F"): (0.0, 0.9),   # Very conservative

    # Surface mutations
    ("D", "E"): (-0.1, 0.8),  # Conservative acidic
    ("E", "D"): (0.1, 0.8),   # Conservative acidic
    ("K", "R"): (-0.1, 0.8),  # Conservative basic
    ("R", "K"): (0.0, 0.7),   # Conservative basic
    ("S", "T"): (-0.1, 0.8),  # Conservative polar
    ("N", "Q"): (-0.1, 0.7),  # Conservative polar

    # Destabilizing mutations
    ("I", "P"): (1.2, 0.8),   # Beta-branched to cyclic
    ("V", "P"): (1.0, 0.7),   # Beta-branched to cyclic
    ("G", "P"): (0.8, 0.6),   # Flexible to rigid
    ("A", "P"): (0.6, 0.7),   # Small to cyclic

    # Charged to hydrophobic (context-dependent, often destabilizing when buried)
    ("K", "L"): (0.8, 0.6),
    ("R", "L"): (0.7, 0.6),
    ("E", "L"): (0.9, 0.6),
    ("D", "L"): (1.0, 0.6),

    # Proline to others (often destabilizing when going from other to Pro)
    ("L", "P"): (0.7, 0.7),
    ("I", "P"): (1.2, 0.8),
    ("V", "P"): (1.0, 0.7),

    # Glycine effects
    ("A", "G"): (0.5, 0.7),   # Rigid to flexible
    ("S", "G"): (0.3, 0.6),   # Small polar to flexible

    # Tryptophan conservation (highly stabilizing)
    ("Y", "W"): (-0.2, 0.7),
    ("F", "W"): (0.3, 0.7),
}


@dataclass
class ScoreResult:
    total: float
    notes: list[str]
    components: dict[str, float]


def mutation_list(wt_seq: str, seq: str, residue_numbers: list[int] | None = None) -> list[str]:
    return mutation_list_with_numbering(wt_seq, seq, residue_numbers=residue_numbers)


def mutation_list_with_numbering(wt_seq: str, seq: str, residue_numbers: list[int] | None = None) -> list[str]:
    muts = []
    labels = residue_numbers if residue_numbers is not None else list(range(1, len(wt_seq) + 1))
    for i, (wt, aa) in enumerate(zip(wt_seq, seq), start=1):
        if wt != aa:
            muts.append(f"{wt}{labels[i-1]}{aa}")
    return muts


def estimate_mutation_stability(wt_aa: str, mut_aa: str, feat: ResidueFeature | None = None) -> tuple[float, float]:
    """
    Estimate the stability impact of a mutation.

    Args:
        wt_aa: Wild-type amino acid
        mut_aa: Mutant amino acid
        feat: Residue feature with structural context (SASA, B-factor)

    Returns:
        Tuple of (delta_delta_G, confidence) where positive delta_delta_G is destabilizing.
    """
    key = (wt_aa, mut_aa)
    if key in MUTATION_STABILITY_ESTIMATE:
        return MUTATION_STABILITY_ESTIMATE[key]

    # Default estimation based on amino acid properties
    # If no specific mutation is known, use a simplified model
    wt_thermo = AA_THERMOSTABILITY.get(wt_aa, 0.0)
    mut_thermo = AA_THERMOSTABILITY.get(mut_aa, 0.0)
    base_delta = wt_thermo - mut_thermo  # Positive means mutation is less stable

    # Adjust based on structural context if available
    confidence = 0.4  # Lower confidence for estimated values
    if feat is not None:
        # High SASA (surface-exposed) mutations are less impactful
        if feat.sasa > 60:
            base_delta *= 0.5
            confidence = 0.5
        elif feat.sasa < 20:
            # Core mutations have higher impact
            base_delta *= 1.2
            confidence = 0.55

    return (base_delta, confidence)


def score_thermostability(
    wt_seq: str,
    seq: str,
    features: list[ResidueFeature],
    mutable_positions: list[int] | None = None,
    position_to_index: dict[int, int] | None = None,
    b_factor_weights: bool = True,
    core_penalty_factor: float = 1.2,
    surface_bonus_factor: float = 0.6,
) -> tuple[float, list[str]]:
    """
    Calculate thermostability score for a sequence variant.

    This function evaluates thermostability based on:
    1. Intrinsic amino acid thermostability contributions
    2. Mutation stability predictions
    3. Physicochemical property changes (volume, hydrophobicity, charge)
    4. B-factor weighted adjustments for flexible/rigid positions

    Args:
        wt_seq: Wild-type sequence
        seq: Variant sequence
        features: Residue features including B-factor and SASA
        mutable_positions: List of mutable positions
        position_to_index: Mapping from position number to sequence index
        b_factor_weights: Whether to apply B-factor based weighting
        core_penalty_factor: Multiplier for stability penalties in core positions
        surface_bonus_factor: Multiplier for stability bonuses on surface

    Returns:
        Tuple of (thermostability_score, notes)
    """
    notes: list[str] = []
    score = 0.0
    feat_map = {f.num: f for f in features}
    mutable_set = set(mutable_positions) if mutable_positions is not None else None

    # Calculate global thermostability profile
    muts = []
    for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, seq)):
        if wt_aa == mut_aa:
            continue

        # Get position info
        pos = i + 1
        if position_to_index is not None:
            pos = position_to_index.get(i, i + 1)
            # Reverse lookup: find the position number from index
            pos_map = {v: k for k, v in position_to_index.items()}
            pos = pos_map.get(i, i + 1)

        feat = feat_map.get(pos)
        muts.append((pos, wt_aa, mut_aa, feat))

    # If no explicit mutable positions, use all positions
    positions_to_score = mutable_positions if mutable_positions else list(range(1, len(wt_seq) + 1))

    for pos, wt_aa, mut_aa, feat in muts:
        if mutable_set is not None and pos not in mutable_set:
            continue

        # 1. Intrinsic thermostability contribution
        wt_thermo = AA_THERMOSTABILITY.get(wt_aa, 0.0)
        mut_thermo = AA_THERMOSTABILITY.get(mut_aa, 0.0)
        thermo_delta = mut_thermo - wt_thermo

        # 2. Mutation stability prediction
        delta_dg, confidence = estimate_mutation_stability(wt_aa, mut_aa, feat)
        stability_score = -delta_dg * confidence  # Negative delta_dg is stabilizing

        # 3. Volume change penalty (steric compatibility)
        wt_vol = AA_VOLUME.get(wt_aa, 100.0)
        mut_vol = AA_VOLUME.get(mut_aa, 100.0)
        vol_ratio = mut_vol / wt_vol if wt_vol > 0 else 1.0
        # Penalize extreme volume changes
        if vol_ratio < 0.6 or vol_ratio > 1.5:
            vol_penalty = -0.3
            notes.append(f"volume_change_{pos}")
        elif 0.7 < vol_ratio < 1.4:
            vol_penalty = 0.1  # Comfortable volume range
        else:
            vol_penalty = 0.0
        if abs(vol_ratio - 1.0) > 0.3:
            notes.append(f"steric_mismatch_{pos}")

        # 4. Hydrophobicity change (context-dependent)
        wt_hydro = AA_HYDROPHOBICITY.get(wt_aa, 0.0)
        mut_hydro = AA_HYDROPHOBICITY.get(mut_aa, 0.0)
        hydro_delta = mut_hydro - wt_hydro

        hydro_score = 0.0
        if feat is not None:
            if feat.sasa < 25:  # Core region
                # In core, maintain hydrophobicity
                if wt_hydro > 0.3 and mut_hydro < 0:
                    hydro_score = -0.4  # Penalty for making core polar
                elif wt_hydro < 0 and mut_hydro > 0.3:
                    hydro_score = -0.3  # Penalty for making core hydrophobic
                elif abs(hydro_delta) < 0.2:
                    hydro_score = 0.2  # Bonus for conservative hydrophobicity
            elif feat.sasa > 50:  # Surface region
                # On surface, allow more polar mutations
                if mut_aa in {"D", "E", "K", "R", "Q", "N", "S", "T"}:
                    hydro_score = 0.15  # Bonus for surface-appropriate residue
                elif abs(hydro_delta) < 0.3:
                    hydro_score = 0.1  # Bonus for conservative change

        # 5. Charge change
        wt_charge = AA_CHARGE.get(wt_aa, 0.0)
        mut_charge = AA_CHARGE.get(mut_aa, 0.0)
        charge_delta = abs(mut_charge - wt_charge)

        charge_score = 0.0
        if feat is not None:
            if feat.sasa < 25:  # Core - charge changes are very disruptive
                if charge_delta > 0.5:
                    charge_score = -0.4
                    notes.append(f"core_charge_change_{pos}")
            else:  # Surface - some charge changes acceptable
                if charge_delta > 1.0:
                    charge_score = -0.2

        # 6. B-factor weighting
        b_factor_multiplier = 1.0
        if b_factor_weights and feat is not None:
            # High B-factor = high flexibility = need more conservative mutations
            # Low B-factor = rigid = can tolerate some changes but stability matters more
            if feat.mean_b > 50:
                b_factor_multiplier = 1.5  # More conservative needed for flexible regions
                if abs(stability_score) > 0.5:
                    notes.append(f"flexible_stability_impact_{pos}")
            elif feat.mean_b < 20:
                b_factor_multiplier = 1.3  # Core stability is crucial
                if stability_score < -0.3:
                    notes.append(f"core_destabilization_{pos}")

        # Calculate combined score for this mutation
        mutation_score = (
            thermo_delta * 0.3 +
            stability_score * 0.35 +
            vol_penalty * 0.15 +
            hydro_score * 0.1 +
            charge_score * 0.1
        ) * b_factor_multiplier

        # Apply context-specific multipliers
        if feat is not None:
            if feat.sasa < 20:  # Core
                mutation_score *= core_penalty_factor
            elif feat.sasa > 60:  # Surface
                mutation_score *= surface_bonus_factor

        score += mutation_score

        # Add specific notes for significant changes
        if stability_score < -0.5:
            notes.append(f"destabilizing_mut_{pos}")
        elif stability_score > 0.5:
            notes.append(f"stabilizing_mut_{pos}")

    # Global thermostability assessment
    if len(muts) > 0:
        avg_thermo = score / len(muts)
        if avg_thermo > 0.3:
            notes.append("globally_stabilizing")
        elif avg_thermo < -0.3:
            notes.append("globally_destabilizing")

    return (round(score, 3), notes)


def score_candidate(
    wt_seq: str,
    seq: str,
    features: list[ResidueFeature],
    oxidation_hotspots: list[int],
    flexible_positions: list[int],
    profile: list[dict[str, float]] | None,
    asr_profile: list[dict[str, float]] | None,
    family_positive_profile: list[dict[str, float]] | None,
    family_negative_profile: list[dict[str, float]] | None,
    manual_preferred: dict[int, dict[str, float]],
    evolution_positions: list[int] | None = None,
    mutable_positions: list[int] | None = None,
    term_weights: dict[str, float] | None = None,
    position_to_index: dict[int, int] | None = None,
    evolution_position_weights: dict[int, float] | None = None,
    residue_numbers: list[int] | None = None,
    profile_prior_scale: float = 0.35,
    asr_prior_scale: float = 0.45,
    family_prior_scale: float = 0.60,
    topic_name: str | None = None,
    topic_cfg: dict[str, object] | None = None,
    enable_thermostability: bool = True,
    thermostability_weight: float = 0.8,
) -> ScoreResult:
    notes: list[str] = []
    weights = {
        "oxidation": 1.0,
        "surface": 1.0,
        "manual": 1.0,
        "evolution": 1.0,
        "burden": 1.0,
        "topic_sequence": 1.0,
        "topic_structure": 1.0,
        "topic_evolution": 1.0,
        "thermostability": thermostability_weight,
    }
    if term_weights:
        weights.update({k: float(v) for k, v in term_weights.items()})

    components = {
        "oxidation": 0.0,
        "surface": 0.0,
        "manual": 0.0,
        "evolution": 0.0,
        "burden": 0.0,
        "topic_sequence": 0.0,
        "topic_structure": 0.0,
        "topic_evolution": 0.0,
        "thermostability": 0.0,
    }
    feat_map = {f.num: f for f in features}
    mutable_set = set(mutable_positions) if mutable_positions is not None else None

    # Calculate thermostability score
    if enable_thermostability and features:
        thermo_score, thermo_notes = score_thermostability(
            wt_seq=wt_seq,
            seq=seq,
            features=features,
            mutable_positions=mutable_positions,
            position_to_index=position_to_index,
            b_factor_weights=True,
        )
        components["thermostability"] = thermo_score
        notes.extend(thermo_notes)
        if thermo_notes:
            notes.append("thermostability_evaluated")

    for pos in oxidation_hotspots:
        if mutable_set is not None and pos not in mutable_set:
            continue
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        wt = wt_seq[idx]
        aa = seq[idx]
        if aa == wt:
            components["oxidation"] -= 1.5
            notes.append(f"keeps_hotspot_{pos}")
        elif aa in SAFE_OXIDATION_MAP.get(wt, {}):
            components["oxidation"] += SAFE_OXIDATION_MAP[wt][aa]
            notes.append(f"hardens_hotspot_{pos}")
        elif aa in {"R", "K", "H", "C", "W", "Y", "M"}:
            components["oxidation"] -= 1.4
            notes.append(f"bad_hotspot_choice_{pos}")
        else:
            components["oxidation"] += 0.2

    for pos in flexible_positions:
        if mutable_set is not None and pos not in mutable_set:
            continue
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        wt = wt_seq[idx]
        aa = seq[idx]
        feat = feat_map[pos]
        if aa == wt:
            continue
        if aa in {"E", "D", "Q", "N", "S", "T"} and feat.sasa >= 40:
            components["surface"] += 0.7
            notes.append(f"surface_hydration_{pos}")
        if aa in {"W", "F", "Y", "C", "M"} and feat.sasa >= 40:
            components["surface"] -= 0.8
            notes.append(f"sticky_surface_{pos}")

    for pos, aa_bias in manual_preferred.items():
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        aa = seq[idx]
        if aa in aa_bias:
            components["manual"] += aa_bias[aa]
            notes.append(f"manual_bias_{pos}")

    evo_positions = evolution_positions or sorted(manual_preferred)
    evo = profile_log_score(
        seq,
        profile,
        evo_positions,
        position_to_index=position_to_index,
        position_weights=evolution_position_weights,
    )
    if profile is not None:
        components["evolution"] += profile_prior_scale * evo
        notes.append("evolution_prior")
    asr_evo = profile_log_score(
        seq,
        asr_profile,
        evo_positions,
        position_to_index=position_to_index,
        position_weights=evolution_position_weights,
    )
    if asr_profile is not None:
        components["evolution"] += asr_prior_scale * asr_evo
        notes.append("asr_prior")
    family_evo = differential_profile_score(
        seq,
        family_positive_profile,
        family_negative_profile,
        evo_positions,
        position_to_index=position_to_index,
        position_weights=evolution_position_weights,
    )
    if family_positive_profile is not None and family_negative_profile is not None:
        components["evolution"] += family_prior_scale * family_evo
        notes.append("family_evolution_prior")
    if evolution_position_weights:
        notes.append("template_weighted_evolution")

    muts = mutation_list_with_numbering(wt_seq, seq, residue_numbers=residue_numbers)
    components["burden"] -= 0.15 * max(0, len(muts) - 1)
    if len(muts) <= 2:
        notes.append("low_burden")

    topic_result = score_topic_candidate(
        topic_name=topic_name,
        wt_seq=wt_seq,
        seq=seq,
        features=features,
        mutable_positions=mutable_positions,
        position_to_index=position_to_index,
        profile=profile,
        asr_profile=asr_profile,
        family_positive_profile=family_positive_profile,
        family_negative_profile=family_negative_profile,
        topic_cfg=topic_cfg,
    )
    for key, value in topic_result.components.items():
        components[key] += value
    notes.extend(topic_result.notes)

    total = round(sum(components[k] * weights.get(k, 1.0) for k in components), 3)
    weighted_components = {k: round(components[k] * weights.get(k, 1.0), 3) for k in components}
    return ScoreResult(total, notes, weighted_components)
