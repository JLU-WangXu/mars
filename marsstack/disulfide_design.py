"""
Disulfide Bond Prediction and Design Module for Thermostability.

This module provides functionality for:
- Predicting disulfide bond formation potential
- Filtering candidate pairs based on distance and angle constraints
- Scoring disulfide bond contribution to stability
- Automatic disulfide bond design suggestions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# =============================================================================
# Distance and Angle Thresholds
# =============================================================================

# Ideal S-S distance for disulfide bond (Angstroms)
IDEAL_SS_DISTANCE: float = 2.05
MIN_SS_DISTANCE: float = 1.9
MAX_SS_DISTANCE: float = 2.4

# CB-CB distance constraints for disulfide formation
MIN_CB_DISTANCE: float = 3.5
MAX_CB_DISTANCE: float = 6.5

# Dihedral angle constraints for disulfide (C-CA-CB-S)
MIN_CHI_DIHEDRAL: float = -120.0
MAX_CHI_DIHEDRAL: float = 120.0

# Ideal chi angle for disulfide
IDEAL_CHI1: float = -60.0  # +/- 30 degrees
IDEAL_CHI2: float = 90.0   # +/- 30 degrees

# Stability scores
DISULFIDE_FORMATION_SCORE: float = 3.5
DISULFIDE_GEOMETRY_BONUS: float = 1.5
DISULFIDE_CONTEXT_BONUS: float = 1.0

# Penalty factors
STRAIN_PENALTY: float = -2.0
ANGLE_DEVIATION_PENALTY: float = -0.5
DISTANCE_DEVIATION_PENALTY: float = -0.8


# =============================================================================
# Secondary Structure Context
# =============================================================================

# Secondary structure types that favor disulfide formation
FAVORABLE_SECONDARY_STRUCTURES: set[str] = {"H", "E", "T"}  # Helix, Sheet, Turn
UNFAVORABLE_SECONDARY_STRUCTURES: set[str] = {"G", "P"}     # 3-helix, Polyproline

# Loop regions are generally more permissive for disulfides
LOOP_REGIONS: set[str] = {"L", "C", "S"}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DisulfideGeometry:
    """Geometry parameters for a disulfide bond."""
    ss_distance: float          # S-S distance
    cb_cb_distance: float        # CB-CB distance
    chi1_i: float               # Dihedral angle CA-CB-S-S for residue i
    chi1_j: float               # Dihedral angle CA-CB-S-S for residue j
    omega: float                # C-CA-CB-S dihedral for disulfide
    ca_distance: float          # CA-CA distance
    is_ideal: bool = False
    geometry_score: float = 0.0


@dataclass
class DisulfidePrediction:
    """Prediction result for a potential disulfide bond."""
    pos_i: int
    pos_j: int
    geometry: DisulfideGeometry
    formation_probability: float
    stability_score: float
    sequence_context_i: str = ""
    sequence_context_j: str = ""
    secondary_structure_i: str = ""
    secondary_structure_j: str = ""
    recommendations: list[str] = field(default_factory=list)


@dataclass
class DisulfideDesignResult:
    """Result of automatic disulfide design."""
    suggested_mutations: list[DisulfideMutation]
    predicted_disulfides: list[DisulfidePrediction]
    total_stability_gain: float
    confidence: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class DisulfideMutation:
    """A mutation to introduce a disulfide bond."""
    position: int
    from_residue: str
    to_residue: str
    partner_position: int
    expected_distance: float
    priority: float
    reason: str


@dataclass
class DisulfideConstraint:
    """Constraint for disulfide bond in design."""
    pos_i: int
    pos_j: int
    required: bool = True
    min_distance: float = MIN_SS_DISTANCE
    max_distance: float = MAX_SS_DISTANCE
    weight: float = 1.0


# =============================================================================
# Geometry Calculation Functions
# =============================================================================


def calculate_ss_distance(
    pos_i_coords: tuple[float, float, float],
    pos_j_coords: tuple[float, float, float],
) -> float:
    """
    Calculate sulfur-sulfur distance between two cysteines.

    Args:
        pos_i_coords: (x, y, z) coordinates of S atom at position i
        pos_j_coords: (x, y, z) coordinates of S atom at position j

    Returns:
        Distance in Angstroms
    """
    dx = pos_i_coords[0] - pos_j_coords[0]
    dy = pos_i_coords[1] - pos_j_coords[1]
    dz = pos_i_coords[2] - pos_j_coords[2]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_cb_cb_distance(
    pos_i_coords: tuple[float, float, float],
    pos_j_coords: tuple[float, float, float],
) -> float:
    """
    Calculate CB-CB distance between two residues.

    Args:
        pos_i_coords: (x, y, z) coordinates of CB atom at position i
        pos_j_coords: (x, y, z) coordinates of CB atom at position j

    Returns:
        Distance in Angstroms
    """
    dx = pos_i_coords[0] - pos_j_coords[0]
    dy = pos_i_coords[1] - pos_j_coords[1]
    dz = pos_i_coords[2] - pos_j_coords[2]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_ca_distance(
    pos_i_coords: tuple[float, float, float],
    pos_j_coords: tuple[float, float, float],
) -> float:
    """
    Calculate CA-CA distance between two residues.

    Args:
        pos_i_coords: (x, y, z) coordinates of CA atom at position i
        pos_j_coords: (x, y, z) coordinates of CA atom at position j

    Returns:
        Distance in Angstroms
    """
    dx = pos_i_coords[0] - pos_j_coords[0]
    dy = pos_i_coords[1] - pos_j_coords[1]
    dz = pos_i_coords[2] - pos_j_coords[2]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_dihedral(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
) -> float:
    """
    Calculate dihedral angle (in degrees) between four points.

    Args:
        p0, p1, p2, p3: Four points defining the dihedral

    Returns:
        Dihedral angle in degrees
    """
    b1 = np.array([p1[i] - p0[i] for i in range(3)])
    b2 = np.array([p2[i] - p1[i] for i in range(3)])
    b3 = np.array([p3[i] - p2[i] for i in range(3)])

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-6 or n2_norm < 1e-6:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    angle = np.arctan2(y, x)
    return np.degrees(angle)


def analyze_disulfide_geometry(
    coords_i: dict[str, tuple[float, float, float]],
    coords_j: dict[str, tuple[float, float, float]],
) -> DisulfideGeometry:
    """
    Analyze the geometry of a potential disulfide bond.

    Args:
        coords_i: Dict with keys 'N', 'CA', 'CB', 'S' for residue i
        coords_j: Dict with keys 'N', 'CA', 'CB', 'S' for residue j

    Returns:
        DisulfideGeometry object with all parameters
    """
    ss_distance = calculate_ss_distance(coords_i['S'], coords_j['S'])
    cb_cb_distance = calculate_cb_cb_distance(coords_i['CB'], coords_j['CB'])
    ca_distance = calculate_ca_distance(coords_i['CA'], coords_j['CA'])

    # Calculate chi1 dihedral: N-CA-CB-S for each residue
    chi1_i = calculate_dihedral(
        coords_i['N'], coords_i['CA'], coords_i['CB'], coords_i['S']
    )
    chi1_j = calculate_dihedral(
        coords_j['N'], coords_j['CA'], coords_j['CB'], coords_j['S']
    )

    # Calculate omega (disulfide dihedral): CA-CB-S-S
    omega = calculate_dihedral(
        coords_i['CA'], coords_i['CB'], coords_i['S'], coords_j['S']
    )

    # Evaluate geometry quality
    is_ideal = (
        MIN_SS_DISTANCE <= ss_distance <= MAX_SS_DISTANCE and
        MIN_CB_DISTANCE <= cb_cb_distance <= MAX_CB_DISTANCE and
        abs(chi1_i - IDEAL_CHI1) <= 45 and
        abs(chi1_j - IDEAL_CHI1) <= 45
    )

    geometry_score = compute_geometry_score(
        ss_distance, cb_cb_distance, chi1_i, chi1_j, omega
    )

    return DisulfideGeometry(
        ss_distance=ss_distance,
        cb_cb_distance=cb_cb_distance,
        chi1_i=chi1_i,
        chi1_j=chi1_j,
        omega=omega,
        ca_distance=ca_distance,
        is_ideal=is_ideal,
        geometry_score=geometry_score,
    )


def compute_geometry_score(
    ss_distance: float,
    cb_cb_distance: float,
    chi1_i: float,
    chi1_j: float,
    omega: float,
) -> float:
    """
    Compute geometry quality score for disulfide bond.

    Args:
        ss_distance: S-S distance
        cb_cb_distance: CB-CB distance
        chi1_i: Chi1 angle for residue i
        chi1_j: Chi1 angle for residue j
        omega: Disulfide dihedral angle

    Returns:
        Geometry score (higher is better)
    """
    score = 0.0

    # S-S distance scoring (Gaussian-like)
    ss_deviation = abs(ss_distance - IDEAL_SS_DISTANCE)
    if ss_deviation <= 0.1:
        score += 1.0
    elif ss_deviation <= 0.2:
        score += 0.7
    elif ss_deviation <= 0.35:
        score += 0.3
    else:
        score += DISTANCE_DEVIATION_PENALTY * ss_deviation

    # CB-CB distance scoring
    if MIN_CB_DISTANCE <= cb_cb_distance <= MAX_CB_DISTANCE:
        score += 0.5
    else:
        score -= 0.5

    # Chi1 angle scoring
    chi1_i_deviation = min(abs(chi1_i - IDEAL_CHI1), abs(chi1_i - IDEAL_CHI1 + 360))
    chi1_j_deviation = min(abs(chi1_j - IDEAL_CHI1), abs(chi1_j - IDEAL_CHI1 + 360))

    if chi1_i_deviation <= 30:
        score += 0.5
    elif chi1_i_deviation <= 60:
        score += 0.2
    else:
        score += ANGLE_DEVIATION_PENALTY

    if chi1_j_deviation <= 30:
        score += 0.5
    elif chi1_j_deviation <= 60:
        score += 0.2
    else:
        score += ANGLE_DEVIATION_PENALTY

    # Omega angle scoring (ideal is ~90 or -90)
    omega_abs = abs(omega)
    if 60 <= omega_abs <= 120:
        score += 0.3
    elif 30 <= omega_abs <= 150:
        score += 0.1
    else:
        score -= 0.2

    return score


# =============================================================================
# Disulfide Formation Prediction
# =============================================================================


def predict_disulfide_formation(
    pos_i: int,
    pos_j: int,
    coords_i: dict[str, tuple[float, float, float]],
    coords_j: dict[str, tuple[float, float, float]],
    secondary_structure_i: str = "",
    secondary_structure_j: str = "",
    sasa_i: float = 50.0,
    sasa_j: float = 50.0,
    existing_disulfides: set[tuple[int, int]] | None = None,
) -> DisulfidePrediction:
    """
    Predict if two positions can form a disulfide bond.

    Args:
        pos_i: Position index of first cysteine
        pos_j: Position index of second cysteine
        coords_i: Atomic coordinates for residue i
        coords_j: Atomic coordinates for residue j
        secondary_structure_i: DSSP secondary structure at pos i
        secondary_structure_j: DSSP secondary structure at pos j
        sasa_i: Solvent accessible surface area at pos i
        sasa_j: Solvent accessible surface area at pos j
        existing_disulfides: Set of existing disulfide pairs

    Returns:
        DisulfidePrediction with formation probability and stability score
    """
    geometry = analyze_disulfide_geometry(coords_i, coords_j)
    recommendations: list[str] = []

    # Check if already in existing disulfides
    existing_disulfides = existing_disulfides or set()
    pair = tuple(sorted([pos_i, pos_j]))
    if pair in existing_disulfides:
        return DisulfidePrediction(
            pos_i=pos_i,
            pos_j=pos_j,
            geometry=geometry,
            formation_probability=1.0,
            stability_score=DISULFIDE_FORMATION_SCORE,
            secondary_structure_i=secondary_structure_i,
            secondary_structure_j=secondary_structure_j,
            recommendations=["Existing disulfide bond confirmed"],
        )

    # Base probability from geometry
    if not (MIN_SS_DISTANCE <= geometry.ss_distance <= MAX_SS_DISTANCE):
        return DisulfidePrediction(
            pos_i=pos_i,
            pos_j=pos_j,
            geometry=geometry,
            formation_probability=0.0,
            stability_score=0.0,
            recommendations=["S-S distance out of range for disulfide formation"],
        )

    if not (MIN_CB_DISTANCE <= geometry.cb_cb_distance <= MAX_CB_DISTANCE):
        return DisulfidePrediction(
            pos_i=pos_i,
            pos_j=pos_j,
            geometry=geometry,
            formation_probability=0.0,
            stability_score=0.0,
            recommendations=["CB-CB distance unsuitable for disulfide"],
        )

    # Calculate formation probability
    formation_prob = 0.5  # Base probability

    # Adjust for geometry
    geometry_factor = min(1.0, geometry.geometry_score / 3.0)
    formation_prob *= (0.4 + 0.6 * geometry_factor)

    # Adjust for secondary structure context
    if secondary_structure_i in FAVORABLE_SECONDARY_STRUCTURES:
        formation_prob *= 1.1
    if secondary_structure_j in FAVORABLE_SECONDARY_STRUCTURES:
        formation_prob *= 1.1
    if secondary_structure_i in UNFAVORABLE_SECONDARY_STRUCTURES:
        formation_prob *= 0.7
    if secondary_structure_j in UNFAVORABLE_SECONDARY_STRUCTURES:
        formation_prob *= 0.7

    # Adjust for solvent exposure
    if sasa_i < 30 or sasa_j < 30:
        formation_prob *= 1.2  # Buried cysteines more likely to form disulfides
        recommendations.append("Favorable: Buried cysteine position")
    elif sasa_i > 100 or sasa_j > 100:
        formation_prob *= 0.8  # Exposed cysteines may form disulfides but less stable
        recommendations.append("Warning: Exposed cysteine may reduce stability")

    # Cap probability
    formation_prob = min(1.0, max(0.0, formation_prob))

    # Calculate stability score
    stability_score = DISULFIDE_FORMATION_SCORE

    if geometry.is_ideal:
        stability_score += DISULFIDE_GEOMETRY_BONUS
        recommendations.append("Excellent geometry for disulfide")

    # Distance bonus
    ss_deviation = abs(geometry.ss_distance - IDEAL_SS_DISTANCE)
    if ss_deviation <= 0.1:
        stability_score += 0.5
        recommendations.append("Near-ideal S-S distance")

    # Context bonus for buried positions
    if sasa_i < 50 and sasa_j < 50:
        stability_score += DISULFIDE_CONTEXT_BONUS
        recommendations.append("Both cysteines in hydrophobic core")

    return DisulfidePrediction(
        pos_i=pos_i,
        pos_j=pos_j,
        geometry=geometry,
        formation_probability=formation_prob,
        stability_score=stability_score,
        secondary_structure_i=secondary_structure_i,
        secondary_structure_j=secondary_structure_j,
        recommendations=recommendations,
    )


def filter_candidate_pairs(
    candidates: list[tuple[int, int, float]],
    coords_map: dict[int, dict[str, tuple[float, float, float]]],
    min_probability: float = 0.3,
    min_ca_distance: float = 3.5,
    max_ca_distance: float = 10.0,
) -> list[tuple[int, int, float]]:
    """
    Filter candidate cysteine pairs based on geometric constraints.

    Args:
        candidates: List of (pos_i, pos_j, base_score) tuples
        coords_map: Dict mapping position to atomic coordinates
        min_probability: Minimum formation probability threshold
        min_ca_distance: Minimum CA-CA distance
        max_ca_distance: Maximum CA-CA distance

    Returns:
        Filtered list of viable candidates with updated scores
    """
    filtered = []

    for pos_i, pos_j, base_score in candidates:
        if pos_i not in coords_map or pos_j not in coords_map:
            continue

        coords_i = coords_map[pos_i]
        coords_j = coords_map[pos_j]

        # Check required atoms
        for atom in ['CA', 'CB', 'S']:
            if atom not in coords_i or atom not in coords_j:
                continue

        # Calculate distances
        ss_dist = calculate_ss_distance(coords_i['S'], coords_j['S'])
        ca_dist = calculate_ca_distance(coords_i['CA'], coords_j['CA'])

        # Filter by basic geometry
        if not (MIN_SS_DISTANCE <= ss_dist <= MAX_SS_DISTANCE):
            continue

        if not (min_ca_distance <= ca_dist <= max_ca_distance):
            continue

        # Calculate geometry score for ranking
        geometry = analyze_disulfide_geometry(coords_i, coords_j)
        combined_score = base_score + geometry.geometry_score * 0.5

        # Estimate probability
        prob = min(1.0, geometry.geometry_score / 2.0)
        if prob >= min_probability:
            filtered.append((pos_i, pos_j, combined_score))

    # Sort by score
    filtered.sort(key=lambda x: -x[2])
    return filtered


# =============================================================================
# Automatic Disulfide Design
# =============================================================================


def design_disulfide_bonds(
    sequence: str,
    coords_map: dict[int, dict[str, tuple[float, float, float]]],
    existing_disulfides: set[tuple[int, int]] | None = None,
    secondary_structure: dict[int, str] | None = None,
    sasa_map: dict[int, float] | None = None,
    target_count: int = 3,
    min_separation: int = 6,
) -> DisulfideDesignResult:
    """
    Automatically design disulfide bonds for thermostability.

    Args:
        sequence: Full amino acid sequence
        coords_map: Dict mapping position to atomic coordinates
        existing_disulfides: Set of existing disulfide pairs
        secondary_structure: Dict mapping position to DSSP code
        sasa_map: Dict mapping position to SASA value
        target_count: Target number of disulfide bonds to design
        min_separation: Minimum sequence separation for disulfide pairs

    Returns:
        DisulfideDesignResult with suggested mutations
    """
    existing_disulfides = existing_disulfides or set()
    secondary_structure = secondary_structure or {}
    sasa_map = sasa_map or {}

    # Find all potential cysteine positions
    potential_positions: list[int] = []
    for pos, residue in enumerate(sequence, 1):
        if residue == 'C':
            potential_positions.append(pos)

    # Score all possible pairs
    pair_scores: list[DisulfidePrediction] = []

    for i, pos_i in enumerate(potential_positions):
        for pos_j in potential_positions[i + 1:]:
            # Check minimum separation
            if abs(pos_j - pos_i) < min_separation:
                continue

            # Check not already a disulfide
            pair = tuple(sorted([pos_i, pos_j]))
            if pair in existing_disulfides:
                continue

            if pos_i not in coords_map or pos_j not in coords_map:
                continue

            coords_i = coords_map[pos_i]
            coords_j = coords_map[pos_j]

            # Predict disulfide formation
            prediction = predict_disulfide_formation(
                pos_i=pos_i,
                pos_j=pos_j,
                coords_i=coords_i,
                coords_j=coords_j,
                secondary_structure_i=secondary_structure.get(pos_i, ""),
                secondary_structure_j=secondary_structure.get(pos_j, ""),
                sasa_i=sasa_map.get(pos_i, 50.0),
                sasa_j=sasa_map.get(pos_j, 50.0),
                existing_disulfides=existing_disulfides,
            )

            if prediction.formation_probability > 0.3:
                pair_scores.append(prediction)

    # Sort by stability score
    pair_scores.sort(key=lambda x: -x.stability_score)

    # Select top pairs ensuring no position conflicts
    selected_pairs: list[DisulfidePrediction] = []
    used_positions: set[int] = set()

    for prediction in pair_scores:
        if len(selected_pairs) >= target_count:
            break

        if prediction.pos_i not in used_positions and prediction.pos_j not in used_positions:
            selected_pairs.append(prediction)
            used_positions.add(prediction.pos_i)
            used_positions.add(prediction.pos_j)

    # Generate mutation suggestions for non-cysteine positions
    mutations: list[DisulfideMutation] = []
    warnings: list[str] = []

    # If we need more disulfides, suggest mutating to cysteine
    if len(selected_pairs) < target_count:
        # Find positions with good geometry but not cysteine
        for pos_i, pos_j in zip([p.pos_i for p in selected_pairs],
                                [p.pos_j for p in selected_pairs]):
            pass  # Already handled above

        # For positions that could form disulfides but aren't cysteines
        remaining_target = target_count - len(selected_pairs)
        if remaining_target > 0:
            warnings.append(
                f"Could only identify {len(selected_pairs)} native disulfide candidates. "
                f"Consider mutating suitable positions to cysteine."
            )

    # Calculate total stability gain
    total_stability_gain = sum(p.stability_score for p in selected_pairs)
    confidence = sum(p.formation_probability for p in selected_pairs) / max(len(selected_pairs), 1)

    return DisulfideDesignResult(
        suggested_mutations=mutations,
        predicted_disulfides=selected_pairs,
        total_stability_gain=total_stability_gain,
        confidence=confidence,
        warnings=warnings,
    )


def suggest_cysteine_mutations(
    sequence: str,
    coords_map: dict[int, dict[str, tuple[float, float, float]]],
    partner_pos: int,
    partner_coords: dict[str, tuple[float, float, float]],
    existing_disulfides: set[tuple[int, int]] | None = None,
    exclude_positions: set[int] | None = None,
) -> list[DisulfideMutation]:
    """
    Suggest positions to mutate to cysteine to form a disulfide with partner.

    Args:
        sequence: Full amino acid sequence
        coords_map: Dict mapping position to atomic coordinates
        partner_pos: Position of the partner cysteine
        partner_coords: Atomic coordinates of partner
        existing_disulfides: Set of existing disulfide pairs
        exclude_positions: Positions to exclude from consideration

    Returns:
        List of DisulfideMutation suggestions
    """
    existing_disulfides = existing_disulfides or set()
    exclude_positions = exclude_positions or set()

    mutations: list[DisulfideMutation] = []

    # Check if partner is already in a disulfide
    partner_in_disulfide = any(
        partner_pos in pair for pair in existing_disulfides
    )
    if partner_in_disulfide:
        return mutations

    # Score each position for mutation potential
    for pos in range(1, len(sequence) + 1):
        if pos in exclude_positions:
            continue

        if pos not in coords_map:
            continue

        residue = sequence[pos - 1]
        if residue == 'C':
            continue  # Already cysteine

        coords = coords_map[pos]

        # Check required atoms
        for atom in ['CA', 'CB', 'N']:
            if atom not in coords:
                continue

        # Calculate geometry
        ss_dist = calculate_ss_distance(coords['S'], partner_coords['S'])
        ca_dist = calculate_ca_distance(coords['CA'], partner_coords['CA'])

        # Check if geometry is feasible
        if not (MIN_SS_DISTANCE - 0.5 <= ss_dist <= MAX_SS_DISTANCE + 1.0):
            continue

        if not (MIN_CB_DISTANCE <= ca_dist <= MAX_CB_DISTANCE + 2.0):
            continue

        # Calculate priority score
        priority = 0.0

        # Distance factor
        ss_deviation = abs(ss_dist - IDEAL_SS_DISTANCE)
        priority += max(0, 1.0 - ss_deviation * 2)

        # Small residue preference (easier to accommodate)
        if residue in {'G', 'A', 'S'}:
            priority += 0.3
        elif residue in {'V', 'T', 'N'}:
            priority += 0.1

        # Solvent exposure factor
        if ca_dist < 5.0:
            priority += 0.2

        mutation = DisulfideMutation(
            position=pos,
            from_residue=residue,
            to_residue='C',
            partner_position=partner_pos,
            expected_distance=ss_dist,
            priority=priority,
            reason=f"Suitable geometry for disulfide with C{partner_pos}"
        )
        mutations.append(mutation)

    # Sort by priority
    mutations.sort(key=lambda x: -x.priority)
    return mutations


# =============================================================================
# Decoder Integration
# =============================================================================


def apply_disulfide_constraints(
    sequence: str,
    disulfide_constraints: list[DisulfideConstraint],
    coords_map: dict[int, dict[str, tuple[float, float, float]]],
    existing_disulfides: set[tuple[int, int]] | None = None,
) -> tuple[bool, float, list[str]]:
    """
    Apply disulfide constraints to a sequence.

    Args:
        sequence: Full amino acid sequence
        disulfide_constraints: List of disulfide constraints
        coords_map: Dict mapping position to atomic coordinates
        existing_disulfides: Set of existing disulfide pairs

    Returns:
        Tuple of (all_satisfied: bool, score: float, violations: list[str])
    """
    existing_disulfides = existing_disulfides or set()
    violations: list[str] = []
    total_score: float = 0.0

    for constraint in disulfide_constraints:
        pos_i = constraint.pos_i
        pos_j = constraint.pos_j

        # Get residues at positions
        residue_i = sequence[pos_i - 1] if pos_i <= len(sequence) else ""
        residue_j = sequence[pos_j - 1] if pos_j <= len(sequence) else ""

        # Check if both are cysteines
        if residue_i != 'C' or residue_j != 'C':
            if constraint.required:
                violations.append(f"Required disulfide C{pos_i}-C{pos_j} not formed")
                total_score += STRAIN_PENALTY
            continue

        # Check if already formed
        pair = tuple(sorted([pos_i, pos_j]))
        if pair in existing_disulfides:
            total_score += DISULFIDE_FORMATION_SCORE * constraint.weight
            continue

        # Calculate distance if coordinates available
        if pos_i in coords_map and pos_j in coords_map:
            ss_dist = calculate_ss_distance(
                coords_map[pos_i]['S'], coords_map[pos_j]['S']
            )

            if not (constraint.min_distance <= ss_dist <= constraint.max_distance):
                if constraint.required:
                    violations.append(
                        f"Disulfide C{pos_i}-C{pos_j} distance {ss_dist:.2f} out of range"
                    )
                total_score += DISTANCE_DEVIATION_PENALTY
            else:
                total_score += DISULFIDE_FORMATION_SCORE * constraint.weight
        else:
            # No coordinates, assume constraint satisfied
            total_score += DISULFIDE_FORMATION_SCORE * constraint.weight

    return len(violations) == 0, total_score, violations


def compute_disulfide_penalty(
    sequence: str,
    position_to_index: dict[int, int],
    field_positions: list[int],
    disulfide_constraints: list[DisulfideConstraint],
    coords_map: dict[int, dict[str, tuple[float, float, float]]],
    existing_disulfides: set[tuple[int, int]] | None = None,
) -> float:
    """
    Compute penalty for disulfide constraints during beam search.

    Args:
        sequence: Current sequence (tuple of chars)
        position_to_index: Mapping from position to sequence index
        field_positions: List of design positions
        disulfide_constraints: List of disulfide constraints
        coords_map: Dict mapping position to atomic coordinates
        existing_disulfides: Set of existing disulfide pairs

    Returns:
        Penalty score (negative = favorable)
    """
    seq_str = "".join(sequence) if isinstance(sequence, tuple) else sequence
    satisfied, score, violations = apply_disulfide_constraints(
        seq_str, disulfide_constraints, coords_map, existing_disulfides
    )

    penalty = -score
    penalty += len(violations) * 1.0  # Additional penalty per violation

    return penalty


# =============================================================================
# Utility Functions
# =============================================================================


def get_disulfide_summary(prediction: DisulfidePrediction) -> str:
    """Generate a human-readable summary of disulfide prediction."""
    lines = [
        f"Disulfide C{prediction.pos_i}-C{prediction.pos_j}:",
        f"  S-S distance: {prediction.geometry.ss_distance:.2f} A",
        f"  CB-CB distance: {prediction.geometry.cb_cb_distance:.2f} A",
        f"  Formation probability: {prediction.formation_probability:.1%}",
        f"  Stability score: {prediction.stability_score:.2f}",
    ]

    if prediction.recommendations:
        lines.append("  Recommendations:")
        for rec in prediction.recommendations:
            lines.append(f"    - {rec}")

    return "\n".join(lines)


def get_design_summary(result: DisulfideDesignResult) -> str:
    """Generate a human-readable summary of design result."""
    lines = [
        f"Disulfide Design Result:",
        f"  Disulfides designed: {len(result.predicted_disulfides)}",
        f"  Total stability gain: {result.total_stability_gain:.2f}",
        f"  Confidence: {result.confidence:.1%}",
    ]

    if result.warnings:
        lines.append("  Warnings:")
        for warning in result.warnings:
            lines.append(f"    - {warning}")

    if result.predicted_disulfides:
        lines.append("  Predicted disulfides:")
        for pred in result.predicted_disulfides:
            lines.append(
                f"    - C{pred.pos_i}-C{pred.pos_j} "
                f"(prob={pred.formation_probability:.1%}, score={pred.stability_score:.2f})"
            )

    return "\n".join(lines)
