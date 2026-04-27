from __future__ import annotations

"""
Physics-based constraint validation module for MARS-FIELD.

Provides validation functions for:
- Hydrogen bond formation
- Salt bridge formation
- Disulfide bond stability
- Hydrophobic core / hydrophilic surface verification
- Charge clash detection
- Backbone dihedral angle feasibility
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


# =============================================================================
# Amino Acid Property Definitions
# =============================================================================

# Hydrogen bond donors (N, O atoms that can donate H)
HBOND_DONORS: set[str] = {"N", "O"}

# Hydrogen bond acceptors (O, N atoms that can accept H)
HBOND_ACCEPTORS: set[str] = {"O", "N"}

# Residues capable of forming hydrogen bonds (side chains)
HBOND_RESIDUES: set[str] = {
    "S", "T", "Y",  # Ser, Thr, Tyr (hydroxyl)
    "N", "Q",       # Asn, Gln (amide)
    "D", "E",       # Asp, Glu (carboxyl)
    "R", "K", "H",  # Arg, Lys, His (basic)
}

# Residues that can participate in salt bridges
SALT_BRIDGE_POSITIVE: set[str] = {"R", "K", "H"}  # Arg, Lys, His
SALT_BRIDGE_NEGATIVE: set[str] = {"D", "E"}  # Asp, Glu

# Hydrophobic residues (favor core)
HYDROPHOBIC: set[str] = {"A", "V", "I", "L", "M", "F", "W", "P", "G"}

# Hydrophilic/polar residues (favor surface)
HYDROPHILIC: set[str] = {"S", "T", "N", "Q", "D", "E", "R", "K", "H", "Y", "C"}

# Charged residues
CHARGED_POSITIVE: set[str] = {"R", "K", "H"}
CHARGED_NEGATIVE: set[str] = {"D", "E"}

# Cysteine residues (for disulfide bonds)
CYSTEINE: set[str] = {"C"}

# =============================================================================
# Distance Thresholds (Angstroms)
# =============================================================================

HBOND_DISTANCE_MAX: float = 3.5  # Maximum H-bond distance
SALT_BRIDGE_DISTANCE_MAX: float = 4.0  # Maximum salt bridge distance
DISULFIDE_DISTANCE_MIN: float = 2.0  # Minimum for disulfide (S-S)
DISULFIDE_DISTANCE_MAX: float = 2.5  # Maximum for stable disulfide (S-S)
CLASH_DISTANCE: float = 2.5  # Distance for charge clash detection

# =============================================================================
# Dihedral Angle Constraints (degrees) - Ramachandran regions
# =============================================================================

# Allowed regions for general amino acids (phi, psi)
RAMACHANDRAN_ALPHA: tuple[tuple[float, float], tuple[float, float]] = (
    (-180.0, -30.0), (-60.0, 60.0)
)  # Alpha helix region

RAMACHANDRAN_BETA: tuple[tuple[float, float], tuple[float, float]] = (
    (-180.0, -170.0), (100.0, 180.0)
)  # Beta sheet region

# Proline has restricted phi
PROLINE_PHI_RANGE: tuple[float, float] = (-75.0, -50.0)
PROLINE_PSI_RANGE: tuple[float, float] = (-10.0, 180.0)

# Glycine has wider range
GLYCINE_PHI_RANGE: tuple[float, float] = (-180.0, 180.0)
GLYCINE_PSI_RANGE: tuple[float, float] = (-180.0, 180.0)

# =============================================================================
# Score Weights
# =============================================================================

HBOND_STABILITY_SCORE: float = 2.0
SALT_BRIDGE_STABILITY_SCORE: float = 2.5
DISULFIDE_STABILITY_SCORE: float = 3.0
HYDROPHOBIC_CORE_SCORE: float = 1.5
HYDROPHILIC_SURFACE_SCORE: float = 1.2
CLASH_PENALTY: float = -5.0
HBOND_PENALTY: float = -1.5


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ConstraintResult:
    """Result of constraint validation for a candidate sequence."""

    total_score: float
    violations: list[str]
    warnings: list[str]
    details: dict[str, Any]


@dataclass
class PairConstraint:
    """Constraint between a pair of positions."""

    pos_i: int
    pos_j: int
    constraint_type: str  # "hbond", "salt_bridge", "disulfide", "charge_clash"
    residue_i: str
    residue_j: str
    distance: float
    score: float


# =============================================================================
# Validation Functions
# =============================================================================


def validate_hbond_formation(
    pos_i: int,
    pos_j: int,
    residue_i: str,
    residue_j: str,
    distance: float,
) -> tuple[bool, float]:
    """
    Validate if two residues can form a hydrogen bond.

    Args:
        pos_i: Position index of first residue
        pos_j: Position index of second residue
        residue_i: Amino acid at position i
        residue_j: Amino acid at position j
        distance: Distance between potential H-bonding atoms

    Returns:
        Tuple of (can_form: bool, score: float)
    """
    if residue_i not in HBOND_RESIDUES or residue_j not in HBOND_RESIDUES:
        return False, 0.0

    if distance > HBOND_DISTANCE_MAX:
        return False, 0.0

    # Strong H-bond donors/acceptors
    strong_pairs = {
        ("D", "R"), ("E", "R"), ("D", "K"), ("E", "K"),
        ("D", "H"), ("E", "H"), ("N", "R"), ("Q", "R"),
        ("S", "R"), ("T", "R"), ("Y", "R"),
    }

    if (residue_i, residue_j) in strong_pairs or (residue_j, residue_i) in strong_pairs:
        # Ideal distance for strong H-bonds: 2.8-3.2 Angstroms
        if 2.5 <= distance <= 3.2:
            return True, HBOND_STABILITY_SCORE * 1.2
        else:
            return True, HBOND_STABILITY_SCORE * 0.8

    # Moderate H-bond pairs
    moderate_pairs = {
        ("S", "N"), ("T", "N"), ("S", "Q"), ("T", "Q"),
        ("N", "Y"), ("Q", "Y"), ("N", "S"), ("Q", "S"),
        ("Y", "S"), ("Y", "T"),
    }

    if (residue_i, residue_j) in moderate_pairs or (residue_j, residue_i) in moderate_pairs:
        return True, HBOND_STABILITY_SCORE * 0.7

    # Weak H-bonds
    return True, HBOND_STABILITY_SCORE * 0.4


def validate_salt_bridge(
    pos_i: int,
    pos_j: int,
    residue_i: str,
    residue_j: str,
    distance: float,
) -> tuple[bool, float]:
    """
    Validate if two residues can form a salt bridge.

    Args:
        pos_i: Position index of first residue
        pos_j: Position index of second residue
        residue_i: Amino acid at position i
        residue_j: Amino acid at position j
        distance: Distance between residues

    Returns:
        Tuple of (can_form: bool, score: float)
    """
    # Check for oppositely charged residues
    pos_charged = residue_i in SALT_BRIDGE_POSITIVE and residue_j in SALT_BRIDGE_NEGATIVE
    neg_charged = residue_i in SALT_BRIDGE_NEGATIVE and residue_j in SALT_BRIDGE_POSITIVE

    if not (pos_charged or neg_charged):
        return False, 0.0

    if distance > SALT_BRIDGE_DISTANCE_MAX:
        return False, 0.0

    # Ideal salt bridge distance: 3.0-3.5 Angstroms
    if distance <= 3.5:
        return True, SALT_BRIDGE_STABILITY_SCORE
    else:
        return True, SALT_BRIDGE_STABILITY_SCORE * 0.6


def validate_disulfide_bond(
    pos_i: int,
    pos_j: int,
    residue_i: str,
    residue_j: str,
    distance: float,
    existing_disulfides: set[tuple[int, int]],
) -> tuple[bool, float]:
    """
    Validate if two cysteines can form a disulfide bond.

    Args:
        pos_i: Position index of first cysteine
        pos_j: Position index of second cysteine
        residue_i: Amino acid at position i (should be C)
        residue_j: Amino acid at position j (should be C)
        distance: Distance between sulfur atoms
        existing_disulfides: Set of existing disulfide pairs

    Returns:
        Tuple of (can_form: bool, score: float)
    """
    if residue_i != "C" or residue_j != "C":
        return False, 0.0

    # Check if already part of a disulfide
    pair = tuple(sorted([pos_i, pos_j]))
    if pair in existing_disulfides:
        return True, DISULFIDE_STABILITY_SCORE

    # Ideal S-S distance for disulfide: 2.0-2.1 Angstroms
    # Cysteines in oxidized form typically have S-S distance ~2.03 Angstroms
    if DISULFIDE_DISTANCE_MIN <= distance <= DISULFIDE_DISTANCE_MAX:
        return True, DISULFIDE_STABILITY_SCORE
    elif distance < DISULFIDE_DISTANCE_MIN:
        return False, -DISULFIDE_STABILITY_SCORE * 0.5  # Too close, strain
    else:
        return False, 0.0


def validate_charge_clash(
    pos_i: int,
    pos_j: int,
    residue_i: str,
    residue_j: str,
    distance: float,
) -> tuple[bool, float]:
    """
    Detect charge clashes (same-charge repulsion).

    Args:
        pos_i: Position index of first residue
        pos_j: Position index of second residue
        residue_i: Amino acid at position i
        residue_j: Amino acid at position j
        distance: Distance between residues

    Returns:
        Tuple of (has_clash: bool, penalty: float)
    """
    if distance > CLASH_DISTANCE:
        return False, 0.0

    # Positive-positive clash
    if residue_i in CHARGED_POSITIVE and residue_j in CHARGED_POSITIVE:
        return True, CLASH_PENALTY * 0.8

    # Negative-negative clash
    if residue_i in CHARGED_NEGATIVE and residue_j in CHARGED_NEGATIVE:
        return True, CLASH_PENALTY * 0.8

    return False, 0.0


def validate_hydrophobic_core(
    residue: str,
    sasa: float,
    is_core_position: bool,
) -> tuple[bool, float]:
    """
    Validate hydrophobic core / hydrophilic surface preferences.

    Args:
        residue: Amino acid at position
        sasa: Solvent accessible surface area
        is_core_position: True if position is in protein core

    Returns:
        Tuple of (is_valid: bool, score: float)
    """
    if is_core_position:
        # Core positions should be hydrophobic
        if residue in HYDROPHOBIC:
            return True, HYDROPHOBIC_CORE_SCORE
        elif residue in HYDROPHILIC:
            return False, -HYDROPHOBIC_CORE_SCORE * 0.5
        else:
            return True, 0.0
    else:
        # Surface positions should be hydrophilic
        if residue in HYDROPHILIC:
            return True, HYDROPHILIC_SURFACE_SCORE
        elif residue in HYDROPHOBIC:
            # Allow some hydrophobic on surface but penalize
            if sasa > 100.0:
                return False, -HYDROPHOBIC_CORE_SCORE * 0.3
            return True, 0.0
        else:
            return True, 0.0


def validate_backbone_dihedral(
    residue: str,
    phi: float,
    psi: float,
) -> tuple[bool, str]:
    """
    Validate backbone dihedral angles (Ramachandran plot).

    Args:
        residue: Amino acid type
        phi: Phi angle in degrees
        psi: Psi angle in degrees

    Returns:
        Tuple of (is_allowed: bool, region: str)
    """
    # Handle glycine (widest allowed range)
    if residue == "G":
        return True, "glycine_allowed"

    # Handle proline (restricted phi)
    if residue == "P":
        if not (PROLINE_PHI_RANGE[0] <= phi <= PROLINE_PHI_RANGE[1]):
            return False, "proline_phi_violation"
        if not (PROLINE_PSI_RANGE[0] <= psi <= PROLINE_PSI_RANGE[1]):
            return False, "proline_psi_violation"
        return True, "proline_allowed"

    # General amino acids - check common regions
    # Alpha helix region: phi ≈ -60°, psi ≈ -45°
    if -180.0 <= phi <= -30.0 and -90.0 <= psi <= 60.0:
        return True, "alpha_helix"

    # Beta sheet region: phi ≈ -135°, psi ≈ 135°
    if -180.0 <= phi <= -90.0 and 90.0 <= psi <= 180.0:
        return True, "beta_sheet"

    # Left-handed helix (rare but allowed for certain residues)
    if 0.0 <= phi <= 90.0 and -90.0 <= psi <= 30.0:
        if residue in {"V", "I", "G"}:
            return True, "left_handed_helix"
        return False, "disallowed_region"

    return False, "disallowed_region"


# =============================================================================
# Batch Constraint Validation
# =============================================================================


def validate_sequence_constraints(
    sequence: str,
    residue_numbers: list[int],
    pair_distances: dict[tuple[int, int], float],
    sasa_map: dict[int, float],
    phi_psi_map: dict[int, tuple[float, float]] | None = None,
    existing_disulfides: set[tuple[int, int]] | None = None,
    core_positions: set[int] | None = None,
) -> ConstraintResult:
    """
    Validate a full sequence against all physics constraints.

    Args:
        sequence: Full amino acid sequence
        residue_numbers: List of residue numbers corresponding to sequence
        pair_distances: Dict mapping (pos_i, pos_j) to distance
        sasa_map: Dict mapping position to SASA value
        phi_psi_map: Optional dict mapping position to (phi, psi) angles
        existing_disulfides: Set of existing disulfide pairs
        core_positions: Set of positions considered core

    Returns:
        ConstraintResult with total score and violation details
    """
    violations: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {
        "hbonds": [],
        "salt_bridges": [],
        "disulfides": [],
        "charge_clashes": [],
        "hydrophobic_violations": [],
        "dihedral_violations": [],
    }
    total_score: float = 0.0

    existing_disulfides = existing_disulfides or set()
    core_positions = core_positions or set()

    seq_to_idx = {res: idx for idx, res in enumerate(residue_numbers)}

    # Check pairwise constraints
    for (pos_i, pos_j), distance in pair_distances.items():
        if pos_i not in seq_to_idx or pos_j not in seq_to_idx:
            continue

        idx_i = seq_to_idx[pos_i]
        idx_j = seq_to_idx[pos_j]
        residue_i = sequence[idx_i]
        residue_j = sequence[idx_j]

        # Check hydrogen bond formation
        can_hbond, hbond_score = validate_hbond_formation(
            pos_i, pos_j, residue_i, residue_j, distance
        )
        if can_hbond and hbond_score > 0:
            total_score += hbond_score
            details["hbonds"].append({
                "positions": (pos_i, pos_j),
                "residues": (residue_i, residue_j),
                "distance": distance,
                "score": hbond_score,
            })

        # Check salt bridge formation
        can_salt, salt_score = validate_salt_bridge(
            pos_i, pos_j, residue_i, residue_j, distance
        )
        if can_salt and salt_score > 0:
            total_score += salt_score
            details["salt_bridges"].append({
                "positions": (pos_i, pos_j),
                "residues": (residue_i, residue_j),
                "distance": distance,
                "score": salt_score,
            })

        # Check disulfide bond
        can_disulfide, disulfide_score = validate_disulfide_bond(
            pos_i, pos_j, residue_i, residue_j, distance, existing_disulfides
        )
        if can_disulfide and disulfide_score > 0:
            total_score += disulfide_score
            details["disulfides"].append({
                "positions": (pos_i, pos_j),
                "residues": (residue_i, residue_j),
                "distance": distance,
                "score": disulfide_score,
            })

        # Check charge clashes
        has_clash, clash_penalty = validate_charge_clash(
            pos_i, pos_j, residue_i, residue_j, distance
        )
        if has_clash:
            total_score += clash_penalty
            violations.append(f"charge_clash:{pos_i}-{pos_j}({residue_i},{residue_j})")
            details["charge_clashes"].append({
                "positions": (pos_i, pos_j),
                "residues": (residue_i, residue_j),
                "distance": distance,
                "penalty": clash_penalty,
            })

    # Check hydrophobic/hydrophilic preferences
    for pos in residue_numbers:
        idx = seq_to_idx[pos]
        residue = sequence[idx]
        sasa = sasa_map.get(pos, 50.0)  # Default SASA if not provided
        is_core = pos in core_positions

        is_valid, score = validate_hydrophobic_core(residue, sasa, is_core)
        if is_valid:
            total_score += score
        else:
            total_score += score
            warnings.append(f"hydrophobic_violation:{pos}({residue},sasa={sasa:.1f},core={is_core})")
            details["hydrophobic_violations"].append({
                "position": pos,
                "residue": residue,
                "sasa": sasa,
                "is_core": is_core,
                "penalty": score,
            })

    # Check backbone dihedral angles
    if phi_psi_map:
        for pos, (phi, psi) in phi_psi_map.items():
            if pos not in seq_to_idx:
                continue
            residue = sequence[seq_to_idx[pos]]
            is_allowed, region = validate_backbone_dihedral(residue, phi, psi)
            if not is_allowed:
                violations.append(f"dihedral_violation:{pos}({residue},{phi:.1f},{psi:.1f})")
                details["dihedral_violations"].append({
                    "position": pos,
                    "residue": residue,
                    "phi": phi,
                    "psi": psi,
                    "region": region,
                })

    return ConstraintResult(
        total_score=round(total_score, 3),
        violations=violations,
        warnings=warnings,
        details=details,
    )


# =============================================================================
# Beam Search Integration Helper
# =============================================================================


def compute_constraint_penalty(
    seq: str,
    position_to_index: dict[int, int],
    field_positions: list[int],
    pair_distances: dict[tuple[int, int], float],
    sasa_map: dict[int, float],
    existing_disulfides: set[tuple[int, int]] | None = None,
    core_positions: set[int] | None = None,
) -> float:
    """
    Compute constraint penalty for beam search scoring.

    This function is designed to be called during beam search
    to filter or penalize sequences that violate physics constraints.

    Args:
        seq: Current sequence (tuple of chars)
        position_to_index: Mapping from position to sequence index
        field_positions: List of design positions
        pair_distances: Distance matrix between positions
        sasa_map: SASA values for positions
        existing_disulfides: Existing disulfide pairs
        core_positions: Core positions

    Returns:
        Penalty score (negative = favorable, positive = unfavorable)
    """
    sequence = "".join(seq) if isinstance(seq, tuple) else seq
    residue_numbers = field_positions

    result = validate_sequence_constraints(
        sequence=sequence,
        residue_numbers=residue_numbers,
        pair_distances=pair_distances,
        sasa_map=sasa_map,
        existing_disulfides=existing_disulfides,
        core_positions=core_positions,
    )

    # Convert to penalty: negative score becomes bonus, positive violations become penalty
    penalty = -result.total_score  # Invert so good constraints = negative penalty

    # Add additional penalty for each violation
    penalty += len(result.violations) * 2.0
    penalty += len(result.warnings) * 0.5

    return penalty


def filter_by_constraints(
    candidates: list[Any],
    pair_distances: dict[tuple[int, int], float],
    sasa_map: dict[int, float],
    existing_disulfides: set[tuple[int, int]] | None = None,
    core_positions: set[int] | None = None,
    max_violations: int = 2,
) -> list[Any]:
    """
    Filter candidates by physics constraints.

    Args:
        candidates: List of candidate objects with 'sequence' attribute
        pair_distances: Distance matrix between positions
        sasa_map: SASA values for positions
        existing_disulfides: Existing disulfide pairs
        core_positions: Core positions
        max_violations: Maximum allowed violations per candidate

    Returns:
        Filtered list of candidates
    """
    filtered = []

    for candidate in candidates:
        if not hasattr(candidate, "sequence"):
            filtered.append(candidate)
            continue

        residue_numbers = list(range(1, len(candidate.sequence) + 1))
        result = validate_sequence_constraints(
            sequence=candidate.sequence,
            residue_numbers=residue_numbers,
            pair_distances=pair_distances,
            sasa_map=sasa_map,
            existing_disulfides=existing_disulfides,
            core_positions=core_positions,
        )

        if len(result.violations) <= max_violations:
            # Attach constraint score to candidate
            if hasattr(candidate, "constraint_score"):
                candidate.constraint_score = result.total_score
            filtered.append(candidate)

    return filtered


# =============================================================================
# Utility Functions
# =============================================================================


def get_constraint_summary(result: ConstraintResult) -> str:
    """Generate a human-readable summary of constraint validation."""
    lines = [
        f"Total Score: {result.total_score:.3f}",
        f"Violations: {len(result.violations)}",
        f"Warnings: {len(result.warnings)}",
    ]

    if result.violations:
        lines.append("  Violations:")
        for v in result.violations:
            lines.append(f"    - {v}")

    if result.warnings:
        lines.append("  Warnings:")
        for w in result.warnings:
            lines.append(f"    - {w}")

    if result.details["hbonds"]:
        lines.append(f"  H-bonds formed: {len(result.details['hbonds'])}")

    if result.details["salt_bridges"]:
        lines.append(f"  Salt bridges formed: {len(result.details['salt_bridges'])}")

    if result.details["disulfides"]:
        lines.append(f"  Disulfide bonds: {len(result.details['disulfides'])}")

    return "\n".join(lines)
