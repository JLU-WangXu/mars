from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import gemmi
import numpy as np


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

VDW = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8}

# Amino acid hydrophobicity (Kyte-Doolittle scale, normalized to -1 to 1)
AA_HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Thermal stability scores for each amino acid (higher = more thermostable)
# Based on thermophile vs mesophile frequency analysis
AA_THERMAL_STABILITY_SCORE = {
    "A": 0.8, "R": 0.3, "N": 0.2, "D": 0.2, "C": 0.9,
    "Q": 0.4, "E": 0.3, "G": 0.1, "H": 0.5, "I": 1.0,
    "L": 0.9, "K": 0.2, "M": 0.8, "F": 0.9, "P": 0.3,
    "S": 0.4, "T": 0.5, "W": 0.7, "Y": 0.6, "V": 0.9,
}

# Secondary structure propensity (PSSM-like scores)
# Helix propensity: positive = helix-forming, Sheet propensity: positive = sheet-forming
AA_HELIX_PROPENSITY = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}

AA_SHEET_PROPENSITY = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}

# Amino acid volume (Angstroms^3, approximate)
AA_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

# Charge at pH 7
AA_CHARGE = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1, "G": 0, "H": 0.1, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}

# Proline and glycine structural impact scores
PRO_GLY_HELIX_BREAKER = {
    "P": 1.0,  # Strong helix breaker
    "G": 0.7,  # Moderate helix breaker, allows tight turns
}

# Disulfide bond formation propensity (Cysteine only)
CYS_DISULFIDE_PROPENSITY = 0.85


@dataclass
class ResidueFeature:
    num: int
    name: str
    aa: str
    sasa: float
    mean_b: float
    min_dist_protected: float
    in_disulfide: bool
    glyco_motif: bool
    # New thermal stability features
    b_variance: float = 0.0
    b_percentile: float = 50.0
    thermal_score: float = 0.5
    helix_propensity: float = 1.0
    sheet_propensity: float = 1.0
    hydrophobicity: float = 0.0
    charge: float = 0.0
    is_proline: bool = False
    is_glycine: bool = False
    is_cysteine: bool = False
    secondary_structure: str = "C"
    buried_hydrophobicity: float = 0.0
    surface_hydrophobicity: float = 0.0
    local_b_density: float = 0.0
    flexibility_index: float = 0.0


@dataclass
class StructureThermalProfile:
    """Aggregated thermal stability metrics for a structure."""
    mean_thermal_score: float = 0.5
    thermal_score_std: float = 0.0
    proline_fraction: float = 0.0
    glycine_fraction: float = 0.0
    helix_content: float = 0.0
    sheet_content: float = 0.0
    mean_b_factor: float = 0.0
    b_factor_std: float = 0.0
    surface_hydrophobicity: float = 0.0
    buried_hydrophobicity: float = 0.0
    net_charge: float = 0.0
    charge_balance: float = 0.0
    disulfide_density: float = 0.0
    flexibility_score: float = 0.0
    overall_thermostability: float = 0.5
    # Amino acid composition features
    thermophilic_aa_fraction: float = 0.0
    charged_aa_fraction: float = 0.0
    aromatic_aa_fraction: float = 0.0
    aliphatic_aa_fraction: float = 0.0
    small_aa_fraction: float = 0.0


def _fibonacci_sphere(n_points: int = 64) -> np.ndarray:
    pts: list[tuple[float, float, float]] = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_points):
        y = 1 - (i / (n_points - 1)) * 2
        r = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        pts.append((math.cos(theta) * r, y, math.sin(theta) * r))
    return np.array(pts, dtype=float)


def _extract_disulfides(pdb_path: Path, chain: str) -> set[int]:
    nums: set[int] = set()
    for line in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("SSBOND"):
            continue
        c1 = line[15].strip()
        c2 = line[29].strip()
        if c1 == chain:
            nums.add(int(line[17:21].strip()))
        if c2 == chain:
            nums.add(int(line[31:35].strip()))
    return nums


def _sequence_from_chain(chain_obj: gemmi.Chain) -> list[tuple[int, str]]:
    seq = []
    for res in chain_obj:
        if res.entity_type != gemmi.EntityType.Polymer:
            continue
        if res.name not in AA3_TO_1:
            continue
        seq.append((res.seqid.num, AA3_TO_1[res.name]))
    return seq


def _assign_secondary_structure(
    chain: gemmi.Chain,
    seq_map: dict[int, str],
) -> dict[int, str]:
    """Assign secondary structure using stride or simplified phi/psi analysis."""
    try:
        structure_ss = {}
        for res in chain:
            if res.entity_type != gemmi.EntityType.Polymer:
                continue
            if res.name not in AA3_TO_1:
                continue
            seqid = res.seqid.num
            try:
                # Try to get secondary structure from DSSP/Stride
                # gemmi provides phi/psi angles
                if hasattr(res, "get_phi_psi") and res.get_phi_psi() is not None:
                    phi, psi = res.get_phi_psi()
                    structure_ss[seqid] = _phi_psi_to_secondary_structure(phi, psi)
                else:
                    structure_ss[seqid] = "C"
            except Exception:
                structure_ss[seqid] = "C"
        return structure_ss
    except Exception:
        return {num: "C" for num in seq_map.keys()}


def _phi_psi_to_secondary_structure(phi: float, psi: float) -> str:
    """Convert phi/psi angles to secondary structure assignment."""
    if phi is None or psi is None:
        return "C"
    # Helix region (alpha-helix typical)
    if -180 <= phi <= -30 and -80 <= psi <= 50:
        return "H"
    # Beta-sheet region
    if -180 <= phi <= -30 and 30 <= psi <= 180:
        return "E"
    if -30 <= phi <= 180 and 30 <= psi <= 180:
        return "E"
    # 3_10 helix
    if -180 <= phi <= -30 and -90 <= psi <= -30:
        return "G"
    # Turn
    if -180 <= phi <= 180 and -180 <= psi <= 180:
        return "T"
    return "C"


def _calculate_local_b_density(
    atoms: list[dict],
    residues: list[dict],
    res_idx: int,
    window: int = 3,
) -> float:
    """Calculate local B-factor density around a residue."""
    if not residues[res_idx]["atoms"]:
        return 0.0
    center_pos = np.mean([atoms[i]["pos"] for i in residues[res_idx]["atoms"]], axis=0)
    nearby_b = []
    for r_idx in range(max(0, res_idx - window), min(len(residues), res_idx + window + 1)):
        if not residues[r_idx]["atoms"]:
            continue
        for atom_idx in residues[r_idx]["atoms"]:
            dist = np.linalg.norm(atoms[atom_idx]["pos"] - center_pos)
            if dist < 10.0:  # Within 10 Angstrom
                nearby_b.append(atoms[atom_idx]["bf"])
    return np.mean(nearby_b) if nearby_b else 0.0


def _classify_surface_buried(sasa: float, threshold: float = 25.0) -> bool:
    """Classify residue as surface (True) or buried (False) based on SASA."""
    return sasa >= threshold


def _calculate_flexibility_index(
    b_factor: float,
    mean_b: float,
    std_b: float,
    sasa: float,
    threshold: float = 25.0,
) -> float:
    """Calculate flexibility index (0 = rigid, 1 = highly flexible)."""
    if std_b == 0:
        return 0.5
    b_contribution = min(1.0, max(0.0, (b_factor - mean_b) / (3 * std_b) + 0.5))
    sasa_contribution = 0.5 if sasa >= threshold else 0.3
    return (b_contribution * 0.7 + sasa_contribution * 0.3)


def analyze_structure(
    pdb_path: Path,
    chain_id: str,
    protected_positions: set[int],
    probe_radius: float = 1.4,
    sasa_points: int = 64,
) -> list[ResidueFeature]:
    """Analyze protein structure and extract comprehensive features."""
    st = gemmi.read_structure(str(pdb_path))
    chain = st[0][chain_id]
    seq = _sequence_from_chain(chain)
    seq_map = {num: aa for num, aa in seq}
    glyco_sites = {
        num
        for idx, (num, aa) in enumerate(seq)
        if aa == "N"
        and idx + 2 < len(seq)
        and seq[idx + 1][1] != "P"
        and seq[idx + 2][1] in {"S", "T"}
    }
    disulfides = _extract_disulfides(pdb_path, chain_id)
    sphere = _fibonacci_sphere(sasa_points)

    # Assign secondary structure
    secondary_structures = _assign_secondary_structure(chain, seq_map)

    atoms = []
    residues = []
    for res in chain:
        if res.entity_type != gemmi.EntityType.Polymer:
            continue
        if res.name not in AA3_TO_1:
            continue
        entry = {
            "num": res.seqid.num,
            "name": res.name,
            "atoms": [],
            "bf": [],
            "bf_sq": [],
        }
        residues.append(entry)
        for atom in res:
            if atom.element.name == "H":
                continue
            pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
            atoms.append(
                {
                    "res_i": len(residues) - 1,
                    "pos": pos,
                    "rad": VDW.get(atom.element.name, 1.7),
                    "bf": atom.b_iso,
                }
            )
            entry["atoms"].append(len(atoms) - 1)
            entry["bf"].append(atom.b_iso)
            entry["bf_sq"].append(atom.b_iso ** 2)

    # Pre-compute shared arrays for efficiency
    all_pos = np.array([a["pos"] for a in atoms], dtype=float)
    all_rad = np.array([a["rad"] for a in atoms], dtype=float)
    all_bf = np.array([a["bf"] for a in atoms], dtype=float)
    protected_atom_pos = np.array(
        [atoms[i]["pos"] for r in residues if r["num"] in protected_positions for i in r["atoms"]],
        dtype=float,
    )

    # Pre-compute B-factor statistics per residue
    res_b_means = np.array([np.mean(r["bf"]) if r["bf"] else 0.0 for r in residues], dtype=float)
    res_b_vars = np.array([
        np.var(r["bf"]) if len(r["bf"]) > 1 else 0.0 for r in residues
    ], dtype=float)

    # Global B statistics for percentile calculation
    global_b_mean = float(np.mean(all_bf))
    global_b_std = float(np.std(all_bf))

    per_res_sasa = [0.0] * len(residues)

    # Optimized SASA calculation with vectorized neighbor search
    for i, atom in enumerate(atoms):
        probe = atom["rad"] + probe_radius
        cutoff = probe + max(all_rad) + probe_radius + 0.01
        d2 = np.sum((all_pos - atom["pos"]) ** 2, axis=1)
        # Use squared distance for faster comparison
        cutoff2 = cutoff ** 2
        neigh = np.where(d2 < cutoff2)[0]
        neigh = neigh[neigh != i]

        exposed = 0
        for unit in sphere:
            sp = atom["pos"] + unit * probe
            occluded = False
            # Vectorized occlusion check
            dist2_to_neigh = np.sum((all_pos[neigh] - sp) ** 2, axis=1)
            probe_r2 = (probe_radius + all_rad[neigh]) ** 2
            if np.any(dist2_to_neigh < probe_r2):
                occluded = True
                break
            if not occluded:
                exposed += 1
        area = 4 * math.pi * (probe**2) * (exposed / sasa_points)
        per_res_sasa[atom["res_i"]] += area

    # Build result with all thermal stability features
    result: list[ResidueFeature] = []
    for i, res in enumerate(residues):
        min_dist = 999.0
        if len(protected_atom_pos) > 0 and residues[i]["atoms"]:
            atom_pos = np.array([atoms[atm_idx]["pos"] for atm_idx in residues[i]["atoms"]], dtype=float)
            dists = np.sqrt(np.sum((protected_atom_pos[:, np.newaxis] - atom_pos) ** 2, axis=2))
            min_dist = float(np.min(dists))

        aa = AA3_TO_1[res["name"]]
        mean_b = res_b_means[i]
        b_variance = res_b_vars[i]
        b_percentile = 50.0
        if global_b_std > 0:
            z_score = (mean_b - global_b_mean) / global_b_std
            b_percentile = float(np.clip(50 + 50 * z_score / 3, 0, 100))

        # Thermal stability features
        thermal_score = AA_THERMAL_STABILITY_SCORE.get(aa, 0.5)
        helix_propensity = AA_HELIX_PROPENSITY.get(aa, 1.0)
        sheet_propensity = AA_SHEET_PROPENSITY.get(aa, 1.0)
        hydrophobicity = AA_HYDROPHOBICITY.get(aa, 0.0)
        charge = AA_CHARGE.get(aa, 0.0)
        is_proline = aa == "P"
        is_glycine = aa == "G"
        is_cysteine = aa == "C"

        secondary_structure = secondary_structures.get(res["num"], "C")

        # Calculate local B-factor density
        local_b_density = _calculate_local_b_density(atoms, residues, i)

        # Flexibility index
        flexibility_index = _calculate_flexibility_index(
            mean_b, global_b_mean, global_b_std, per_res_sasa[i]
        )

        result.append(
            ResidueFeature(
                num=res["num"],
                name=res["name"],
                aa=aa,
                sasa=round(per_res_sasa[i], 3),
                mean_b=round(mean_b, 3),
                min_dist_protected=round(min_dist, 3),
                in_disulfide=res["num"] in disulfides,
                glyco_motif=res["num"] in glyco_sites,
                b_variance=round(b_variance, 3),
                b_percentile=round(b_percentile, 2),
                thermal_score=round(thermal_score, 3),
                helix_propensity=round(helix_propensity, 3),
                sheet_propensity=round(sheet_propensity, 3),
                hydrophobicity=round(hydrophobicity, 2),
                charge=round(charge, 2),
                is_proline=is_proline,
                is_glycine=is_glycine,
                is_cysteine=is_cysteine,
                secondary_structure=secondary_structure,
                local_b_density=round(local_b_density, 3),
                flexibility_index=round(flexibility_index, 3),
            )
        )

    return result


def compute_thermal_profile(
    features: list[ResidueFeature],
) -> StructureThermalProfile:
    """Compute aggregated thermal stability profile for a structure."""
    if not features:
        return StructureThermalProfile()

    n = len(features)

    # Basic composition
    aa_counts = {}
    for f in features:
        aa_counts[f.aa] = aa_counts.get(f.aa, 0) + 1

    # Thermostable amino acid fraction (I, L, V, M, F, W, C, A)
    thermophilic_aa = {"I", "L", "V", "M", "F", "W", "C", "A"}
    thermophilic_count = sum(aa_counts.get(aa, 0) for aa in thermophilic_aa)

    # Charged residues
    charged_aa = {"R", "K", "D", "E", "H"}
    charged_count = sum(aa_counts.get(aa, 0) for aa in charged_aa)

    # Aromatic residues
    aromatic_aa = {"F", "W", "Y", "H"}
    aromatic_count = sum(aa_counts.get(aa, 0) for aa in aromatic_aa)

    # Aliphatic residues
    aliphatic_aa = {"A", "V", "I", "L", "M"}
    aliphatic_count = sum(aa_counts.get(aa, 0) for aa in aliphatic_aa)

    # Small residues
    small_aa = {"G", "A", "S", "T", "C"}
    small_count = sum(aa_counts.get(aa, 0) for aa in small_aa)

    # Secondary structure content
    helix_count = sum(1 for f in features if f.secondary_structure == "H")
    sheet_count = sum(1 for f in features if f.secondary_structure == "E")

    # Thermal stability scores
    thermal_scores = [f.thermal_score for f in features]
    mean_thermal = np.mean(thermal_scores)
    std_thermal = np.std(thermal_scores)

    # B-factors
    b_factors = [f.mean_b for f in features]
    mean_b = np.mean(b_factors)
    std_b = np.std(b_factors)

    # Surface vs buried hydrophobicity
    surface_hydro = [f.hydrophobicity for f in features if f.sasa >= 25.0]
    buried_hydro = [f.hydrophobicity for f in features if f.sasa < 25.0]
    mean_surface_hydro = np.mean(surface_hydro) if surface_hydro else 0.0
    mean_buried_hydro = np.mean(buried_hydro) if buried_hydro else 0.0

    # Charge distribution
    charges = [f.charge for f in features]
    net_charge = sum(charges)
    positive = sum(1 for c in charges if c > 0)
    negative = sum(1 for c in charges if c < 0)
    charge_balance = abs(positive - negative) / n if n > 0 else 0.0

    # Disulfide density
    disulfide_count = sum(1 for f in features if f.in_disulfide)
    disulfide_density = disulfide_count / n if n > 0 else 0.0

    # Flexibility score
    flexibilities = [f.flexibility_index for f in features]
    mean_flexibility = np.mean(flexibilities)

    # Overall thermostability (weighted composite score)
    overall_thermostability = _calculate_overall_thermostability(
        mean_thermal=mean_thermal,
        std_thermal=std_thermal,
        helix_content=helix_count / n,
        sheet_content=sheet_count / n,
        thermophilic_fraction=thermophilic_count / n,
        disulfide_density=disulfide_density,
        net_charge_ratio=min(abs(net_charge) / n, 1.0),
        surface_hydrophobicity=mean_surface_hydro,
        buried_hydrophobicity=mean_buried_hydro,
        flexibility=mean_flexibility,
    )

    return StructureThermalProfile(
        mean_thermal_score=round(mean_thermal, 4),
        thermal_score_std=round(std_thermal, 4),
        proline_fraction=round(aa_counts.get("P", 0) / n, 4),
        glycine_fraction=round(aa_counts.get("G", 0) / n, 4),
        helix_content=round(helix_count / n, 4),
        sheet_content=round(sheet_count / n, 4),
        mean_b_factor=round(mean_b, 3),
        b_factor_std=round(std_b, 3),
        surface_hydrophobicity=round(mean_surface_hydro, 3),
        buried_hydrophobicity=round(mean_buried_hydro, 3),
        net_charge=round(net_charge, 2),
        charge_balance=round(charge_balance, 4),
        disulfide_density=round(disulfide_density, 4),
        flexibility_score=round(mean_flexibility, 4),
        overall_thermostability=round(overall_thermostability, 4),
        thermophilic_aa_fraction=round(thermophilic_count / n, 4),
        charged_aa_fraction=round(charged_count / n, 4),
        aromatic_aa_fraction=round(aromatic_count / n, 4),
        aliphatic_aa_fraction=round(aliphatic_count / n, 4),
        small_aa_fraction=round(small_count / n, 4),
    )


def _calculate_overall_thermostability(
    mean_thermal: float,
    std_thermal: float,
    helix_content: float,
    sheet_content: float,
    thermophilic_fraction: float,
    disulfide_density: float,
    net_charge_ratio: float,
    surface_hydrophobicity: float,
    buried_hydrophobicity: float,
    flexibility: float,
) -> float:
    """
    Calculate overall thermostability score (0-1 scale).
    Higher = more thermostable.
    """
    # Thermal amino acid composition (weight: 0.3)
    composition_score = mean_thermal

    # Secondary structure contribution - moderate helix/sheet is good (weight: 0.2)
    # Too much helix or sheet can be destabilizing
    ideal_ss = 0.45  # ~45% regular secondary structure is typical
    ss_deviation = abs((helix_content + sheet_content) - ideal_ss)
    ss_score = max(0, 1 - ss_deviation * 3)

    # Thermophilic amino acid fraction (weight: 0.15)
    thermophilic_score = thermophilic_fraction

    # Disulfide bonds contribute to stability (weight: 0.1)
    # But too many disulfides can strain the structure
    disulfide_score = min(1.0, disulfide_density * 10)

    # Charge balance - some charged residues are important, but too many destabilize (weight: 0.1)
    charge_score = 1 - net_charge_ratio

    # Hydrophobic core vs surface (weight: 0.1)
    # Good thermophiles have higher buried hydrophobicity
    hydro_score = 0.5
    if buried_hydrophobicity > 0:
        hydro_score = min(1.0, (buried_hydrophobicity + 1) / 4)

    # Flexibility - lower flexibility = more stable (weight: 0.05)
    flexibility_score = 1 - flexibility

    # Combine scores with weights
    overall = (
        composition_score * 0.30 +
        ss_score * 0.20 +
        thermophilic_score * 0.15 +
        disulfide_score * 0.10 +
        charge_score * 0.10 +
        hydro_score * 0.10 +
        flexibility_score * 0.05
    )

    return max(0.0, min(1.0, overall))


def detect_oxidation_hotspots(
    features: list[ResidueFeature],
    min_sasa: float,
    min_dist_protected: float,
) -> list[int]:
    hotspots = []
    for feat in features:
        if feat.min_dist_protected < min_dist_protected:
            continue
        if feat.aa == "C" and feat.in_disulfide:
            continue
        if feat.aa in {"M", "W", "Y", "H", "C"} and feat.sasa >= min_sasa:
            hotspots.append(feat.num)
    return hotspots


def detect_flexible_surface_positions(
    features: list[ResidueFeature],
    min_sasa: float,
    b_percentile: float = 85.0,
) -> list[int]:
    bvals = np.array([f.mean_b for f in features], dtype=float)
    thresh = float(np.percentile(bvals, b_percentile))
    return [f.num for f in features if f.sasa >= min_sasa and f.mean_b >= thresh]


def detect_thermostability_hotspots(
    features: list[ResidueFeature],
    thermal_threshold: float = 0.7,
    min_sasa: float = 40.0,
) -> dict[str, list[int]]:
    """
    Detect regions of potential thermostability concern.
    Returns dict with categories of hotspots.
    """
    hotspots = {
        "low_thermal_score": [],
        "high_flexibility": [],
        "surface_polar": [],
        "low_disulfide_regions": [],
    }

    n = len(features)
    if n == 0:
        return hotspots

    # Identify low thermal score residues on surface
    for feat in features:
        if feat.thermal_score < thermal_threshold and feat.sasa >= min_sasa:
            hotspots["low_thermal_score"].append(feat.num)
        if feat.flexibility_index > 0.7 and feat.sasa >= min_sasa:
            hotspots["high_flexibility"].append(feat.num)

    # Find consecutive regions
    if n > 5:
        window = 5
        for i in range(n - window + 1):
            window_features = features[i:i + window]
            window_thermal = np.mean([f.thermal_score for f in window_features])
            window_disulfide = sum(1 for f in window_features if f.in_disulfide)

            # Region with low average thermal score and no disulfides
            if window_thermal < 0.5 and window_disulfide == 0:
                start_num = features[i].num
                end_num = features[i + window - 1].num
                hotspots["low_disulfide_regions"].extend([start_num, end_num])

    # Surface polar/charged regions (could be destabilizing)
    for feat in features:
        if feat.sasa >= min_sasa and feat.aa in {"D", "E", "R", "K", "N", "Q"}:
            if feat.thermal_score < 0.4:
                hotspots["surface_polar"].append(feat.num)

    return hotspots


def calculate_mutation_potential(
    features: list[ResidueFeature],
    target_positions: list[int],
) -> list[dict]:
    """
    Calculate potential mutations for improving thermostability.
    Returns scoring for each position indicating mutation suitability.
    """
    recommendations = []

    for feat in features:
        if feat.num not in target_positions:
            continue

        mutations = []

        # Current residue info
        current_thermal = feat.thermal_score

        # Propose stabilizing mutations based on context
        aa = feat.aa

        if aa in {"D", "E"} and feat.sasa < 25.0:
            # Buried acidic -> consider neutralization
            mutations.append({
                "to": "Q",  # Glutamine is more thermostable
                "reason": "buried_acid_to_amide",
                "score_delta": 0.15,
            })

        if aa in {"K", "R"} and feat.sasa >= 40.0:
            # Surface charged -> could introduce proline for rigidity
            mutations.append({
                "to": "P",
                "reason": "surface_charge_to_proline",
                "score_delta": -0.05,
            })

        if aa == "M" and feat.sasa >= 30.0:
            # Oxidation-susceptible methionine
            mutations.append({
                "to": "L",  # Leucine is more oxidatively stable
                "reason": "met_oxidation_risk",
                "score_delta": 0.10,
            })

        if aa == "C" and not feat.in_disulfide and feat.sasa < 20.0:
            # Buried cysteine -> potential for new disulfide
            mutations.append({
                "to": "C",
                "reason": "potential_disulfide_pair",
                "score_delta": 0.20,
            })

        if aa in {"S", "T"} and feat.thermal_score < 0.5:
            # Low-thermal serine/threonine -> consider valine/alanine
            mutations.append({
                "to": "V" if aa == "T" else "A",
                "reason": "polar_to_hydrophobic",
                "score_delta": 0.25,
            })

        if aa == "G":
            # Glycine -> consider alanine for rigidity
            mutations.append({
                "to": "A",
                "reason": "glycine_to_alanine_rigidity",
                "score_delta": 0.30,
            })

        recommendations.append({
            "position": feat.num,
            "current_aa": aa,
            "current_thermal": current_thermal,
            "sasa": feat.sasa,
            "secondary_structure": feat.secondary_structure,
            "suggested_mutations": mutations,
        })

    return recommendations
