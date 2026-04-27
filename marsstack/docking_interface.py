"""Molecular docking interface for binding pocket stability evaluation.

This module provides interfaces for:
- HADDOCK/HDX integration for protein-protein/protein-ligand docking
- Binding pocket thermodynamic stability assessment
- Interface hydrophobicity analysis
- Hydrogen bond network evaluation
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# Standard amino acid hydrophobicity values (Kyte-Doolittle scale)
HYDROPHOBICITY: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


@dataclass
class HADDOCKConfig:
    """Configuration for HADDOCK docking runs."""

    server_url: str = "https://wenmr.science.uu.nl/haddock3/api"
    num_structures: int = 1000
    sampling: Literal["light", "medium", "exhaustive"] = "medium"
    rigid_bodies: int = 10
    refine_steps: int = 5
    cleanup: bool = True
    timeout: int = 3600


@dataclass
class HDXConfig:
    """Configuration for HDX-MS analysis."""

    temperature: float = 25.0  # Celsius
    ph: float = 7.0
    buffer_conditions: dict[str, float] = field(default_factory=lambda: {
        "NaCl": 150.0, "MgCl2": 2.0, "DTT": 1.0
    })
    incubation_times: list[float] = field(default_factory=lambda: [
        0.0, 10.0, 60.0, 600.0, 3600.0
    ])
    peptide_coverage_threshold: float = 0.7


@dataclass
class BindingPocketMetrics:
    """Container for binding pocket stability metrics."""

    pocket_residues: list[int] = field(default_factory=list)
    interface_area: float = 0.0
    hydrophobicity_score: float = 0.0
    hbond_count: int = 0
    hbond_satisfied_ratio: float = 0.0
    thermal_stability_tm: float = 0.0  # Tm in Celsius
    thermal_stability_dg: float = 0.0  # Delta G in kcal/mol
    hdx_protection_factor: float = 0.0
    buried_sasa: float = 0.0
    polar_ratio: float = 0.0
    compactness: float = 0.0
    conservation_score: float = 0.0
    interface_energy: float = 0.0


@dataclass
class HbondDonorAcceptor:
    """Hydrogen bond donor-acceptor pair."""

    donor_res: int
    donor_atom: str
    acceptor_res: int
    acceptor_atom: str
    distance: float
    angle: float
    energy: float = 0.0


class HADDOCKInterface:
    """Interface for HADDOCK protein-protein docking predictions."""

    def __init__(self, config: HADDOCKConfig | None = None):
        self.config = config or HADDOCKConfig()

    def prepare_input(self, receptor_pdb: str, ligand_pdb: str) -> dict[str, Any]:
        """Prepare HADDOCK input from PDB structures.

        Args:
            receptor_pdb: Path to receptor PDB file
            ligand_pdb: Path to ligand PDB file

        Returns:
            Dictionary with prepared input data
        """
        return {
            "receptor": receptor_pdb,
            "ligand": ligand_pdb,
            "num_structures": self.config.num_structures,
            "sampling": self.config.sampling,
            "rigid_bodies": self.config.rigid_bodies,
            "refine_steps": self.config.refine_steps,
        }

    def submit_job(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Submit docking job to HADDOCK server.

        Args:
            input_data: Prepared input data

        Returns:
            Job submission response with job ID
        """
        # Mock implementation for interface
        return {
            "job_id": "mock_job_id",
            "status": "submitted",
            "message": "Job submitted successfully",
        }

    def query_status(self, job_id: str) -> dict[str, Any]:
        """Query HADDOCK job status.

        Args:
            job_id: HADDOCK job identifier

        Returns:
            Status response dictionary
        """
        return {
            "job_id": job_id,
            "status": "running",
            "progress": 0.5,
        }

    def retrieve_results(self, job_id: str) -> dict[str, Any]:
        """Retrieve HADDOCK results.

        Args:
            job_id: HADDOCK job identifier

        Returns:
            Docking results with scores
        """
        return {
            "job_id": job_id,
            "status": "completed",
            "best_model": "model_1.pdb",
            "haddock_score": -50.0,
            "z_score": -2.5,
        }


class HDXInterface:
    """Interface for HDX-MS (Hydrogen-Deuterium Exchange Mass Spectrometry) analysis."""

    def __init__(self, config: HDXConfig | None = None):
        self.config = config or HDXConfig()

    def calculate_protection_factor(
        self,
        peptide_sequence: str,
        uptake_data: list[tuple[float, float]],
    ) -> float:
        """Calculate HDX protection factor for a peptide.

        Args:
            peptide_sequence: Amino acid sequence of peptide
            uptake_data: List of (time, deuterium_uptake) tuples

        Returns:
            Protection factor (higher = more protected)
        """
        if len(uptake_data) < 2:
            return 0.0

        # Calculate rate constant from uptake curve
        t_values = [t for t, _ in uptake_data]
        d_values = [d for _, d in uptake_data]

        # Simple exponential fit for deuterium uptake
        d_max = sum(1.0 for aa in peptide_sequence if aa not in ["P", "G"])
        if d_max == 0:
            return 0.0

        # Calculate initial rate
        dt = t_values[1] - t_values[0]
        dd = d_values[1] - d_values[0]
        if dt == 0:
            return 0.0

        initial_rate = dd / dt
        # Protection factor inversely related to exchange rate
        protection = d_max / max(initial_rate, 1e-6)

        return math.log10(max(protection, 1.0))

    def predict_stability_from_sequence(
        self,
        sequence: str,
        interface_residues: list[int],
    ) -> dict[str, float]:
        """Predict HDX protection from sequence and interface context.

        Args:
            sequence: Full protein sequence
            interface_residues: List of interface residue indices (0-based)

        Returns:
            Dictionary with stability predictions
        """
        interface_seq = "".join(sequence[i] for i in interface_residues if i < len(sequence))

        # Estimate protection based on residue composition
        proline_count = interface_seq.count("P")
        glycine_count = interface_seq.count("G")
        hydrophobic_count = sum(
            1 for aa in interface_seq if HYDROPHOBICITY.get(aa, 0) > 0
        )

        total = max(len(interface_seq), 1)
        proline_ratio = proline_count / total
        hydrophobic_ratio = hydrophobic_count / total

        # Higher proline content and hydrophobicity suggest better protection
        protection_score = (proline_ratio * 0.5 + hydrophobic_ratio * 0.5) * 100

        return {
            "protection_factor": protection_score,
            "proline_ratio": proline_ratio,
            "hydrophobic_ratio": hydrophobic_ratio,
            "estimated_tm_shift": hydrophobic_ratio * 5.0,  # Approximate Tm shift
        }


class BindingPocketAnalyzer:
    """Analyzer for binding pocket thermodynamic stability."""

    def __init__(self, sequence: str | None = None):
        self.sequence = sequence or ""

    def analyze_interface_hydrophobicity(
        self,
        pocket_residues: list[int],
        structure_coords: dict[int, tuple[float, float, float]] | None = None,
    ) -> float:
        """Calculate interface hydrophobicity score.

        Args:
            pocket_residues: List of pocket residue indices
            structure_coords: Optional 3D coordinates for buried SASA calculation

        Returns:
            Hydrophobicity score (-1 to 1 scale)
        """
        if not pocket_residues:
            return 0.0

        residues = [self.sequence[i] if i < len(self.sequence) else "X"
                    for i in pocket_residues]

        hydrophobicity_values = [HYDROPHOBICITY.get(r, 0) for r in residues]
        avg_hydrophobicity = sum(hydrophobicity_values) / len(hydrophobicity_values)

        # Normalize to -1 to 1 scale
        max_hydro = max(HYDROPHOBICITY.values())
        min_hydro = min(HYDROPHOBICITY.values())
        normalized = (avg_hydrophobicity - min_hydro) / (max_hydro - min_hydro) * 2 - 1

        return round(normalized, 4)

    def calculate_polar_ratio(self, pocket_residues: list[int]) -> float:
        """Calculate polar residue ratio in binding pocket.

        Args:
            pocket_residues: List of pocket residue indices

        Returns:
            Polar residue ratio (0 to 1)
        """
        if not pocket_residues:
            return 0.0

        residues = [self.sequence[i] if i < len(self.sequence) else "X"
                    for i in pocket_residues]

        polar_aa = {"S", "T", "N", "Q", "Y", "C", "H", "D", "E", "K", "R"}
        polar_count = sum(1 for r in residues if r in polar_aa)

        return round(polar_count / len(pocket_residues), 4)

    def estimate_thermal_stability(
        self,
        pocket_residues: list[int],
        interface_area: float = 0.0,
    ) -> dict[str, float]:
        """Estimate thermal stability from pocket composition.

        Args:
            pocket_residues: List of pocket residue indices
            interface_area: Interface surface area in Angstroms squared

        Returns:
            Dictionary with Tm (Celsius) and Delta G (kcal/mol) estimates
        """
        hydrophobic_score = self.analyze_interface_hydrophobicity(pocket_residues)

        # Empirical estimation based on hydrophobic effect
        # Each kcal/mol of hydrophobic free energy ~ 2-3C Tm shift
        hydrophobic_dg = hydrophobic_score * 2.5  # Estimate hydrophobic contribution

        # Interface area contribution (roughly 25 cal/A^2 for apolar surfaces)
        area_dg = interface_area * 0.025 / 1000.0  # Convert to kcal/mol

        total_dg = hydrophobic_dg + area_dg

        # Tm estimation (rough rule of thumb: 1 kcal/mol ~ 2.5C)
        estimated_tm = 25.0 + total_dg * 2.5  # Base Tm + contribution

        return {
            "tm_celsius": round(estimated_tm, 2),
            "delta_g_kcal_mol": round(total_dg, 3),
        }


class HydrogenBondNetworkAnalyzer:
    """Analyzer for hydrogen bond networks at protein interfaces."""

    def __init__(self):
        self.hbond_energy_cutoff = -2.0  # kcal/mol
        self.hbond_distance_max = 3.5  # Angstroms
        self.hbond_angle_min = 120.0  # degrees

    def detect_hbonds(
        self,
        donor_residues: list[int],
        acceptor_residues: list[int],
        coords: dict[int, dict[str, tuple[float, float, float]]],
    ) -> list[HbondDonorAcceptor]:
        """Detect hydrogen bonds between donor and acceptor residues.

        Args:
            donor_residues: Residue indices with H-bond donors
            acceptor_residues: Residue indices with H-bond acceptors
            coords: Residue coordinates with atom names as keys

        Returns:
            List of detected hydrogen bonds
        """
        hbonds = []
        donor_atoms = {"N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "OH", "OG", "OG1", "SG"}
        acceptor_atoms = {"O", "OD1", "OD2", "OE1", "OE2", "OH", "OG", "OG1", "OD", "OE", "S", "SG"}

        for d_res in donor_residues:
            if d_res not in coords:
                continue
            for d_atom, d_coord in coords[d_res].items():
                if d_atom not in donor_atoms:
                    continue

                for a_res in acceptor_residues:
                    if a_res == d_res:
                        continue
                    if a_res not in coords:
                        continue

                    for a_atom, a_coord in coords[a_res].items():
                        if a_atom not in acceptor_atoms:
                            continue

                        # Calculate donor-acceptor distance
                        dist = math.sqrt(
                            sum((dc - ac) ** 2 for dc, ac in zip(d_coord, a_coord))
                        )

                        if dist > self.hbond_distance_max:
                            continue

                        # Simplified angle calculation (would need H position)
                        angle = 180.0 - (dist / self.hbond_distance_max) * 60.0

                        # Estimate H-bond energy
                        energy = self._estimate_hbond_energy(dist, angle)

                        if energy < self.hbond_energy_cutoff:
                            hbonds.append(HbondDonorAcceptor(
                                donor_res=d_res,
                                donor_atom=d_atom,
                                acceptor_res=a_res,
                                acceptor_atom=a_atom,
                                distance=round(dist, 3),
                                angle=round(angle, 1),
                                energy=round(energy, 3),
                            ))

        return hbonds

    def _estimate_hbond_energy(self, distance: float, angle: float) -> float:
        """Estimate H-bond energy from distance and angle.

        Args:
            distance: D-A distance in Angstroms
            angle: D-H-A angle in degrees

        Returns:
            Estimated energy in kcal/mol
        """
        # Simple Lennard-Jones-like potential
        r0 = 2.9  # Optimal distance
        theta0 = 180.0  # Linear geometry

        dist_term = ((r0 / distance) ** 12 - 2 * (r0 / distance) ** 6)
        angle_term = math.cos(math.radians(theta0 - angle))

        energy = -5.0 * dist_term * angle_term  # Scale factor

        return energy

    def evaluate_network_connectivity(
        self,
        hbonds: list[HbondDonorAcceptor],
    ) -> dict[str, Any]:
        """Evaluate hydrogen bond network connectivity.

        Args:
            hbonds: List of detected hydrogen bonds

        Returns:
            Dictionary with network metrics
        """
        if not hbonds:
            return {
                "hbond_count": 0,
                "connectivity_score": 0.0,
                "network_size": 0,
                "avg_hbond_energy": 0.0,
            }

        # Build adjacency representation
        residues = set()
        for hb in hbonds:
            residues.add(hb.donor_res)
            residues.add(hb.acceptor_res)

        # Calculate metrics
        total_energy = sum(hb.energy for hb in hbonds)
        avg_energy = total_energy / len(hbonds)

        # Connectivity score based on edges vs nodes
        # For n nodes, maximum edges is n*(n-1)/2
        n_residues = len(residues)
        max_edges = n_residues * (n_residues - 1) / 2 if n_residues > 1 else 1
        connectivity = len(hbonds) / max_edges if max_edges > 0 else 0

        return {
            "hbond_count": len(hbonds),
            "network_size": n_residues,
            "connectivity_score": round(connectivity, 4),
            "avg_hbond_energy": round(avg_energy, 3),
            "total_network_energy": round(total_energy, 3),
        }

    def generate_network_summary(
        self,
        hbonds: list[HbondDonorAcceptor],
    ) -> str:
        """Generate human-readable hydrogen bond network summary.

        Args:
            hbonds: List of detected hydrogen bonds

        Returns:
            Formatted summary string
        """
        if not hbonds:
            return "No hydrogen bonds detected."

        metrics = self.evaluate_network_connectivity(hbonds)

        lines = [
            "Hydrogen Bond Network Summary",
            "=" * 30,
            f"Total H-bonds: {metrics['hbond_count']}",
            f"Network size: {metrics['network_size']} residues",
            f"Connectivity: {metrics['connectivity_score']:.2%}",
            f"Avg. energy: {metrics['avg_hbond_energy']:.2f} kcal/mol",
            "",
            "Key interactions:",
        ]

        # Sort by energy and show strongest
        sorted_hbonds = sorted(hbonds, key=lambda x: x.energy)[:5]
        for hb in sorted_hbonds:
            lines.append(
                f"  {hb.donor_res}:{hb.donor_atom} -- "
                f"{hb.acceptor_res}:{hb.acceptor_atom} "
                f"(d={hb.distance:.2f}A, E={hb.energy:.2f})"
            )

        return "\n".join(lines)


class LightweightDockingPredictor:
    """Lightweight alternatives to external docking tools.

    Uses sequence-based and simple structural features for rapid
    binding stability estimation when full docking is not available.
    """

    def __init__(self, sequence: str):
        self.sequence = sequence

    def predict_interface_residues(
        self,
        partner_sequence: str | None = None,
    ) -> list[int]:
        """Predict interface residues from sequence features.

        Args:
            partner_sequence: Optional partner sequence for comparison

        Returns:
            List of predicted interface residue indices
        """
        interface_scores: list[float] = []

        for i, aa in enumerate(self.sequence):
            score = 0.0

            # Hydrophobic patches often indicate interfaces
            if HYDROPHOBICITY.get(aa, 0) > 1.0:
                score += 0.3

            # Aromatic residues
            if aa in {"F", "Y", "W"}:
                score += 0.2

            # Charged residues at edges
            if aa in {"D", "E", "K", "R"}:
                score += 0.15

            # Conservation signal (placeholder - would need MSA)
            score += 0.1

            interface_scores.append(score)

        # Return residues above threshold
        threshold = 0.3
        return [i for i, s in enumerate(interface_scores) if s >= threshold]

    def estimate_binding_energy(
        self,
        pocket_residues: list[int],
    ) -> float:
        """Estimate binding free energy from pocket composition.

        Args:
            pocket_residues: List of pocket residue indices

        Returns:
            Estimated Delta G in kcal/mol
        """
        if not pocket_residues:
            return 0.0

        residues = [self.sequence[i] if i < len(self.sequence) else "X"
                    for i in pocket_residues]

        # Empirical scoring
        energy = 0.0

        for aa in residues:
            # Hydrophobic contributions (favorable)
            if HYDROPHOBICITY.get(aa, 0) > 2.0:
                energy -= 0.5
            # Charged interactions (context dependent)
            elif aa in {"D", "E"}:
                energy += 0.2
            elif aa in {"K", "R"}:
                energy += 0.2
            # Proline and glycine (flexibility)
            elif aa in {"P", "G"}:
                energy += 0.1

        return round(energy, 3)

    def predict_stability_score(
        self,
        pocket_residues: list[int],
    ) -> dict[str, float]:
        """Predict binding stability score.

        Args:
            pocket_residues: List of pocket residue indices

        Returns:
            Dictionary with various stability metrics
        """
        analyzer = BindingPocketAnalyzer(self.sequence)

        hydrophobicity = analyzer.analyze_interface_hydrophobicity(pocket_residues)
        polar_ratio = analyzer.calculate_polar_ratio(pocket_residues)
        binding_energy = self.estimate_binding_energy(pocket_residues)
        thermal = analyzer.estimate_thermal_stability(pocket_residues)

        # Composite stability score (0-100)
        stability_score = (
            (hydrophobicity + 1) / 2 * 30 +  # Hydrophobic contribution
            (1 - polar_ratio) * 20 +  # Less polar = more stable
            max(0, -binding_energy) * 10 +  # Favorable binding energy
            thermal["tm_celsius"] / 100 * 20  # Thermal contribution
        )

        return {
            "stability_score": round(min(100, max(0, stability_score)), 2),
            "hydrophobicity": hydrophobicity,
            "polar_ratio": polar_ratio,
            "binding_energy": binding_energy,
            "estimated_tm": thermal["tm_celsius"],
        }


def analyze_binding_pocket(
    sequence: str,
    pocket_residues: list[int],
    structure_coords: dict[int, dict[str, tuple[float, float, float]]] | None = None,
    use_lightweight: bool = True,
) -> BindingPocketMetrics:
    """Main function for comprehensive binding pocket analysis.

    Args:
        sequence: Protein sequence
        pocket_residues: List of pocket residue indices
        structure_coords: Optional 3D coordinates for H-bond analysis
        use_lightweight: Use lightweight predictors instead of external tools

    Returns:
        Complete binding pocket metrics
    """
    analyzer = BindingPocketAnalyzer(sequence)
    hbond_analyzer = HydrogenBondNetworkAnalyzer()

    # Basic metrics
    hydrophobicity = analyzer.analyze_interface_hydrophobicity(pocket_residues)
    polar_ratio = analyzer.calculate_polar_ratio(pocket_residues)
    thermal = analyzer.estimate_thermal_stability(pocket_residues)

    metrics = BindingPocketMetrics(
        pocket_residues=pocket_residues,
        hydrophobicity_score=hydrophobicity,
        polar_ratio=polar_ratio,
        thermal_stability_tm=thermal["tm_celsius"],
        thermal_stability_dg=thermal["delta_g_kcal_mol"],
    )

    # Lightweight predictions
    if use_lightweight:
        predictor = LightweightDockingPredictor(sequence)
        stability = predictor.predict_stability_score(pocket_residues)
        metrics.interface_energy = stability["binding_energy"]

    # H-bond analysis if coordinates available
    if structure_coords is not None:
        # Identify donors and acceptors
        donor_residues = [
            i for i in pocket_residues
            if i < len(sequence) and sequence[i] in {"N", "Q", "K", "R", "S", "T", "H", "W"}
        ]
        acceptor_residues = [
            i for i in pocket_residues
            if i < len(sequence) and sequence[i] in {"D", "E", "N", "Q", "S", "T", "Y", "C"}
        ]

        hbonds = hbond_analyzer.detect_hbonds(donor_residues, acceptor_residues, structure_coords)
        network_metrics = hbond_analyzer.evaluate_network_connectivity(hbonds)

        metrics.hbond_count = network_metrics["hbond_count"]
        metrics.hbond_satisfied_ratio = network_metrics["connectivity_score"]

    return metrics
