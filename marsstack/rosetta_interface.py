"""Rosetta energy scoring interface for MARS algorithm.

Provides integration with Rosetta protein structure modeling suite for:
- FastDesign-based sequence design optimization
- Score12/Talaris2014 scoring functions
- Rapid mutant evaluation
- Result parsing and analysis

Requirements:
    - Rosetta 3.12+ installed locally, OR
    - Docker/Singularity container with Rosetta

Installation options:
    1. Local: Download from https://www.rosettacommons.org/software
    2. Docker: Use rosettacommons/rosetta image
    3. Singularity: Build from Rosetta Docker image

Environment variables:
    ROSETTA_DB: Path to Rosetta database (default: ~/Rosetta/database)
    ROSETTA_BIN: Path to Rosetta binaries directory
    ROSETTA_MODE: "local", "docker", or "singularity"
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# =============================================================================
# Configuration and Detection
# =============================================================================

# Rosetta executable paths (common locations)
ROSETTA_BIN_PATHS = [
    Path.home() / "Rosetta" / "main" / "source" / "bin",
    Path.home() / "rosetta" / "bin",
    Path("/opt/Rosetta/bin"),
    Path("/usr/local/Rosetta/bin"),
]

# Docker/Singularity image
ROSETTA_IMAGE = "rosettacommons/rosetta:latest"


def find_rosetta_installation() -> dict[str, Path | None]:
    """Detect local Rosetta installation.

    Returns:
        Dict with 'bin_path', 'database', and 'available' keys.
    """
    result = {
        "bin_path": None,
        "database": None,
        "available": False,
    }

    # Check environment variables first
    if bin_path_env := Path(os.environ.get("ROSETTA_BIN", "")):
        if bin_path_env.exists():
            result["bin_path"] = bin_path_env
            result["available"] = True

    if db_env := Path(os.environ.get("ROSETTA_DB", "")):
        if db_env.exists():
            result["database"] = db_env

    # Search common paths
    for search_path in ROSETTA_BIN_PATHS:
        if search_path.exists():
            result["bin_path"] = search_path
            result["available"] = True

            # Try to find database
            db_search = [
                search_path.parent / "database",
                search_path.parent.parent / "database",
                search_path.parent.parent / "main" / "database",
            ]
            for db_path in db_search:
                if db_path.exists():
                    result["database"] = db_path
                    break
            break

    return result


def check_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_singularity_available() -> bool:
    """Check if Singularity is available."""
    try:
        subprocess.run(
            ["singularity", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RosettaScoreResult:
    """Parsed Rosetta score output."""

    total_score: float
    fa_atr: float | None = None  # Attractive Lennard-Jones
    fa_rep: float | None = None  # Repulsive Lennard-Jones
    fa_sol: float | None = None  # Solvation
    fa_intra_sol_xover4: float | None = None  # Intra-residue solvation
    fa_elec: float | None = None  # Electrostatics
    pro_close: float | None = None  # Proline ring closure
    hbond_sr_bb: float | None = None  # HBond short-range backbone
    hbond_lr_bb: float | None = None  # HBond long-range backbone
    hbond_bb_sc: float | None = None  # HBond backbone-sidechain
    hbond_sc: float | None = None  # HBond sidechain-sidechain
    rama_prepro: float | None = None  # Ramachandran (pre-proline)
    p_aa_pp: float | None = None  # Probability amino acid
    ref: float | None = None  # Reference energy
    score12: float | None = None  # Combined Score12
    talaris: float | None = None  # Talaris2014 score
    dslf_fa13: float | None = None  # Disulfide fa13
    atoms: int = 0
    description: str = ""

    @property
    def stability_score(self) -> float:
        """Combined stability metric (lower = more stable)."""
        return self.total_score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_score": self.total_score,
            "fa_atr": self.fa_atr,
            "fa_rep": self.fa_rep,
            "fa_sol": self.fa_sol,
            "fa_elec": self.fa_elec,
            "hbond_sr_bb": self.hbond_sr_bb,
            "hbond_lr_bb": self.hbond_lr_bb,
            "rama_prepro": self.rama_prepro,
            "p_aa_pp": self.p_aa_pp,
            "ref": self.ref,
            "atoms": self.atoms,
            "description": self.description,
            "stability_score": self.stability_score,
        }


@dataclass
class MutantEvaluationResult:
    """Result of single-point mutant evaluation."""

    wild_type: str
    position: int
    mutant: str
    wild_type_score: float
    mutant_score: float
    ddg: float  # Delta delta G (mutant - wild_type)
    relative_stability: float  # ddg in context of wild_type

    @property
    def is_stabilizing(self) -> bool:
        """True if mutation is stabilizing (negative ddg)."""
        return self.ddg < 0

    @property
    def is_neutral(self) -> bool:
        """True if mutation is roughly neutral (|ddg| < 1 REU)."""
        return abs(self.ddg) < 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "wild_type": self.wild_type,
            "position": self.position,
            "mutant": self.mutant,
            "wild_type_score": self.wild_type_score,
            "mutant_score": self.mutant_score,
            "ddg": self.ddg,
            "relative_stability": self.relative_stability,
            "is_stabilizing": self.is_stabilizing,
            "is_neutral": self.is_neutral,
        }


@dataclass
class FastDesignResult:
    """Result of Rosetta FastDesign run."""

    final_sequence: str | None = None
    final_score: float | None = None
    scores: list[float] = field(default_factory=list)
    rmsd_trajectory: list[float] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0
    runtime_seconds: float = 0.0
    output_pdb: Path | None = None


# =============================================================================
# Score File Parsing
# =============================================================================


def parse_score_file(score_file: Path) -> list[RosettaScoreResult]:
    """Parse Rosetta score file (.sc).

    Args:
        score_file: Path to Rosetta score output file.

    Returns:
        List of ScoreResult objects for each structure.
    """
    results: list[RosettaScoreResult] = []

    if not score_file.exists():
        return results

    with open(score_file, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return results

    # Parse header to find column indices
    header = lines[0].split()
    col_indices: dict[str, int] = {}
    for idx, col in enumerate(header):
        col_indices[col.lower()] = idx

    # Parse data lines
    for line in lines[1:]:
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) < len(header):
            continue

        try:
            result = RosettaScoreResult(
                total_score=float(parts[col_indices.get("total_score", 0)]),
            )

            # Extract optional score terms
            if "fa_atr" in col_indices:
                result.fa_atr = _safe_float(parts[col_indices["fa_atr"]])
            if "fa_rep" in col_indices:
                result.fa_rep = _safe_float(parts[col_indices["fa_rep"]])
            if "fa_sol" in col_indices:
                result.fa_sol = _safe_float(parts[col_indices["fa_sol"]])
            if "fa_elec" in col_indices:
                result.fa_elec = _safe_float(parts[col_indices["fa_elec"]])
            if "hbond_sr_bb" in col_indices:
                result.hbond_sr_bb = _safe_float(parts[col_indices["hbond_sr_bb"]])
            if "hbond_lr_bb" in col_indices:
                result.hbond_lr_bb = _safe_float(parts[col_indices["hbond_lr_bb"]])
            if "rama_prepro" in col_indices:
                result.rama_prepro = _safe_float(parts[col_indices["rama_prepro"]])
            if "p_aa_pp" in col_indices:
                result.p_aa_pp = _safe_float(parts[col_indices["p_aa_pp"]])
            if "ref" in col_indices:
                result.ref = _safe_float(parts[col_indices["ref"]])
            if "score12" in col_indices:
                result.score12 = _safe_float(parts[col_indices["score12"]])
            if "talaris" in col_indices:
                result.talaris = _safe_float(parts[col_indices["talaris"]])
            if "atoms" in col_indices:
                result.atoms = int(float(parts[col_indices["atoms"]]))
            if "description" in col_indices:
                result.description = parts[col_indices["description"]]

            results.append(result)
        except (ValueError, IndexError):
            continue

    return results


def parse_pdb_sequence(pdb_path: Path, chain_id: str = "A") -> str:
    """Extract sequence from PDB file.

    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain identifier.

    Returns:
        One-letter amino acid sequence.
    """
    AA_MAP = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }

    sequence = []
    seen_residues = set()

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == chain_id:
                res_name = line[17:20].strip()
                res_num = line[22:26].strip()
                atom_name = line[12:16].strip()

                if atom_name == "CA" and res_num not in seen_residues:
                    seen_residues.add(res_num)
                    sequence.append(AA_MAP.get(res_name, "?"))

    return "".join(sequence)


def _safe_float(value: str) -> float | None:
    """Safely parse float value."""
    try:
        return float(value)
    except ValueError:
        return None


# =============================================================================
# Rosetta Executable Wrapper
# =============================================================================


@dataclass
class RosettaConfig:
    """Configuration for Rosetta execution."""

    bin_path: Path | None = None
    database: Path | None = None
    mode: str = "local"  # "local", "docker", "singularity"
    image: str = ROSETTA_IMAGE
    extra_flags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode == "local" and self.bin_path is None:
            install = find_rosetta_installation()
            if install["available"]:
                self.bin_path = install["bin_path"]
                self.database = install["database"]


class RosettaExecutor:
    """Execute Rosetta applications."""

    def __init__(self, config: RosettaConfig | None = None):
        self.config = config or RosettaConfig()

    def _build_command(
        self,
        executable: str,
        input_pdb: Path,
        extra_flags: dict[str, Any] | None = None,
    ) -> list[str]:
        """Build Rosetta command line."""
        flags = {
            "-s": str(input_pdb.absolute()),
        }

        if self.config.database:
            flags["-database"] = str(self.config.database.absolute())

        flags.update(self.config.extra_flags)
        if extra_flags:
            flags.update(extra_flags)

        cmd = []
        if self.config.mode == "docker":
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{input_pdb.parent}:/work",
                "-w", "/work",
                self.config.image,
                executable,
            ]
        elif self.config.mode == "singularity":
            cmd = [
                "singularity", "exec",
                f"docker://{self.config.image}",
                executable,
            ]
        else:
            if self.config.bin_path:
                cmd.append(str(self.config.bin_path / executable))
            else:
                cmd.append(executable)

        for key, value in flags.items():
            cmd.extend([key, str(value)])

        return cmd

    def run(
        self,
        executable: str,
        input_pdb: Path,
        output_dir: Path | None = None,
        extra_flags: dict[str, Any] | None = None,
        timeout: int = 3600,
    ) -> tuple[subprocess.CompletedProcess, Path | None]:
        """Run Rosetta application.

        Args:
            executable: Rosetta executable name (e.g., 'score_jd2.default.linuxgccrelease').
            input_pdb: Input PDB file.
            output_dir: Output directory (creates temp if None).
            extra_flags: Additional Rosetta flags.
            timeout: Timeout in seconds.

        Returns:
            Tuple of (CompletedProcess, output_score_file or None).
        """
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(executable, input_pdb, extra_flags)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(output_dir),
            )

            # Find score file
            score_file = None
            for pattern in ["*.sc", "score*.sc"]:
                matches = list(output_dir.glob(pattern))
                if matches:
                    score_file = matches[0]
                    break

            return result, score_file

        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"Rosetta execution timed out after {timeout}s") from e
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Rosetta executable not found: {executable}. "
                f"Check installation or set ROSETTA_BIN environment variable."
            ) from e


# =============================================================================
# High-Level Interface
# =============================================================================


class RosettaScorer:
    """High-level interface for Rosetta scoring operations."""

    def __init__(self, config: RosettaConfig | None = None):
        self.executor = RosettaExecutor(config)
        self.config = self.executor.config

    def score_pdb(
        self,
        pdb_path: Path,
        score_function: str = "talaris2014",
    ) -> RosettaScoreResult | None:
        """Score a PDB structure.

        Args:
            pdb_path: Path to PDB file.
            score_function: Scoring function ('score12', 'talaris2014', 'ref2015').

        Returns:
            ScoreResult or None if scoring failed.
        """
        # Map score function names to Rosetta executables
        exe_map = {
            "score12": "score_jd2.default.linuxgccrelease",
            "talaris2014": "score_jd2.default.linuxgccrelease",
            "ref2015": "score_jd2.default.linuxgccrelease",
        }

        executable = exe_map.get(score_function, "score_jd2.default.linuxgccrelease")

        flags = {
            "-in:file:l": str(pdb_path.absolute()),
            "-out:file:score_only": "",  # Output score file only
        }

        if score_function == "score12":
            flags["-score:weights"] = "score12"
        elif score_function == "talaris2014":
            flags["-score:weights"] = "talaris2014"
        elif score_function == "ref2015":
            flags["-score:weights"] = "ref2015"

        result, score_file = self.executor.run(executable, pdb_path, extra_flags=flags)

        if score_file and score_file.exists():
            scores = parse_score_file(score_file)
            return scores[0] if scores else None

        return None

    def score_sequences(
        self,
        pdb_path: Path,
        sequences: list[str],
        score_function: str = "talaris2014",
    ) -> list[RosettaScoreResult]:
        """Score multiple sequences using the same structure.

        Args:
            pdb_path: Reference PDB structure.
            sequences: List of sequences to score.
            score_function: Scoring function name.

        Returns:
            List of ScoreResult for each sequence.
        """
        results: list[RosettaScoreResult] = []
        temp_dir = Path(tempfile.mkdtemp())

        try:
            for i, seq in enumerate(sequences):
                # Create silent file entry for sequence
                # This is a simplified version - full implementation would use
                # Rosetta's silent file format for batch scoring
                result = self._score_single_sequence(
                    pdb_path, seq, score_function, temp_dir, f"seq_{i}"
                )
                if result:
                    results.append(result)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return results

    def _score_single_sequence(
        self,
        pdb_path: Path,
        sequence: str,
        score_function: str,
        temp_dir: Path,
        tag: str,
    ) -> RosettaScoreResult | None:
        """Score a single sequence (placeholder for full implementation)."""
        # Full implementation would:
        # 1. Use FastDesign to pack the structure with new sequence
        # 2. Extract final coordinates
        # 3. Score the result
        # This is a simplified interface
        return None

    def fastdesign(
        self,
        pdb_path: Path,
        target_sequence: str | None = None,
        nstruct: int = 1,
        rentype: str = "polynomial",
        iterations: int = 3,
    ) -> FastDesignResult:
        """Run Rosetta FastDesign for sequence optimization.

        Args:
            pdb_path: Input PDB structure.
            target_sequence: Optional target sequence (None = redesign all).
            nstruct: Number of independent design trajectories.
            rentype: Rotamer explosion type.
            iterations: Design iterations.

        Returns:
            FastDesignResult with final sequence and scores.
        """
        import time

        start_time = time.time()
        output_dir = Path(tempfile.mkdtemp())

        flags = {
            "-s": str(pdb_path.absolute()),
            "-out:path:all": str(output_dir),
            "-out:file:silent": str(output_dir / "design.out"),
            "-nstruct": str(nstruct),
            "-ex1": "",
            "-ex2": "",
            "-ex3": "",
            "-use_input_sc": "",
            "-flip_HNQ": "",
            "-no_his": "",
        }

        if target_sequence:
            # Set sequence design constraints
            flags["-sequence_file"] = str(target_sequence)

        result, _ = self.executor.run(
            "fastdesign_jd2.default.linuxgccrelease",
            pdb_path,
            output_dir=output_dir,
            extra_flags=flags,
            timeout=7200,  # 2 hour timeout
        )

        runtime = time.time() - start_time

        # Parse results
        fastdesign_result = FastDesignResult(runtime_seconds=runtime)

        silent_file = output_dir / "design.out"
        if silent_file.exists():
            with open(silent_file, "r") as f:
                content = f.read()

            # Extract scores
            score_matches = re.findall(r"total_score\s+([-\d.]+)", content)
            fastdesign_result.scores = [float(s) for s in score_matches]
            if fastdesign_result.scores:
                fastdesign_result.final_score = fastdesign_result.scores[0]

            # Extract final sequence
            seq_match = re.search(r"sequence\s+(\S+)", content)
            if seq_match:
                fastdesign_result.final_sequence = seq_match.group(1)

            fastdesign_result.converged = len(fastdesign_result.scores) > 0

        # Cleanup
        shutil.rmtree(output_dir, ignore_errors=True)

        return fastdesign_result

    def evaluate_mutant(
        self,
        pdb_path: Path,
        position: int,
        mutant: str,
        wild_type_seq: str | None = None,
        score_function: str = "talaris2014",
    ) -> MutantEvaluationResult | None:
        """Evaluate a single-point mutant.

        Args:
            pdb_path: Wild-type PDB structure.
            position: Position to mutate (1-indexed).
            mutant: Mutant amino acid (one-letter code).
            wild_type_seq: Wild-type sequence (auto-detected if None).
            score_function: Scoring function name.

        Returns:
            MutantEvaluationResult with ddg calculation.
        """
        # Detect wild-type amino acid
        if wild_type_seq is None:
            wild_type_seq = parse_pdb_sequence(pdb_path)

        if position > len(wild_type_seq) or position < 1:
            raise ValueError(f"Position {position} out of range for sequence of length {len(wild_type_seq)}")

        wild_type_aa = wild_type_seq[position - 1]

        # Score wild-type
        wt_result = self.score_pdb(pdb_path, score_function)
        if wt_result is None:
            return None

        # Create mutant structure (simplified - uses PointMutantMover in full impl)
        mutant_pdb = self._create_mutant_pdb(pdb_path, position, mutant)
        if mutant_pdb is None:
            return None

        try:
            # Score mutant
            mut_result = self.score_pdb(mutant_pdb, score_function)
            if mut_result is None:
                return None

            ddg = mut_result.total_score - wt_result.total_score

            return MutantEvaluationResult(
                wild_type=wild_type_aa,
                position=position,
                mutant=mutant,
                wild_type_score=wt_result.total_score,
                mutant_score=mut_result.total_score,
                ddg=ddg,
                relative_stability=ddg / abs(wt_result.total_score) if wt_result.total_score != 0 else 0,
            )
        finally:
            if mutant_pdb.exists():
                mutant_pdb.unlink()

    def _create_mutant_pdb(
        self,
        pdb_path: Path,
        position: int,
        mutant: str,
    ) -> Path | None:
        """Create mutant PDB file by modifying residue at position.

        This is a simplified implementation. Full version would use
        Rosetta's PointMutantMover or CartMutateMover.

        Returns:
            Path to temporary mutant PDB file.
        """
        AA_3TO1 = {
            "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
            "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
            "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
            "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
        }

        if mutant not in AA_3TO1:
            return None

        mutant_resname = AA_3TO1[mutant]
        temp_pdb = Path(tempfile.mktemp(suffix=".pdb"))

        try:
            with open(pdb_path, "r") as f_in, open(temp_pdb, "w") as f_out:
                for line in f_in:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        res_num = int(line[22:26].strip())
                        if res_num == position:
                            # Replace residue name
                            line = line[:17] + mutant_resname.ljust(3) + line[20:]
                    f_out.write(line)

            return temp_pdb
        except Exception:
            if temp_pdb.exists():
                temp_pdb.unlink()
            return None

    def batch_evaluate_mutants(
        self,
        pdb_path: Path,
        mutants: list[tuple[int, str]],  # List of (position, mutant_aa)
        wild_type_seq: str | None = None,
        parallel: bool = True,
    ) -> list[MutantEvaluationResult]:
        """Batch evaluate multiple mutants.

        Args:
            pdb_path: Wild-type PDB structure.
            mutants: List of (position, mutant_amino_acid) tuples.
            wild_type_seq: Wild-type sequence (auto-detected if None).
            parallel: Use parallel execution if True.

        Returns:
            List of MutantEvaluationResult for each mutant.
        """
        if wild_type_seq is None:
            wild_type_seq = parse_pdb_sequence(pdb_path)

        results: list[MutantEvaluationResult] = []

        if parallel:
            # Use multiprocessing for parallel evaluation
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.evaluate_mutant, pdb_path, pos, mut, wild_type_seq): (pos, mut)
                    for pos, mut in mutants
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
        else:
            for position, mutant in mutants:
                result = self.evaluate_mutant(pdb_path, position, mutant, wild_type_seq)
                if result:
                    results.append(result)

        return results


# =============================================================================
# Convenience Functions
# =============================================================================


def check_rosetta_available() -> dict[str, Any]:
    """Check Rosetta availability and return configuration info.

    Returns:
        Dictionary with availability status and recommended setup.
    """
    info: dict[str, Any] = {
        "available": False,
        "mode": None,
        "local": None,
        "docker": None,
        "singularity": None,
        "recommendations": [],
    }

    # Check local installation
    local = find_rosetta_installation()
    info["local"] = local
    if local["available"]:
        info["available"] = True
        info["mode"] = "local"

    # Check Docker
    docker_available = check_docker_available()
    info["docker"] = {"available": docker_available}
    if docker_available and not info["available"]:
        info["recommendations"].append(
            "Docker available. Run Rosetta with: RosettaScorer(RosettaConfig(mode='docker'))"
        )

    # Check Singularity
    singularity_available = check_singularity_available()
    info["singularity"] = {"available": singularity_available}
    if singularity_available and not info["available"]:
        info["recommendations"].append(
            "Singularity available. Run Rosetta with: RosettaScorer(RosettaConfig(mode='singularity'))"
        )

    # Build recommendations
    if not info["available"]:
        info["recommendations"] = [
            "Install Rosetta locally: https://www.rosettacommons.org/software",
            "Or use Docker: docker pull rossettacommons/rosetta",
            "Or use Singularity to convert Docker image to SIF",
            "Set ROSETTA_BIN and ROSETTA_DB environment variables",
        ]

    return info


def quick_score(pdb_path: Path, score_function: str = "talaris2014") -> float | None:
    """Quick scoring function for one-liners.

    Args:
        pdb_path: PDB file to score.
        score_function: Scoring function name.

    Returns:
        Total score or None if scoring failed.
    """
    scorer = RosettaScorer()
    result = scorer.score_pdb(pdb_path, score_function)
    return result.total_score if result else None


# Import os for environment variable access
import os
