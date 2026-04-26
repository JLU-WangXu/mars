from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..decoder import PositionField


@dataclass
class ProteinDesignContext:
    target: str
    pdb_path: Path
    chain_id: str
    wt_sequence: str
    design_positions: list[int]
    protected_positions: set[int]
    position_to_index: dict[int, int]
    score_weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidencePaths:
    homologs_fasta: Path | None = None
    aligned_fasta: Path | None = None
    asr_fasta: Path | None = None
    asr_aligned_fasta: Path | None = None
    family_manifest: Path | None = None
    template_structure_path: Path | None = None


@dataclass
class GeometricEvidence:
    features: list[dict[str, Any]]
    oxidation_hotspots: list[int]
    flexible_positions: list[int]


@dataclass
class EvolutionEvidence:
    homolog_profile: list[dict[str, float]] | None
    family_recommendations: dict[int, dict[str, float]]
    position_weights: dict[int, float]
    aligned_entries: list[tuple[str, str]] = field(default_factory=list)
    family_positive_profile: list[dict[str, float]] | None = None
    family_negative_profile: list[dict[str, float]] | None = None
    profile_summary: dict[str, Any] = field(default_factory=dict)
    template_context_reference: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AncestralEvidence:
    ancestral_field: dict[int, dict[str, Any]]
    aligned_entries: list[tuple[str, str]] = field(default_factory=list)
    asr_profile: list[dict[str, float]] | None = None
    recommendations: dict[int, dict[str, float]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalEvidence:
    recommendations: dict[int, dict[str, float]]
    diagnostics: dict[int, list[dict[str, Any]]]
    atlas: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EnvironmentEvidence:
    context_tokens: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResidueEnergyField:
    position_fields: list[PositionField]
    pairwise_tensor: dict[tuple[int, int], dict[tuple[str, str], float]]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    context: ProteinDesignContext
    paths: EvidencePaths
    geometric: GeometricEvidence
    evolution: EvolutionEvidence
    ancestral: AncestralEvidence
    retrieval: RetrievalEvidence
    environment: EnvironmentEvidence
    prior_recommendations: dict[str, Any] = field(default_factory=dict)


@dataclass
class FieldBuildResult:
    bundle: EvidenceBundle
    field: ResidueEnergyField
