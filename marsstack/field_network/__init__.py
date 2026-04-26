"""Canonical MARS-FIELD network abstractions."""

from .contracts import (
    AncestralEvidence,
    EvidenceBundle,
    EvidencePaths,
    EnvironmentEvidence,
    FieldBuildResult,
    EvolutionEvidence,
    GeometricEvidence,
    ProteinDesignContext,
    ResidueEnergyField,
    RetrievalEvidence,
)
from .encoders import (
    AncestralLineageEncoder,
    EnvironmentHypernetwork,
    GeometricEncoder,
    PhyloSequenceEncoder,
    RetrievalMemoryEncoder,
)
from .proposals import (
    SOURCE_PRIORITY,
    build_local_proposal_candidates,
    build_manual_candidates,
    classify_source_group,
    parse_mpnn_fasta,
    parse_sample_fasta,
    register_candidate,
    write_shortlist_fasta,
)
from .residue_field import UnifiedResidueFieldNetwork
from .scoring import ScoringInputs, score_candidate_rows
from .neural_dataset import NeuralTargetBatch, build_runtime_neural_target_batch, load_neural_corpus, load_neural_target_batch
from .neural_model import MarsFieldNeuralModel, NeuralFieldOutput
from .neural_generator import build_neural_residue_field, train_holdout_neural_model
from .system import MarsFieldSystem

__all__ = [
    "AncestralEvidence",
    "EvidenceBundle",
    "EvidencePaths",
    "EnvironmentEvidence",
    "FieldBuildResult",
    "EvolutionEvidence",
    "GeometricEvidence",
    "ProteinDesignContext",
    "ResidueEnergyField",
    "RetrievalEvidence",
    "AncestralLineageEncoder",
    "EnvironmentHypernetwork",
    "GeometricEncoder",
    "PhyloSequenceEncoder",
    "RetrievalMemoryEncoder",
    "SOURCE_PRIORITY",
    "build_local_proposal_candidates",
    "build_manual_candidates",
    "classify_source_group",
    "parse_mpnn_fasta",
    "parse_sample_fasta",
    "register_candidate",
    "write_shortlist_fasta",
    "UnifiedResidueFieldNetwork",
    "ScoringInputs",
    "score_candidate_rows",
    "NeuralTargetBatch",
    "build_runtime_neural_target_batch",
    "load_neural_corpus",
    "load_neural_target_batch",
    "MarsFieldNeuralModel",
    "NeuralFieldOutput",
    "build_neural_residue_field",
    "train_holdout_neural_model",
    "MarsFieldSystem",
]
