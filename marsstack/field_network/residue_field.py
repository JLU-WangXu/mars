from __future__ import annotations

from pathlib import Path

from ..energy_head import build_pairwise_energy_tensor, compute_design_pair_distances
from ..evidence_field import build_unified_evidence_fields
from ..structure_features import ResidueFeature
from .contracts import (
    AncestralEvidence,
    EnvironmentEvidence,
    EvolutionEvidence,
    GeometricEvidence,
    ProteinDesignContext,
    ResidueEnergyField,
    RetrievalEvidence,
)


class UnifiedResidueFieldNetwork:
    def construct(
        self,
        context: ProteinDesignContext,
        geometric: GeometricEvidence,
        evolution: EvolutionEvidence,
        ancestral: AncestralEvidence,
        retrieval: RetrievalEvidence,
        environment: EnvironmentEvidence,
        proposal_rows: list[dict[str, object]],
        top_k_per_position: int = 4,
        max_rows_per_source: int = 16,
    ) -> ResidueEnergyField:
        features = [
            ResidueFeature(
                num=int(row["num"]),
                name=str(row["name"]),
                aa=str(row["aa"]),
                sasa=float(row["sasa"]),
                mean_b=float(row["mean_b"]),
                min_dist_protected=float(row["min_dist_protected"]),
                in_disulfide=bool(row["in_disulfide"]),
                glyco_motif=bool(row["glyco_motif"]),
            )
            for row in geometric.features
        ]
        fields = build_unified_evidence_fields(
            wt_seq=context.wt_sequence,
            design_positions=context.design_positions,
            position_to_index=context.position_to_index,
            features=features,
            oxidation_hotspots=geometric.oxidation_hotspots,
            flexible_positions=geometric.flexible_positions,
            manual_bias=context.metadata.get("manual_bias", {}),
            profile=evolution.homolog_profile,
            family_recommendations=evolution.family_recommendations,
            ancestral_field=ancestral.ancestral_field,
            retrieval_recommendations=retrieval.recommendations,
            proposal_rows=proposal_rows,
            top_k_per_position=top_k_per_position,
            max_rows_per_source=max_rows_per_source,
        )
        pairwise = build_pairwise_energy_tensor(
            rows=proposal_rows,
            fields=fields,
            position_to_index=context.position_to_index,
            pair_distances=compute_design_pair_distances(
                pdb_path=context.pdb_path,
                chain_id=context.chain_id,
                positions=context.design_positions,
            ),
        )
        return ResidueEnergyField(
            position_fields=fields,
            pairwise_tensor=pairwise,
            diagnostics={
                "environment_tokens": environment.context_tokens,
                "retrieval_positions": list(retrieval.recommendations.keys()),
                "ancestral_active": bool(ancestral.ancestral_field),
                "pairwise_edges": len(pairwise),
            },
        )
