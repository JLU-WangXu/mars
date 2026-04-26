from __future__ import annotations

from pathlib import Path

from ..decoder import ConstrainedBeamDecoder
from . import proposals as proposal_ops
from .scoring import ScoringInputs, score_candidate_rows
from .contracts import EvidenceBundle, EvidencePaths, FieldBuildResult, ProteinDesignContext, ResidueEnergyField
from .encoders import (
    AncestralLineageEncoder,
    EnvironmentHypernetwork,
    GeometricEncoder,
    PhyloSequenceEncoder,
    RetrievalMemoryEncoder,
)
from .residue_field import UnifiedResidueFieldNetwork


class MarsFieldSystem:
    def __init__(self, outputs_root: Path) -> None:
        self.outputs_root = outputs_root
        self.project_root = outputs_root.parent
        self.geometric_encoder = GeometricEncoder()
        self.phylo_encoder = PhyloSequenceEncoder()
        self.ancestral_encoder = AncestralLineageEncoder()
        self.retrieval_encoder = RetrievalMemoryEncoder(outputs_root=outputs_root)
        self.environment_hypernet = EnvironmentHypernetwork()
        self.field_network = UnifiedResidueFieldNetwork()

    def build_evidence(
        self,
        context: ProteinDesignContext,
        paths: EvidencePaths,
        oxidation_min_sasa: float = 25.0,
        oxidation_min_dist_protected: float = 8.0,
        flexible_surface_min_sasa: float = 40.0,
        min_identity: float = 0.20,
        family_min_identity: float = 0.20,
        family_top_k: int = 3,
        family_min_delta: float = 0.05,
        asr_top_k: int = 3,
        asr_min_prob: float = 0.10,
        template_weight_cfg: dict[str, object] | None = None,
    ) -> EvidenceBundle:
        geometric = self.geometric_encoder.encode(
            context=context,
            oxidation_min_sasa=oxidation_min_sasa,
            oxidation_min_dist_protected=oxidation_min_dist_protected,
            flexible_surface_min_sasa=flexible_surface_min_sasa,
        )
        evolution = self.phylo_encoder.encode(
            context=context,
            project_root=self.project_root,
            paths=paths,
            min_identity=min_identity,
            family_min_identity=family_min_identity,
            family_top_k=family_top_k,
            family_min_delta=family_min_delta,
            geometric=geometric,
            template_weight_cfg=template_weight_cfg,
        )
        ancestral = self.ancestral_encoder.encode(
            context=context,
            project_root=self.project_root,
            paths=paths,
            min_identity=min_identity,
            asr_top_k=asr_top_k,
            asr_min_prob=asr_min_prob,
        )
        retrieval = self.retrieval_encoder.encode(context=context)
        environment = self.environment_hypernet.encode(context=context)
        return EvidenceBundle(
            context=context,
            paths=paths,
            geometric=geometric,
            evolution=evolution,
            ancestral=ancestral,
            retrieval=retrieval,
            environment=environment,
            prior_recommendations={
                "family_recommendations": evolution.family_recommendations,
                "asr_recommendations": ancestral.recommendations,
                "ancestral_field": ancestral.ancestral_field,
            },
        )

    def generate_candidates(
        self,
        context: ProteinDesignContext,
        manual_bias: dict[int, dict[str, float]],
        geometric_features: list[object],
        proposal_rows: list[dict[str, object]] | None = None,
        profile: list[dict[str, float]] | None = None,
        family_recommendations: dict[int, dict[str, float]] | None = None,
        asr_recommendations: dict[int, dict[str, float]] | None = None,
        topic_name: str | None = None,
        topic_cfg: dict[str, object] | None = None,
        local_enabled: bool = True,
        local_max_variants_per_position: int = 5,
        local_max_candidates: int = 256,
    ):
        from collections import OrderedDict

        candidates: "OrderedDict[str, dict[str, object]]" = OrderedDict()
        for entry in proposal_ops.build_manual_candidates(
            wt_seq=context.wt_sequence,
            manual_bias=manual_bias,
            position_to_index=context.position_to_index,
        ):
            proposal_ops.register_candidate(candidates, entry)

        if local_enabled:
            local_candidates = proposal_ops.build_local_proposal_candidates(
                wt_seq=context.wt_sequence,
                design_positions=context.design_positions,
                position_to_index=context.position_to_index,
                features=geometric_features,
                manual_bias=manual_bias,
                oxidation_hotspots=list(context.metadata.get("oxidation_hotspots", [])),
                flexible_positions=list(context.metadata.get("flexible_positions", [])),
                profile=profile,
                family_recommendations=family_recommendations,
                asr_recommendations=asr_recommendations,
                topic_name=topic_name,
                topic_cfg=topic_cfg,
                max_variants_per_position=local_max_variants_per_position,
                max_candidates=local_max_candidates,
            )
            for entry in local_candidates:
                proposal_ops.register_candidate(candidates, entry)

        if proposal_rows:
            for entry in proposal_rows:
                proposal_ops.register_candidate(candidates, entry)
        return candidates

    def score_candidates(
        self,
        candidates: list[dict[str, object]],
        scoring: ScoringInputs,
    ) -> list[dict[str, object]]:
        return score_candidate_rows(candidates=candidates, scoring=scoring)

    def construct_field(
        self,
        bundle: EvidenceBundle,
        proposal_rows: list[dict[str, object]],
    ) -> FieldBuildResult:
        field = self.field_network.construct(
            context=bundle.context,
            geometric=bundle.geometric,
            evolution=bundle.evolution,
            ancestral=bundle.ancestral,
            retrieval=bundle.retrieval,
            environment=bundle.environment,
            proposal_rows=proposal_rows,
        )
        return FieldBuildResult(
            bundle=bundle,
            field=field,
        )

    def build_field(
        self,
        context: ProteinDesignContext,
        proposal_rows: list[dict[str, object]],
        paths: EvidencePaths,
        **kwargs,
    ) -> FieldBuildResult:
        bundle = self.build_evidence(
            context=context,
            paths=paths,
            **kwargs,
        )
        return self.construct_field(
            bundle=bundle,
            proposal_rows=proposal_rows,
        )

    def decode(
        self,
        context: ProteinDesignContext,
        field: ResidueEnergyField,
        beam_size: int = 32,
        max_candidates: int = 32,
        mutation_penalty: float = 0.15,
    ):
        decoder = ConstrainedBeamDecoder(
            beam_size=beam_size,
            max_candidates=max_candidates,
            mutation_penalty=mutation_penalty,
        )
        return decoder.decode(
            wt_seq=context.wt_sequence,
            position_to_index=context.position_to_index,
            fields=field.position_fields,
            pairwise_energies=field.pairwise_tensor,
        )
