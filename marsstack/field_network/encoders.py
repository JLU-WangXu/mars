from __future__ import annotations

from pathlib import Path

from ..ancestral_field import build_ancestral_posterior_field
from ..evolution import (
    build_family_pair_profiles,
    build_profile,
    build_profile_from_homologs,
    build_structure_position_weights,
    differential_family_recommendations,
    load_aligned_fasta,
    load_fasta,
    load_yaml,
    profile_recommendations,
)
from ..retrieval_memory import retrieve_residue_memory
from ..structure_features import (
    analyze_structure,
    detect_flexible_surface_positions,
    detect_oxidation_hotspots,
)
from .contracts import (
    AncestralEvidence,
    EvidencePaths,
    EnvironmentEvidence,
    EvolutionEvidence,
    GeometricEvidence,
    ProteinDesignContext,
    RetrievalEvidence,
)


def _resolve_project_path(project_root: Path, path_obj: Path | None) -> Path | None:
    if path_obj is None:
        return None
    if path_obj.is_absolute():
        return path_obj
    dataset_candidate = (project_root / "datasets" / path_obj).resolve()
    if dataset_candidate.exists():
        return dataset_candidate
    return (project_root / path_obj).resolve()


def _summarize_aligned_entries(entries: list[tuple[str, str]], wt_seq: str) -> tuple[int, float]:
    if not entries:
        return 0, 0.0
    accepted = max(0, len(entries) - 1)
    if accepted <= 0:
        return accepted, 0.0
    coverage = [sum(1 for _, seq in entries[1:] if seq[i] != "-") for i in range(len(wt_seq))]
    mean_coverage = round(sum(coverage) / max(1, len(wt_seq) * accepted), 3)
    return accepted, mean_coverage


class GeometricEncoder:
    def encode(
        self,
        context: ProteinDesignContext,
        oxidation_min_sasa: float,
        oxidation_min_dist_protected: float,
        flexible_surface_min_sasa: float,
    ) -> GeometricEvidence:
        features = analyze_structure(
            pdb_path=context.pdb_path,
            chain_id=context.chain_id,
            protected_positions=context.protected_positions,
        )
        oxidation_hotspots = detect_oxidation_hotspots(
            features=features,
            min_sasa=float(oxidation_min_sasa),
            min_dist_protected=float(oxidation_min_dist_protected),
        )
        flexible_positions = sorted(
            set(
                detect_flexible_surface_positions(
                    features=features,
                    min_sasa=float(flexible_surface_min_sasa),
                )
            )
            | set(context.design_positions)
        )
        return GeometricEvidence(
            features=[vars(item) for item in features],
            oxidation_hotspots=oxidation_hotspots,
            flexible_positions=flexible_positions,
        )


class PhyloSequenceEncoder:
    def encode(
        self,
        context: ProteinDesignContext,
        project_root: Path,
        paths: EvidencePaths,
        min_identity: float = 0.20,
        family_min_identity: float | None = None,
        family_top_k: int = 3,
        family_min_delta: float = 0.05,
        geometric: GeometricEvidence | None = None,
        template_weight_cfg: dict[str, object] | None = None,
    ) -> EvolutionEvidence:
        homologs_fasta = _resolve_project_path(project_root, paths.homologs_fasta)
        aligned_fasta = _resolve_project_path(project_root, paths.aligned_fasta)
        family_manifest_path = _resolve_project_path(project_root, paths.family_manifest)
        profile = None
        aligned_entries: list[tuple[str, str]] = []
        diagnostics: dict[str, object] = {}
        profile_summary = {
            "input_homologs": 0,
            "accepted_homologs": 0,
            "mean_coverage": 0.0,
            "mean_coverage_fraction": 0.0,
            "family_prior_enabled": False,
            "family_dataset_id": "",
            "family_adaptation_axis": "",
            "input_positive": 0,
            "input_negative": 0,
            "accepted_positive": 0,
            "accepted_negative": 0,
            "mean_positive_coverage": 0.0,
            "mean_negative_coverage": 0.0,
            "template_weighting_enabled": False,
            "template_weight_source": "",
            "template_context_reference": "",
            "evolution_position_weights": {},
        }

        if homologs_fasta is not None and homologs_fasta.exists():
            homolog_entries = load_fasta(homologs_fasta)
            aligned_entries, profile = build_profile_from_homologs(
                wt_seq=context.wt_sequence,
                homolog_entries=homolog_entries,
                min_identity=float(min_identity),
            )
            accepted_homologs, mean_coverage = _summarize_aligned_entries(aligned_entries, context.wt_sequence)
            profile_summary.update(
                {
                    "input_homologs": len(homolog_entries),
                    "accepted_homologs": accepted_homologs,
                    "mean_coverage": mean_coverage,
                    "mean_coverage_fraction": mean_coverage,
                }
            )
        elif aligned_fasta is not None and aligned_fasta.exists():
            aligned = load_aligned_fasta(aligned_fasta)
            aligned_entries = [(f"aligned_{i+1}", seq) for i, seq in enumerate(aligned)]
            profile = build_profile(
                aligned,
                context.wt_sequence,
            )
            accepted_homologs, mean_coverage = _summarize_aligned_entries([("wt_reference", context.wt_sequence)] + aligned_entries, context.wt_sequence)
            profile_summary.update(
                {
                    "input_homologs": accepted_homologs,
                    "accepted_homologs": accepted_homologs,
                    "mean_coverage": mean_coverage,
                    "mean_coverage_fraction": mean_coverage,
                }
            )

        family_recommendations: dict[int, dict[str, float]] = {}
        family_positive_profile = None
        family_negative_profile = None
        template_context_reference = ""
        if family_manifest_path is not None and family_manifest_path.exists():
            family_manifest = load_yaml(family_manifest_path)
            family_positive_fasta = _resolve_project_path(project_root, Path(str(family_manifest.get("positive_fasta", ""))))
            family_negative_fasta = _resolve_project_path(project_root, Path(str(family_manifest.get("negative_fasta", ""))))
            _, positive_profile, negative_profile = build_family_pair_profiles(
                wt_seq=context.wt_sequence,
                positive_entries=load_fasta(family_positive_fasta),
                negative_entries=load_fasta(family_negative_fasta),
                min_identity=float(family_min_identity if family_min_identity is not None else min_identity),
            )
            family_positive_profile = positive_profile
            family_negative_profile = negative_profile
            family_recommendations = differential_family_recommendations(
                positive_profile=positive_profile,
                negative_profile=negative_profile,
                positions=context.design_positions,
                position_to_index=context.position_to_index,
                top_k=int(family_top_k),
                min_delta=float(family_min_delta),
            )
            diagnostics["family_positive_loaded"] = True
            family_summary, _, _ = build_family_pair_profiles(
                wt_seq=context.wt_sequence,
                positive_entries=load_fasta(family_positive_fasta),
                negative_entries=load_fasta(family_negative_fasta),
                min_identity=float(family_min_identity if family_min_identity is not None else min_identity),
            )
            profile_summary.update(
                {
                    "family_prior_enabled": True,
                    "family_dataset_id": str(family_manifest.get("dataset_id", family_manifest_path.stem)),
                    "family_adaptation_axis": str(family_manifest.get("adaptation_axis", "")),
                    "input_positive": int(family_summary["input_positive"]),
                    "input_negative": int(family_summary["input_negative"]),
                    "accepted_positive": int(family_summary["accepted_positive"]),
                    "accepted_negative": int(family_summary["accepted_negative"]),
                    "mean_positive_coverage": float(family_summary["mean_positive_coverage"]),
                    "mean_negative_coverage": float(family_summary["mean_negative_coverage"]),
                }
            )
            template_context_reference = str(family_manifest.get("representative_structure_path", ""))

        position_weights = {}
        if geometric is not None:
            from ..structure_features import ResidueFeature

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
            cfg = dict(template_weight_cfg or {})
            if "enabled" in cfg:
                cfg.pop("enabled", None)
            position_weights = build_structure_position_weights(
                features=features,
                positions=context.design_positions,
                oxidation_hotspots=geometric.oxidation_hotspots,
                flexible_positions=geometric.flexible_positions,
                cfg=cfg,
            )
            if position_weights:
                profile_summary.update(
                    {
                        "template_weighting_enabled": True,
                        "template_weight_source": "target_structure_features",
                        "template_context_reference": template_context_reference or (str(paths.template_structure_path) if paths.template_structure_path else ""),
                        "evolution_position_weights": position_weights,
                    }
                )

        return EvolutionEvidence(
            aligned_entries=aligned_entries,
            homolog_profile=profile,
            family_recommendations=family_recommendations,
            position_weights=position_weights,
            family_positive_profile=family_positive_profile,
            family_negative_profile=family_negative_profile,
            profile_summary=profile_summary,
            template_context_reference=template_context_reference,
            diagnostics=diagnostics,
        )


class AncestralLineageEncoder:
    def encode(
        self,
        context: ProteinDesignContext,
        project_root: Path,
        paths: EvidencePaths,
        min_identity: float = 0.20,
        asr_top_k: int = 3,
        asr_min_prob: float = 0.10,
    ) -> AncestralEvidence:
        asr_fasta = _resolve_project_path(project_root, paths.asr_fasta)
        asr_aligned_fasta = _resolve_project_path(project_root, paths.asr_aligned_fasta)
        asr_profile = None
        asr_entries: list[tuple[str, str]] = []
        diagnostics: dict[str, object] = {"asr_loaded": False}
        if asr_aligned_fasta is not None and asr_aligned_fasta.exists():
            aligned = load_aligned_fasta(asr_aligned_fasta)
            asr_entries = [(f"asr_aligned_{i+1}", seq) for i, seq in enumerate(aligned)]
            asr_profile = build_profile(aligned, context.wt_sequence)
            accepted_asr, mean_asr_coverage = _summarize_aligned_entries([("wt_reference", context.wt_sequence)] + asr_entries, context.wt_sequence)
            diagnostics.update(
                {
                    "asr_loaded": True,
                    "input_asr": accepted_asr,
                    "accepted_asr": accepted_asr,
                    "mean_asr_coverage": mean_asr_coverage,
                }
            )
        elif asr_fasta is not None and asr_fasta.exists():
            raw_entries = load_fasta(asr_fasta)
            asr_entries, asr_profile = build_profile_from_homologs(
                wt_seq=context.wt_sequence,
                homolog_entries=raw_entries,
                min_identity=float(min_identity),
            )
            accepted_asr, mean_asr_coverage = _summarize_aligned_entries(asr_entries, context.wt_sequence)
            diagnostics.update(
                {
                    "asr_loaded": True,
                    "input_asr": len(raw_entries),
                    "accepted_asr": accepted_asr,
                    "mean_asr_coverage": mean_asr_coverage,
                }
            )
        ancestral_field = build_ancestral_posterior_field(
            asr_profile=asr_profile,
            wt_seq=context.wt_sequence,
            positions=context.design_positions,
            position_to_index=context.position_to_index,
            top_k=int(asr_top_k),
            min_prob=float(asr_min_prob),
        )
        recommendations = profile_recommendations(
            profile=asr_profile,
            wt_seq=context.wt_sequence,
            positions=context.design_positions,
            position_to_index=context.position_to_index,
            top_k=max(2, int(asr_top_k)),
            min_prob=float(asr_min_prob),
        )
        return AncestralEvidence(
            ancestral_field=ancestral_field,
            aligned_entries=asr_entries,
            asr_profile=asr_profile,
            recommendations=recommendations,
            diagnostics=diagnostics,
        )


class RetrievalMemoryEncoder:
    def __init__(self, outputs_root: Path) -> None:
        self.outputs_root = outputs_root

    def encode(
        self,
        context: ProteinDesignContext,
    ) -> RetrievalEvidence:
        recommendations, diagnostics, atlas = retrieve_residue_memory(
            target=context.target,
            pdb_path=context.pdb_path,
            chain_id=context.chain_id,
            protected_positions=context.protected_positions,
            design_positions=context.design_positions,
            outputs_root=self.outputs_root,
        )
        return RetrievalEvidence(
            recommendations=recommendations,
            diagnostics=diagnostics,
            atlas=atlas,
        )


class EnvironmentHypernetwork:
    def encode(self, context: ProteinDesignContext) -> EnvironmentEvidence:
        return EnvironmentEvidence(
            context_tokens={
                "oxidation": float(context.score_weights.get("oxidation", 1.0)),
                "surface": float(context.score_weights.get("surface", 1.0)),
                "evolution": float(context.score_weights.get("evolution", 1.0)),
                "burden": float(context.score_weights.get("burden", 1.0)),
            }
        )
