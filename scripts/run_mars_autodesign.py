from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import gemmi
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_mars_pipeline as pipeline_ops
from marsstack.energy_head import serialize_pairwise_energy_tensor
from marsstack.evidence_field import serialize_evidence_fields
from marsstack.field_network import EvidencePaths, MarsFieldSystem, ProteinDesignContext, ScoringInputs
from marsstack.mars_score import SAFE_OXIDATION_MAP
from marsstack.structure_features import AA3_TO_1, analyze_structure, detect_flexible_surface_positions, detect_oxidation_hotspots


DEFAULT_METHOD = {
    "oxidation_min_sasa": 25.0,
    "oxidation_min_dist_protected": 8.0,
    "flexible_surface_min_sasa": 40.0,
    "score_weights": {
        "oxidation": 1.0,
        "surface": 1.0,
        "manual": 1.0,
        "evolution": 1.0,
        "burden": 1.0,
    },
    "local_proposals": {
        "enabled": True,
        "max_variants_per_position": 2,
        "max_candidates": 96,
    },
    "learned_fusion": {
        "decoder_enabled": True,
        "decoder_beam_size": 24,
        "decoder_max_candidates": 24,
        "decoder_mutation_penalty": 0.15,
    },
}


def has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "autodesign_target"


def resolve_chain(st: gemmi.Structure, requested_chain: str | None) -> str:
    model = st[0]
    if requested_chain:
        if requested_chain not in [chain.name for chain in model]:
            raise ValueError(f"Chain '{requested_chain}' not found in structure.")
        return requested_chain
    for chain in model:
        for res in chain:
            if res.entity_type == gemmi.EntityType.Polymer and res.name in AA3_TO_1:
                return chain.name
    raise ValueError("Could not infer a usable polymer chain from the structure.")


def extract_chain_records(pdb_path: Path, chain_id: str) -> tuple[list[dict[str, object]], set[int], set[int], set[int]]:
    st = gemmi.read_structure(str(pdb_path))
    chain = st[0][chain_id]

    ligand_atoms: list[tuple[float, float, float]] = []
    for model in st:
        for model_chain in model:
            for res in model_chain:
                if res.entity_type == gemmi.EntityType.Polymer:
                    continue
                if res.name.upper() in {"HOH", "WAT", "DOD"}:
                    continue
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    ligand_atoms.append((atom.pos.x, atom.pos.y, atom.pos.z))

    records: list[dict[str, object]] = []
    missing_backbone: set[int] = set()
    ligand_adjacent: set[int] = set()
    nonstandard_polymer: set[int] = set()

    for res in chain:
        if res.entity_type != gemmi.EntityType.Polymer:
            continue
        if res.name not in AA3_TO_1:
            nonstandard_polymer.add(int(res.seqid.num))
            continue
        atom_map = {atom.name.strip(): atom for atom in res if atom.element.name != "H"}
        backbone_complete = {"N", "CA", "C"}.issubset(atom_map)
        if not backbone_complete:
            missing_backbone.add(int(res.seqid.num))

        is_ligand_adjacent = False
        if ligand_atoms:
            for atom in atom_map.values():
                ax, ay, az = atom.pos.x, atom.pos.y, atom.pos.z
                for lx, ly, lz in ligand_atoms:
                    dx = ax - lx
                    dy = ay - ly
                    dz = az - lz
                    if (dx * dx + dy * dy + dz * dz) <= 36.0:
                        is_ligand_adjacent = True
                        break
                if is_ligand_adjacent:
                    break
        if is_ligand_adjacent:
            ligand_adjacent.add(int(res.seqid.num))

        records.append(
            {
                "num": int(res.seqid.num),
                "aa": AA3_TO_1[res.name],
                "backbone_complete": backbone_complete,
                "ligand_adjacent": is_ligand_adjacent,
            }
        )
    return records, missing_backbone, ligand_adjacent, nonstandard_polymer


def default_manual_bias(wt: str) -> dict[str, float]:
    if wt in SAFE_OXIDATION_MAP:
        return {aa: round(float(score), 2) for aa, score in SAFE_OXIDATION_MAP[wt].items()}
    hydrating = {"Q": 1.1, "N": 1.0, "S": 0.8, "T": 0.8, "E": 0.7, "D": 0.7}
    if wt in {"F", "W", "Y", "L", "I", "V", "M", "A", "C"}:
        return hydrating
    if wt in {"K", "R", "H", "D", "E", "N", "Q", "S", "T"}:
        return {"Q": 0.9, "N": 0.9, "E": 0.7, "D": 0.7, "S": 0.6, "T": 0.6}
    return {"Q": 0.8, "N": 0.8, "S": 0.6, "T": 0.6}


def rank_positions(
    feature_rows: list[dict[str, object]],
    residue_records: list[dict[str, object]],
    protected_positions: set[int],
    oxidation_hotspots: set[int],
    flexible_positions: set[int],
) -> list[dict[str, object]]:
    feature_map = {int(row["num"]): row for row in feature_rows}
    mean_bs = [float(row["mean_b"]) for row in feature_rows]
    min_b = min(mean_bs) if mean_bs else 0.0
    max_b = max(mean_bs) if mean_bs else 1.0
    scale_b = max(max_b - min_b, 1e-6)

    ranked: list[dict[str, object]] = []
    for record in residue_records:
        pos = int(record["num"])
        if pos in protected_positions:
            continue
        feature = feature_map[pos]
        sasa = float(feature["sasa"])
        mean_b = float(feature["mean_b"])
        surface_norm = max(0.0, min(1.5, sasa / 80.0))
        flex_norm = max(0.0, min(1.5, (mean_b - min_b) / scale_b))
        oxidation_bonus = 1.8 * surface_norm if pos in oxidation_hotspots else 0.0
        flexibility_bonus = 1.2 * flex_norm if pos in flexible_positions else 0.0
        surface_bonus = 0.6 * surface_norm
        burial_penalty = -0.9 if sasa < 12.0 else 0.0
        glyco_penalty = -0.4 if bool(feature.get("glyco_motif")) else 0.0
        score = round(oxidation_bonus + flexibility_bonus + surface_bonus + burial_penalty + glyco_penalty, 4)
        ranked.append(
            {
                "position": pos,
                "wt_residue": str(record["aa"]),
                "sasa": round(sasa, 3),
                "mean_b": round(mean_b, 3),
                "oxidation_prone": "yes" if pos in oxidation_hotspots else "no",
                "high_flexibility": "yes" if pos in flexible_positions else "no",
                "ligand_adjacent": "yes" if bool(record["ligand_adjacent"]) else "no",
                "backbone_complete": "yes" if bool(record["backbone_complete"]) else "no",
                "eligibility_score": score,
                "rank_reason": "; ".join(
                    [
                        *(["oxidation_hotspot"] if pos in oxidation_hotspots else []),
                        *(["flexible_surface"] if pos in flexible_positions else []),
                        *(["surface_opportunity"] if sasa >= 25.0 else []),
                        *(["buried_deprioritized"] if sasa < 12.0 else []),
                    ]
                )
                or "surface_background",
            }
        )
    ranked.sort(key=lambda row: (-float(row["eligibility_score"]), int(row["position"])))
    return ranked


def runtime_capabilities() -> dict[str, object]:
    proteinmpnn_root = ROOT / "vendors" / "ProteinMPNN"
    esm_root = ROOT / "vendors" / "esm-main"
    proteinmpnn_available = (
        proteinmpnn_root.exists()
        and (proteinmpnn_root / "protein_mpnn_run.py").exists()
        and (proteinmpnn_root / "helper_scripts" / "parse_multiple_chains.py").exists()
    )
    esm_if_available = esm_root.exists() and has_module("torch") and has_module("gemmi")
    try:
        from marsstack.field_network.neural_dataset import load_neural_corpus

        neural_corpus_count = len(load_neural_corpus(ROOT / "outputs"))
    except Exception:
        neural_corpus_count = 0
    neural_available = has_module("torch") and neural_corpus_count > 0
    return {
        "proteinmpnn_available": proteinmpnn_available,
        "esm_if_available": esm_if_available,
        "neural_available": neural_available,
        "neural_corpus_count": neural_corpus_count,
    }


def build_generated_config(
    *,
    target_name: str,
    pdb_path: Path,
    chain_id: str,
    wt_sequence: str,
    protected_positions: list[int],
    design_positions: list[int],
    manual_bias: dict[int, dict[str, float]],
    branches: dict[str, object],
    homologs_fasta: Path | None,
    asr_fasta: Path | None,
    family_manifest: Path | None,
) -> dict[str, object]:
    cfg: dict[str, object] = {
        "protein": {
            "name": f"{target_name}_auto",
            "pdb_path": str(pdb_path),
            "chain": chain_id,
            "wt_sequence": wt_sequence,
            "protected_positions": protected_positions,
        },
        "generation": {
            "design_positions": design_positions,
            "num_seq_per_target": 24,
            "sampling_temp": "0.1 0.2",
            "seed": 42,
            "batch_size": 1,
            "use_soluble_model": True,
            "protein_mpnn": {
                "enabled": bool(branches["proteinmpnn_available"]),
            },
            "esm_if": {
                "enabled": bool(branches["esm_if_available"]),
                "num_samples": 8,
                "temperature": 1.0e-6,
                "nogpu": False,
            },
        },
        "method": {
            **DEFAULT_METHOD,
            "manual_bias": manual_bias,
            "neural_rerank": {
                "enabled": bool(branches["neural_available"]),
                "epochs": 1,
                "lr": 0.001,
                "selection_policy": "hybrid",
                "decoder_enabled": bool(branches["neural_available"]),
            },
        },
        "evolution": {
            "min_identity": 0.2,
        },
        "benchmark": {
            "family": "autodesign",
        },
    }
    if homologs_fasta:
        cfg["evolution"]["homologs_fasta"] = str(homologs_fasta)
    if asr_fasta:
        cfg["evolution"]["asr_fasta"] = str(asr_fasta)
        cfg["evolution"]["asr_prior_scale"] = 0.55
    if family_manifest:
        cfg["evolution"]["family_manifest"] = str(family_manifest)
        cfg["evolution"]["family_top_k"] = 3
        cfg["evolution"]["family_min_delta"] = 0.05
        cfg["evolution"]["family_prior_scale"] = 0.60
    return cfg


def write_summary_markdown(out_dir: Path, summary: dict[str, object]) -> None:
    lines = [
        f"# {summary['target_name']} autodesign summary",
        "",
        f"- mode: {summary['mode']}",
        f"- pdb: {summary['pdb_path']}",
        f"- chain: {summary['chain']}",
        f"- inferred WT length: {summary['wt_length']}",
        f"- hard protected positions: {summary['protected_count']}",
        f"- ligand-adjacent protected: {summary['ligand_adjacent_count']}",
        f"- disulfide protected: {summary['disulfide_count']}",
        f"- missing-backbone protected: {summary['missing_backbone_count']}",
        f"- eligible positions: {summary['eligible_count']}",
        f"- reported top positions: {summary['reported_position_count']}",
        f"- design positions used: {summary['design_position_count']}",
        f"- branch protein_mpnn: {'yes' if summary['branches']['proteinmpnn_available'] else 'no'}",
        f"- branch esm_if: {'yes' if summary['branches']['esm_if_available'] else 'no'}",
        f"- branch neural: {'yes' if summary['branches']['neural_available'] else 'no'}",
        f"- homologs provided: {'yes' if summary['inputs']['homologs_fasta'] else 'no'}",
        f"- asr provided: {'yes' if summary['inputs']['asr_fasta'] else 'no'}",
        f"- family manifest provided: {'yes' if summary['inputs']['family_manifest'] else 'no'}",
        "",
        "## Top reported positions",
        "",
    ]
    for item in summary["top_positions"]:
        lines.append(
            f"- {item['position']} {item['wt_residue']}: score={item['eligibility_score']} reason={item['rank_reason']}"
        )
    if summary.get("pipeline_output_dir"):
        lines.extend(["", f"- pipeline output dir: {summary['pipeline_output_dir']}"])
    (out_dir / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def write_generated_config(path: Path, cfg: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def build_analysis_outputs(
    *,
    out_dir: Path,
    context: ProteinDesignContext,
    branches: dict[str, object],
    manual_bias: dict[int, dict[str, float]],
    evidence_paths: EvidencePaths,
    method_cfg: dict[str, object],
    top_k: int,
    reported_positions: list[dict[str, object]],
) -> dict[str, object]:
    mars_system = MarsFieldSystem(outputs_root=ROOT / "outputs")
    features = analyze_structure(context.pdb_path, context.chain_id, context.protected_positions)
    feature_rows = [vars(f) for f in features]
    pd.DataFrame(feature_rows).to_csv(out_dir / "structure_features.csv", index=False)

    evidence_bundle = mars_system.build_evidence(
        context=context,
        paths=evidence_paths,
        oxidation_min_sasa=float(method_cfg["oxidation_min_sasa"]),
        oxidation_min_dist_protected=float(method_cfg["oxidation_min_dist_protected"]),
        flexible_surface_min_sasa=float(method_cfg["flexible_surface_min_sasa"]),
        min_identity=float(evidence_paths.homologs_fasta is not None and 0.2 or 0.2),
    )
    oxidation_hotspots = evidence_bundle.geometric.oxidation_hotspots
    flexible_positions = evidence_bundle.geometric.flexible_positions
    profile = evidence_bundle.evolution.homolog_profile
    family_positive_profile = evidence_bundle.evolution.family_positive_profile
    family_negative_profile = evidence_bundle.evolution.family_negative_profile
    family_recommendations = evidence_bundle.evolution.family_recommendations
    evolution_position_weights = evidence_bundle.evolution.position_weights
    asr_profile = evidence_bundle.ancestral.asr_profile
    asr_recommendations = evidence_bundle.ancestral.recommendations
    profile_summary = dict(evidence_bundle.evolution.profile_summary)
    profile_summary.update(
        {
            "asr_prior_enabled": bool(asr_profile is not None),
            "accepted_asr": int(evidence_bundle.ancestral.diagnostics.get("accepted_asr", 0)),
            "oxidation_hotspots": oxidation_hotspots,
            "flexible_surface_positions": flexible_positions,
        }
    )

    candidates = mars_system.generate_candidates(
        context=context,
        manual_bias=manual_bias,
        geometric_features=features,
        proposal_rows=[],
        profile=profile,
        family_recommendations=family_recommendations,
        asr_recommendations=asr_recommendations,
        local_enabled=True,
        local_max_variants_per_position=int(method_cfg["local_proposals"]["max_variants_per_position"]),
        local_max_candidates=int(method_cfg["local_proposals"]["max_candidates"]),
    )
    scoring_inputs = ScoringInputs(
        wt_seq=context.wt_sequence,
        features=features,
        oxidation_hotspots=oxidation_hotspots,
        flexible_positions=flexible_positions,
        profile=profile,
        asr_profile=asr_profile,
        family_positive_profile=family_positive_profile,
        family_negative_profile=family_negative_profile,
        manual_preferred=manual_bias,
        design_positions=context.design_positions,
        term_weights=method_cfg["score_weights"],
        position_to_index=context.position_to_index,
        evolution_position_weights=evolution_position_weights,
        residue_numbers=list(context.position_to_index.keys()),
        profile_prior_scale=float(evidence_paths.homologs_fasta is not None and 0.35 or 0.35),
        asr_prior_scale=float(0.55 if evidence_paths.asr_fasta else 0.45),
        family_prior_scale=float(0.60 if evidence_paths.family_manifest else 0.60),
        topic_name=None,
        topic_cfg=None,
    )
    rows = mars_system.score_candidates(list(candidates.values()), scoring_inputs)
    for row in rows:
        row["selection_score"] = float(row.get("mars_score", 0.0))
        row["selection_score_name"] = "mars_score"
        row["engineering_score"] = float(row.get("mars_score", 0.0))
        row["source_group"] = row.get("source_group", pipeline_ops.classify_source_group(str(row["source"])))
    rows.sort(key=lambda item: (-float(item["selection_score"]), str(item["mutations"]), str(item["source"])))
    field_build = mars_system.construct_field(bundle=evidence_bundle, proposal_rows=rows)
    decoded_candidates = mars_system.decode(
        context=context,
        field=field_build.field,
        beam_size=int(method_cfg["learned_fusion"]["decoder_beam_size"]),
        max_candidates=int(method_cfg["learned_fusion"]["decoder_max_candidates"]),
        mutation_penalty=float(method_cfg["learned_fusion"]["decoder_mutation_penalty"]),
    )

    save_rows_csv(out_dir / "combined_ranked_candidates.csv", rows)
    pipeline_ops.proposal_ops.write_shortlist_fasta(rows[:top_k], out_dir / "shortlist_top.fasta")
    (out_dir / "position_fields.json").write_text(json.dumps(serialize_evidence_fields(field_build.field.position_fields), indent=2), encoding="utf-8")
    (out_dir / "pairwise_energy_tensor.json").write_text(json.dumps(serialize_pairwise_energy_tensor(field_build.field.pairwise_tensor), indent=2), encoding="utf-8")
    (out_dir / "retrieval_memory_hits.json").write_text(
        json.dumps(
            {
                "recommendations": field_build.bundle.retrieval.recommendations,
                "neighbors": field_build.bundle.retrieval.diagnostics,
                "atlas": field_build.bundle.retrieval.atlas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "ancestral_field.json").write_text(json.dumps(field_build.bundle.ancestral.ancestral_field, indent=2), encoding="utf-8")
    (out_dir / "feature_summary.json").write_text(
        json.dumps(
            {
                "protein": context.target,
                "chain": context.chain_id,
                "oxidation_hotspots": oxidation_hotspots,
                "flexible_surface_positions": flexible_positions,
                "design_positions": context.design_positions,
                "protected_positions": sorted(context.protected_positions),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "profile_summary.json").write_text(json.dumps(profile_summary, indent=2), encoding="utf-8")
    (out_dir / "decoder_preview.json").write_text(
        json.dumps(
            [
                {
                    "sequence": item.sequence,
                    "mutations": item.mutations,
                    "decoder_score": item.decoder_score,
                    "mutation_count": item.mutation_count,
                    "supporting_sources": item.supporting_sources,
                }
                for item in decoded_candidates
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "mode": "analyze",
        "target_name": context.target,
        "pdb_path": str(context.pdb_path),
        "chain": context.chain_id,
        "wt_length": len(context.wt_sequence),
        "protected_count": len(context.protected_positions),
        "ligand_adjacent_count": sum(1 for row in reported_positions if row["ligand_adjacent"] == "yes"),
        "disulfide_count": sum(1 for row in feature_rows if bool(row.get("in_disulfide"))),
        "missing_backbone_count": 0,
        "eligible_count": len(reported_positions),
        "reported_position_count": len(reported_positions[:24]),
        "design_position_count": len(context.design_positions),
        "branches": branches,
        "inputs": {
            "homologs_fasta": str(evidence_paths.homologs_fasta) if evidence_paths.homologs_fasta else "",
            "asr_fasta": str(evidence_paths.asr_fasta) if evidence_paths.asr_fasta else "",
            "family_manifest": str(evidence_paths.family_manifest) if evidence_paths.family_manifest else "",
        },
        "top_positions": reported_positions[:24],
        "candidate_count": len(rows),
        "decoder_preview_count": len(decoded_candidates),
        "active_evidence_branches": {
            "geometry": True,
            "evolution": bool(profile is not None),
            "ancestry": bool(asr_profile is not None),
            "retrieval": True,
            "environment": True,
        },
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_markdown(out_dir, summary)
    save_rows_csv(out_dir / "ranked_design_positions.csv", reported_positions)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze or design from an arbitrary protein PDB.")
    parser.add_argument("mode", choices=["analyze", "design"])
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--chain", type=str, default="")
    parser.add_argument("--protein-name", type=str, default="")
    parser.add_argument("--homologs-fasta", type=Path, default=None)
    parser.add_argument("--asr-fasta", type=Path, default=None)
    parser.add_argument("--family-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "autodesign")
    parser.add_argument("--top-design-positions", type=int, default=12)
    parser.add_argument("--report-positions", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    pdb_path = args.pdb.resolve()
    st = gemmi.read_structure(str(pdb_path))
    chain_id = resolve_chain(st, args.chain or None)
    residue_records, missing_backbone, ligand_adjacent, nonstandard_polymer = extract_chain_records(pdb_path, chain_id)
    wt_sequence = "".join(str(item["aa"]) for item in residue_records)
    if not wt_sequence:
        raise ValueError("No standard polymer residues could be extracted from the requested chain.")

    protein_name = args.protein_name.strip() if args.protein_name else pdb_path.stem
    slug = slugify(protein_name)
    out_dir = (args.output_root / slug).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    copied_pdb = out_dir / pdb_path.name
    shutil.copy2(pdb_path, copied_pdb)

    temp_features = analyze_structure(copied_pdb, chain_id, protected_positions=set())
    residue_numbers = [int(f.num) for f in temp_features]
    disulfide_positions = {int(f.num) for f in temp_features if f.in_disulfide}
    protected_positions = sorted(set(missing_backbone) | set(ligand_adjacent) | set(disulfide_positions))
    position_to_index = {num: idx for idx, num in enumerate(residue_numbers)}
    feature_rows = [vars(item) for item in temp_features]
    oxidation_hotspots = set(detect_oxidation_hotspots(temp_features, 25.0, 0.0))
    flexible_positions = set(detect_flexible_surface_positions(temp_features, 25.0))
    ranked_positions = rank_positions(feature_rows, residue_records, set(protected_positions), oxidation_hotspots, flexible_positions)
    reported_positions = ranked_positions[: args.report_positions]
    chosen_design_positions = [int(item["position"]) for item in ranked_positions[: args.top_design_positions]]

    manual_bias = {}
    for pos in chosen_design_positions:
        wt = wt_sequence[position_to_index[pos]]
        manual_bias[pos] = default_manual_bias(wt)

    branches = runtime_capabilities()
    evidence_paths = EvidencePaths(
        homologs_fasta=args.homologs_fasta.resolve() if args.homologs_fasta else None,
        asr_fasta=args.asr_fasta.resolve() if args.asr_fasta else None,
        family_manifest=args.family_manifest.resolve() if args.family_manifest else None,
    )
    generated_cfg = build_generated_config(
        target_name=slug,
        pdb_path=copied_pdb,
        chain_id=chain_id,
        wt_sequence=wt_sequence,
        protected_positions=protected_positions,
        design_positions=chosen_design_positions,
        manual_bias=manual_bias,
        branches=branches,
        homologs_fasta=evidence_paths.homologs_fasta,
        asr_fasta=evidence_paths.asr_fasta,
        family_manifest=evidence_paths.family_manifest,
    )
    config_path = out_dir / "generated_config.yaml"
    write_generated_config(config_path, generated_cfg)

    context = ProteinDesignContext(
        target=generated_cfg["protein"]["name"],
        pdb_path=copied_pdb,
        chain_id=chain_id,
        wt_sequence=wt_sequence,
        design_positions=chosen_design_positions,
        protected_positions=set(protected_positions),
        position_to_index=position_to_index,
        score_weights=dict(DEFAULT_METHOD["score_weights"]),
        metadata={
            "oxidation_hotspots": sorted(oxidation_hotspots),
            "flexible_positions": sorted(flexible_positions),
            "nonstandard_polymer_positions": sorted(nonstandard_polymer),
        },
    )

    method_cfg = {**DEFAULT_METHOD}
    summary = build_analysis_outputs(
        out_dir=out_dir,
        context=context,
        branches=branches,
        manual_bias=manual_bias,
        evidence_paths=evidence_paths,
        method_cfg=method_cfg,
        top_k=args.top_k,
        reported_positions=reported_positions,
    )
    summary["missing_backbone_count"] = len(missing_backbone)
    summary["ligand_adjacent_count"] = len(ligand_adjacent)
    summary["disulfide_count"] = len(disulfide_positions)
    summary["nonstandard_polymer_count"] = len(nonstandard_polymer)
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_markdown(out_dir, summary)

    if args.mode == "design":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_mars_pipeline.py"),
            "--config",
            str(config_path),
            "--top-k",
            str(args.top_k),
            "--decoder-enabled",
            "true",
            "--neural-rerank",
            "true" if branches["neural_available"] else "false",
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        pipeline_out = ROOT / "outputs" / f"{generated_cfg['protein']['name'].lower()}_pipeline"
        design_summary = dict(summary)
        design_summary["mode"] = "design"
        design_summary["pipeline_output_dir"] = str(pipeline_out)
        design_summary["active_evidence_branches"] = {
            "geometry": True,
            "evolution": bool(args.homologs_fasta),
            "ancestry": bool(args.asr_fasta),
            "retrieval": True,
            "environment": True,
        }
        (out_dir / "analysis_summary.json").write_text(json.dumps(design_summary, indent=2), encoding="utf-8")
        write_summary_markdown(out_dir, design_summary)

    print(f"Wrote autodesign outputs to {out_dir}")


if __name__ == "__main__":
    main()
