from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
HYBRID_SELECTION_TOLERANCE = 0.10

COMPONENT_COLUMNS = {
    "oxidation": "score_oxidation",
    "surface": "score_surface",
    "manual": "score_manual",
    "evolution": "score_evolution",
    "burden": "score_burden",
}


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def load_target_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maybe_run_neural_rerank(
    protein_name: str,
    reuse_existing: bool,
    epochs: int,
    lr: float,
) -> Path | None:
    pipeline_out = ROOT / "outputs" / f"{protein_name.lower()}_pipeline"
    summary_path = pipeline_out / "neural_field_rerank" / "neural_rerank_summary.json"
    if reuse_existing and summary_path.exists():
        return summary_path
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_mars_field_neural_reranker.py"),
        "--target",
        protein_name,
        "--epochs",
        str(int(epochs)),
        "--lr",
        str(float(lr)),
    ]
    run(cmd)
    return summary_path if summary_path.exists() else None


def choose_policy_row(
    overall_row: pd.Series,
    neural_summary: dict[str, object],
    ranked: pd.DataFrame,
    selection_policy: str,
) -> tuple[pd.Series, str]:
    if selection_policy == "current" or not neural_summary:
        return overall_row, "current"

    neural_mutations = str(neural_summary.get("top_neural_mutations", "") or "")
    neural_matches = ranked[ranked["mutations"] == neural_mutations]
    neural_row = neural_matches.iloc[0] if not neural_matches.empty else None
    if neural_row is None:
        return overall_row, "current_fallback"

    if selection_policy == "neural":
        return neural_row, "neural"

    neural_mars = safe_float(neural_summary.get("top_neural_mars_score"))
    current_mars = safe_float(overall_row.get("mars_score"))
    neural_selection = safe_float(neural_row.get("selection_score", neural_row.get("ranking_score")))
    current_selection = safe_float(overall_row.get("selection_score", overall_row.get("ranking_score")))
    selection_delta = neural_selection - current_selection
    if neural_mars >= current_mars and selection_delta >= -HYBRID_SELECTION_TOLERANCE:
        return neural_row, "hybrid_neural"
    return overall_row, "hybrid_current"


def compute_ablation_score(row: pd.Series, weights: dict[str, float]) -> float:
    return round(sum(float(row[column]) * float(weights.get(component, 1.0)) for component, column in COMPONENT_COLUMNS.items()), 3)


def top_by_source(ranked: pd.DataFrame, source: str) -> pd.Series | None:
    subset = ranked[ranked["source"] == source]
    if subset.empty:
        return None
    return subset.iloc[0]


def value_or_na(row: pd.Series | None, field: str) -> str:
    if row is None:
        return "NA"
    return str(row[field])


def score_or_na(row: pd.Series | None, field: str = "mars_score") -> str:
    if row is None:
        return "NA"
    return str(float(row[field]))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_score(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def join_unique(values: pd.Series) -> str:
    cleaned = [str(value) for value in values if not pd.isna(value) and str(value)]
    if not cleaned:
        return "NA"
    return "; ".join(dict.fromkeys(cleaned))


def summarize_family_prior(enabled: pd.Series) -> str:
    flags = [bool(value) for value in enabled.tolist()]
    if flags and all(flags):
        return "family_prior_family"
    if any(flags):
        return "mixed_family"
    return "structure_only_family"


def build_family_summary(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    return (
        benchmark_df.groupby("family", as_index=False)
        .agg(
            n_targets=("target", "count"),
            mean_overall_score=("overall_score", "mean"),
            mean_best_learned_score=("best_learned_score", "mean"),
            family_prior_targets=("family_prior_enabled", "sum"),
        )
        .sort_values("family")
        .reset_index(drop=True)
    )


def build_heldout_family_units(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in sorted(benchmark_df["family"].astype(str).unique().tolist()):
        heldout = benchmark_df[benchmark_df["family"] == family].copy()
        context = benchmark_df[benchmark_df["family"] != family].copy()
        best_overall = heldout.sort_values(["overall_score", "target"], ascending=[False, True]).iloc[0]
        best_learned = heldout.sort_values(["best_learned_score", "target"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "heldout_family": family,
                "split_type": summarize_family_prior(heldout["family_prior_enabled"]),
                "heldout_targets": join_unique(heldout["target"]),
                "family_dataset_ids": join_unique(heldout["family_dataset_id"]),
                "heldout_n_targets": int(len(heldout)),
                "context_n_targets": int(len(context)),
                "heldout_mean_overall_score": round(float(heldout["overall_score"].mean()), 3),
                "heldout_mean_best_learned_score": round(float(heldout["best_learned_score"].mean()), 3),
                "context_mean_overall_score": round(float(context["overall_score"].mean()), 3) if not context.empty else float("nan"),
                "context_mean_best_learned_score": round(float(context["best_learned_score"].mean()), 3) if not context.empty else float("nan"),
                "best_heldout_overall_target": str(best_overall["target"]),
                "best_heldout_overall_source": str(best_overall["overall_source"]),
                "best_heldout_overall_mutations": str(best_overall["overall_mutations"]),
                "best_heldout_overall_score": float(best_overall["overall_score"]),
                "best_heldout_learned_target": str(best_learned["target"]),
                "best_heldout_learned_source": str(best_learned["best_learned_source"]),
                "best_heldout_learned_mutations": str(best_learned["best_learned_mutations"]),
                "best_heldout_learned_score": float(best_learned["best_learned_score"]),
            }
        )
    return pd.DataFrame(rows)


def build_neural_comparison_summary(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    if "neural_rerank_enabled" not in benchmark_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in benchmark_df.iterrows():
        enabled = bool(row.get("neural_rerank_enabled", False))
        if not enabled:
            continue
        neural_mut = str(row.get("neural_top_mutations", ""))
        overall_mut = str(row.get("overall_mutations", ""))
        learned_mut = str(row.get("best_learned_mutations", ""))
        neural_mars = safe_float(row.get("neural_top_mars_score"))
        overall_mars = safe_float(row.get("overall_mars_score"))
        learned_mars = safe_float(row.get("best_learned_mars_score"))
        rows.append(
            {
                "target": str(row["target"]),
                "family": str(row["family"]),
                "neural_top_source": str(row.get("neural_top_source", "")),
                "neural_top_mutations": neural_mut,
                "neural_top_energy": safe_float(row.get("neural_top_energy")),
                "neural_top_policy_pred": safe_float(row.get("neural_top_policy_pred")),
                "neural_top_policy_z": safe_float(row.get("neural_top_policy_z")),
                "neural_top_policy_score": safe_float(row.get("neural_top_policy_score")),
                "neural_top_mars_score": neural_mars,
                "overall_mutations": overall_mut,
                "overall_mars_score": overall_mars,
                "best_learned_mutations": learned_mut,
                "best_learned_mars_score": learned_mars,
                "neural_matches_overall": neural_mut == overall_mut,
                "neural_matches_best_learned": neural_mut == learned_mut,
                "neural_mars_delta_vs_overall": round(neural_mars - overall_mars, 3),
                "neural_mars_delta_vs_best_learned": round(neural_mars - learned_mars, 3),
                "neural_gate_geom": safe_float(row.get("neural_gate_geom")),
                "neural_gate_phylo": safe_float(row.get("neural_gate_phylo")),
                "neural_gate_asr": safe_float(row.get("neural_gate_asr")),
                "neural_gate_retrieval": safe_float(row.get("neural_gate_retrieval")),
                "neural_gate_environment": safe_float(row.get("neural_gate_environment")),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", type=Path, default=ROOT / "configs" / "benchmark_triplet.yaml")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--decoder-enabled", choices=["true", "false"], default=None)
    parser.add_argument("--neural-rerank", action="store_true")
    parser.add_argument("--neural-epochs", type=int, default=3)
    parser.add_argument("--neural-lr", type=float, default=1e-3)
    parser.add_argument("--selection-policy", choices=["current", "neural", "hybrid"], default=None)
    args = parser.parse_args()

    bench_cfg = yaml.safe_load(args.benchmark_config.read_text(encoding="utf-8"))
    target_configs = [Path(p) for p in bench_cfg["targets"]]
    ablations: dict[str, dict[str, float]] = bench_cfg["ablations"]
    neural_cfg = dict(bench_cfg.get("neural_rerank", {}) or {})
    neural_enabled = bool(args.neural_rerank or neural_cfg.get("enabled", False))
    neural_epochs = int(neural_cfg.get("epochs", args.neural_epochs if args.neural_epochs is not None else 3))
    neural_lr = float(neural_cfg.get("lr", args.neural_lr if args.neural_lr is not None else 1e-3))
    selection_policy = str(bench_cfg.get("selection_policy", args.selection_policy or "current"))

    benchmark_name = args.benchmark_config.stem
    out_root = ROOT / "outputs" / benchmark_name
    out_root.mkdir(parents=True, exist_ok=True)

    benchmark_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    protocol_targets: list[dict[str, object]] = []

    for rel_config in target_configs:
        config_path = rel_config if rel_config.is_absolute() else (ROOT / rel_config)
        cfg = load_target_config(config_path)
        protocol_targets.append(
            {
                "config": str(config_path.relative_to(ROOT)),
                "config_sha256": file_sha256(config_path),
                "protein_name": str(cfg["protein"]["name"]),
                "benchmark_label": str(cfg.get("benchmark", {}).get("label", cfg["protein"]["name"])),
                "family_label": str(cfg.get("benchmark", {}).get("family", cfg.get("benchmark", {}).get("label", cfg["protein"]["name"]))),
            }
        )
        protein_name = str(cfg["protein"]["name"])
        target_label = str(cfg.get("benchmark", {}).get("label", protein_name))
        family_label = str(cfg.get("benchmark", {}).get("family", target_label))
        pipeline_out = ROOT / "outputs" / f"{protein_name.lower()}_pipeline"
        ranked_path = pipeline_out / "combined_ranked_candidates.csv"
        profile_path = pipeline_out / "profile_summary.json"
        fusion_summary_path = pipeline_out / "learned_fusion_summary.json"
        neural_summary_path = pipeline_out / "neural_field_rerank" / "neural_rerank_summary.json"

        if not args.reuse_existing or not ranked_path.exists() or not profile_path.exists():
            cmd = [sys.executable, str(ROOT / "scripts" / "run_mars_pipeline.py"), "--config", str(config_path), "--top-k", str(args.top_k)]
            if args.decoder_enabled is not None:
                cmd.extend(["--decoder-enabled", args.decoder_enabled])
            if neural_enabled:
                cmd.extend(["--neural-rerank", "true"])
            run(cmd)

        neural_summary: dict[str, object] = {}
        if neural_enabled:
            neural_path = maybe_run_neural_rerank(
                protein_name=protein_name,
                reuse_existing=args.reuse_existing,
                epochs=neural_epochs,
                lr=neural_lr,
            )
            if neural_path is not None and neural_path.exists():
                neural_summary = json.loads(neural_path.read_text(encoding="utf-8"))

        ranked = pd.read_csv(ranked_path)
        if ranked.empty:
            continue
        score_field = "ranking_score" if "ranking_score" in ranked.columns else "mars_score"
        ranking_model = str(ranked.iloc[0].get("ranking_model", "mars_score_v0"))
        overall_row = ranked.iloc[0]
        learned = ranked[ranked["source_group"] == "learned"] if "source_group" in ranked.columns else ranked[ranked["source"].isin(["baseline_mpnn", "mars_mpnn"])]
        heuristic = ranked[ranked["source_group"] == "heuristic_local"] if "source_group" in ranked.columns else ranked.iloc[0:0]
        manual = ranked[ranked["source_group"] == "manual_control"] if "source_group" in ranked.columns else ranked[ranked["source"] == "manual"]
        best_learned = learned.iloc[0] if not learned.empty else overall_row
        top_mars = top_by_source(ranked, "mars_mpnn")
        top_baseline = top_by_source(ranked, "baseline_mpnn")
        top_esm_if = top_by_source(ranked, "esm_if")
        top_manual = manual.iloc[0] if not manual.empty else None
        top_local = heuristic.iloc[0] if not heuristic.empty else None
        profile_summary = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        fusion_summary = yaml.safe_load(fusion_summary_path.read_text(encoding="utf-8")) if fusion_summary_path.exists() else {}
        policy_row, policy_resolution = choose_policy_row(
            overall_row=overall_row,
            neural_summary=neural_summary,
            ranked=ranked,
            selection_policy=selection_policy,
        )

        benchmark_rows.append(
                {
                "target": target_label,
                "family": family_label,
                "config": str(config_path.relative_to(ROOT)),
                "ranking_model": ranking_model,
                "ranking_score_field": score_field,
                "decoder_enabled": bool(fusion_summary.get("decoder_enabled", False)),
                "decoder_injected": bool(fusion_summary.get("decoder_injected", False)),
                "decoder_novel_count": int(fusion_summary.get("decoder_novel_count", 0)),
                "decoder_rejected_count": int(fusion_summary.get("decoder_rejected_count", 0)),
                "best_decoder_candidate": str(fusion_summary.get("best_decoder_candidate", "")),
                "best_decoder_ranking_score": float(fusion_summary.get("best_decoder_ranking_score", 0.0)),
                "neural_decoder_enabled": bool(fusion_summary.get("neural_decoder_enabled", False)),
                "neural_decoder_injected": bool(fusion_summary.get("neural_decoder_injected", False)),
                "neural_decoder_generated_count": int(fusion_summary.get("neural_decoder_generated_count", 0)),
                "neural_decoder_novel_count": int(fusion_summary.get("neural_decoder_novel_count", 0)),
                "neural_decoder_rejected_count": int(fusion_summary.get("neural_decoder_rejected_count", 0)),
                "best_neural_decoder_candidate": str(fusion_summary.get("best_neural_decoder_candidate", "")),
                "best_neural_decoder_ranking_score": float(fusion_summary.get("best_neural_decoder_ranking_score", 0.0)),
                "neural_rerank_enabled": bool(neural_summary),
                "neural_epochs": int(neural_summary.get("epochs", 0) or 0),
                "neural_top_mutations": str(neural_summary.get("top_neural_mutations", "")),
                "neural_top_source": str(neural_summary.get("top_neural_source", "")),
                "neural_top_energy": float(neural_summary.get("top_neural_energy", 0.0) or 0.0),
                "neural_top_policy_pred": float(neural_summary.get("top_neural_policy_pred", 0.0) or 0.0),
                "neural_top_policy_z": float(neural_summary.get("top_neural_policy_z", 0.0) or 0.0),
                "neural_top_policy_score": float(neural_summary.get("top_neural_policy_score", 0.0) or 0.0),
                "neural_top_mars_score": float(neural_summary.get("top_neural_mars_score", 0.0) or 0.0),
                "neural_gate_geom": float((neural_summary.get("gate_means", {}) or {}).get("geom", 0.0) or 0.0),
                "neural_gate_phylo": float((neural_summary.get("gate_means", {}) or {}).get("phylo", 0.0) or 0.0),
                "neural_gate_asr": float((neural_summary.get("gate_means", {}) or {}).get("asr", 0.0) or 0.0),
                "neural_gate_retrieval": float((neural_summary.get("gate_means", {}) or {}).get("retrieval", 0.0) or 0.0),
                "neural_gate_environment": float((neural_summary.get("gate_means", {}) or {}).get("environment", 0.0) or 0.0),
                "overall_candidate": overall_row["candidate_id"],
                "overall_source": overall_row["source"],
                "overall_mutations": overall_row["mutations"],
                "overall_selection_score": float(overall_row.get("selection_score", overall_row[score_field])),
                "overall_selection_score_name": str(overall_row.get("selection_score_name", score_field)),
                "overall_engineering_score": float(overall_row.get("engineering_score", overall_row["mars_score"])),
                "overall_score": float(overall_row[score_field]),
                "overall_mars_score": float(overall_row["mars_score"]),
                "best_learned_candidate": best_learned["candidate_id"],
                "best_learned_source": best_learned["source"],
                "best_learned_mutations": best_learned["mutations"],
                "best_learned_selection_score": float(best_learned.get("selection_score", best_learned[score_field])),
                "best_learned_selection_score_name": str(best_learned.get("selection_score_name", score_field)),
                "best_learned_engineering_score": float(best_learned.get("engineering_score", best_learned["mars_score"])),
                "best_learned_score": float(best_learned[score_field]),
                "best_learned_mars_score": float(best_learned["mars_score"]),
                "selection_policy": selection_policy,
                "policy_resolution": policy_resolution,
                "policy_candidate": str(policy_row["candidate_id"]),
                "policy_source": str(policy_row["source"]),
                "policy_mutations": str(policy_row["mutations"]),
                "policy_selection_score": float(policy_row.get("selection_score", policy_row[score_field])),
                "policy_selection_score_name": str(policy_row.get("selection_score_name", score_field)),
                "policy_engineering_score": float(policy_row.get("engineering_score", policy_row["mars_score"])),
                "top_baseline_mutations": value_or_na(top_baseline, "mutations"),
                "top_baseline_score": score_or_na(top_baseline, score_field),
                "top_mars_mutations": value_or_na(top_mars, "mutations"),
                "top_mars_score": score_or_na(top_mars, score_field),
                "top_esm_if_mutations": value_or_na(top_esm_if, "mutations"),
                "top_esm_if_score": score_or_na(top_esm_if, score_field),
                "top_local_mutations": value_or_na(top_local, "mutations"),
                "top_local_score": score_or_na(top_local, score_field),
                "top_manual_mutations": value_or_na(top_manual, "mutations"),
                "top_manual_score": score_or_na(top_manual, score_field),
                "accepted_homologs": int(profile_summary["accepted_homologs"]),
                "mean_coverage": float(profile_summary["mean_coverage"]),
                "asr_prior_enabled": bool(profile_summary.get("asr_prior_enabled", False)),
                "accepted_asr": int(profile_summary.get("accepted_asr", 0)),
                "family_prior_enabled": bool(profile_summary.get("family_prior_enabled", False)),
                "family_dataset_id": str(profile_summary.get("family_dataset_id", "")),
                "accepted_positive": int(profile_summary.get("accepted_positive", 0)),
                "accepted_negative": int(profile_summary.get("accepted_negative", 0)),
                "template_weighting_enabled": bool(profile_summary.get("template_weighting_enabled", False)),
                "template_context_reference": str(profile_summary.get("template_context_reference", "")),
            }
        )

        for ablation_name, overrides in ablations.items():
            rescored = ranked.copy()
            rescored["ablation_score"] = ranked.apply(lambda row: compute_ablation_score(row, overrides), axis=1)
            rescored = rescored.sort_values(["ablation_score", "mutations", "source"], ascending=[False, True, True]).reset_index(drop=True)
            best = rescored.iloc[0]
            ablation_rows.append(
                {
                    "target": target_label,
                    "ablation": ablation_name,
                    "top_candidate": best["candidate_id"],
                    "top_source": best["source"],
                    "top_mutations": best["mutations"],
                    "ablation_score": float(best["ablation_score"]),
                }
            )

    benchmark_df = pd.DataFrame(benchmark_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    family_df = build_family_summary(benchmark_df)
    heldout_df = build_heldout_family_units(benchmark_df)
    neural_df = build_neural_comparison_summary(benchmark_df)
    benchmark_df.to_csv(out_root / "benchmark_summary.csv", index=False)
    ablation_df.to_csv(out_root / "ablation_summary.csv", index=False)
    family_df.to_csv(out_root / "family_summary.csv", index=False)
    heldout_df.to_csv(out_root / "heldout_family_units.csv", index=False)
    if not neural_df.empty:
        neural_df.to_csv(out_root / "neural_comparison_summary.csv", index=False)
    protocol_manifest = {
        "protocol_version": "mars_field_benchmark_v1",
        "benchmark_name": benchmark_name,
        "benchmark_config": str(args.benchmark_config.relative_to(ROOT) if args.benchmark_config.is_relative_to(ROOT) else args.benchmark_config),
        "benchmark_config_sha256": file_sha256(args.benchmark_config),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_k": int(args.top_k),
        "reuse_existing": bool(args.reuse_existing),
        "decoder_enabled_override": args.decoder_enabled,
        "neural_rerank": bool(neural_enabled),
        "neural_epochs": int(neural_epochs),
        "neural_lr": float(neural_lr),
        "split_policy": {
            "target_level": "one target per config entry",
            "family_level": "post-hoc held-out family aggregation from benchmark summary",
            "neural_holdout": "leave-one-target-out reranking when --neural-rerank is enabled",
        },
        "score_contract": {
            "selection_score": "candidate selection score used for ranking in the current pipeline output",
            "engineering_score": "mars_score",
            "candidate_table_fields": ["selection_score", "engineering_score", "ranking_score", "mars_score"],
        },
        "targets": protocol_targets,
    }
    (out_root / "benchmark_protocol_manifest.json").write_text(
        json.dumps(protocol_manifest, indent=2),
        encoding="utf-8",
    )
    protocol_lines = [
        f"# Mars {benchmark_name} benchmark protocol",
        "",
        f"- benchmark config: `{protocol_manifest['benchmark_config']}`",
        f"- benchmark config sha256: `{protocol_manifest['benchmark_config_sha256']}`",
        f"- generated at (UTC): `{protocol_manifest['generated_at_utc']}`",
        f"- top_k: `{protocol_manifest['top_k']}`",
        f"- reuse_existing: `{protocol_manifest['reuse_existing']}`",
        f"- decoder_enabled override: `{protocol_manifest['decoder_enabled_override']}`",
        f"- neural_rerank: `{protocol_manifest['neural_rerank']}`",
        f"- neural_epochs: `{protocol_manifest['neural_epochs']}`",
        f"- neural_lr: `{protocol_manifest['neural_lr']}`",
        "",
        "## Split policy",
        "",
        f"- target level: {protocol_manifest['split_policy']['target_level']}",
        f"- family level: {protocol_manifest['split_policy']['family_level']}",
        f"- neural holdout: {protocol_manifest['split_policy']['neural_holdout']}",
        "",
        "## Targets",
        "",
    ]
    for item in protocol_targets:
        protocol_lines.append(
            f"- `{item['benchmark_label']}` | family=`{item['family_label']}` | config=`{item['config']}` | sha256=`{item['config_sha256']}`"
        )
    (out_root / "benchmark_protocol_manifest.md").write_text("\n".join(protocol_lines) + "\n", encoding="utf-8")

    lines = [
        f"# Mars {benchmark_name} summary",
        "",
        "## Targets",
        "",
    ]
    for row in benchmark_rows:
        lines.append(
            f"- {row['target']}: overall `{row['overall_mutations']}` ({row['overall_source']}, selection={row['overall_selection_score']}, engineering={row['overall_engineering_score']}); "
            f"policy `{row['policy_mutations']}` ({row['policy_source']}, selection={row['policy_selection_score']}, engineering={row['policy_engineering_score']}, mode={row['policy_resolution']}); "
            f"best learned `{row['best_learned_mutations']}` ({row['best_learned_source']}, selection={row['best_learned_selection_score']}, engineering={row['best_learned_engineering_score']}); "
            f"neural `{row['neural_top_mutations'] or 'NA'}` ({row['neural_top_source'] or 'NA'}, energy={format_score(row['neural_top_energy'])}, mars={format_score(row['neural_top_mars_score'])}); "
            f"Mars `{row['top_mars_mutations']}`; ESM-IF `{row['top_esm_if_mutations']}`; "
            f"local `{row['top_local_mutations']}`; manual `{row['top_manual_mutations']}`; "
            f"homologs={row['accepted_homologs']}; asr={row['accepted_asr']}; "
            f"family_prior={row['family_dataset_id'] or 'NA'}; template_weighting={'yes' if row['template_weighting_enabled'] else 'no'}; "
            f"ranker={row['ranking_model']}; decoder={'on' if row['decoder_enabled'] else 'off'}; "
            f"decoder_novel={row['decoder_novel_count']}; decoder_rejected={row['decoder_rejected_count']}; "
            f"neural_rerank={'on' if row['neural_rerank_enabled'] else 'off'}"
        )
    lines.extend(["", "## Family Means", ""])
    for _, row in family_df.iterrows():
        lines.append(
            f"- {row['family']}: n={int(row['n_targets'])}, "
            f"mean overall={format_score(row['mean_overall_score'])}, "
            f"mean best learned={format_score(row['mean_best_learned_score'])}, "
            f"family-prior targets={int(row['family_prior_targets'])}"
        )
    lines.extend(["", "## Held-Out Families", ""])
    for _, row in heldout_df.iterrows():
        lines.append(
            f"- {row['heldout_family']} [{row['split_type']}]: held-out mean overall={format_score(row['heldout_mean_overall_score'])}, "
            f"held-out mean best learned={format_score(row['heldout_mean_best_learned_score'])}; "
            f"context mean overall={format_score(row['context_mean_overall_score'])}, "
            f"context mean best learned={format_score(row['context_mean_best_learned_score'])}; "
            f"best overall `{row['best_heldout_overall_mutations']}` ({row['best_heldout_overall_source']}, score={format_score(row['best_heldout_overall_score'])}); "
            f"best learned `{row['best_heldout_learned_mutations']}` ({row['best_heldout_learned_source']}, score={format_score(row['best_heldout_learned_score'])}); "
            f"targets={row['heldout_targets']}; family_prior={row['family_dataset_ids']}"
        )
    lines.extend(["", "## Ablations", ""])
    for target in benchmark_df["target"].tolist():
        target_rows = ablation_df[ablation_df["target"] == target]
        for _, row in target_rows.iterrows():
            lines.append(
                f"- {target} / {row['ablation']}: `{row['top_mutations']}` ({row['top_source']}, score={row['ablation_score']})"
            )
    if not neural_df.empty:
        lines.extend(["", "## Neural Comparison", ""])
        for _, row in neural_df.iterrows():
            lines.append(
                f"- {row['target']}: neural `{row['neural_top_mutations']}` ({row['neural_top_source']}, energy={format_score(row['neural_top_energy'])}, mars={format_score(row['neural_top_mars_score'])}); "
                f"match overall={'yes' if row['neural_matches_overall'] else 'no'}; "
                f"match best learned={'yes' if row['neural_matches_best_learned'] else 'no'}; "
                f"delta vs overall={format_score(row['neural_mars_delta_vs_overall'])}; "
                f"delta vs best learned={format_score(row['neural_mars_delta_vs_best_learned'])}; "
                f"gates geom={format_score(row.get('neural_gate_geom', 0.0))}, phylo={format_score(row.get('neural_gate_phylo', 0.0))}, "
                f"asr={format_score(row.get('neural_gate_asr', 0.0))}, retrieval={format_score(row.get('neural_gate_retrieval', 0.0))}, "
                f"env={format_score(row.get('neural_gate_environment', 0.0))}"
            )
    (out_root / "benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_root / "heldout_family_units.md").write_text(
        "\n".join(
            [
                f"# Mars {benchmark_name} held-out family summary",
                "",
                "## Held-Out Families",
                "",
                *[
                    (
                        f"- {row['heldout_family']} [{row['split_type']}]: held-out targets `{row['heldout_targets']}`; "
                        f"held-out mean overall={format_score(row['heldout_mean_overall_score'])}, "
                        f"held-out mean best learned={format_score(row['heldout_mean_best_learned_score'])}; "
                        f"context mean overall={format_score(row['context_mean_overall_score'])}, "
                        f"context mean best learned={format_score(row['context_mean_best_learned_score'])}; "
                        f"best overall `{row['best_heldout_overall_mutations']}` ({row['best_heldout_overall_source']}, score={format_score(row['best_heldout_overall_score'])}); "
                        f"best learned `{row['best_heldout_learned_mutations']}` ({row['best_heldout_learned_source']}, score={format_score(row['best_heldout_learned_score'])}); "
                        f"family prior `{row['family_dataset_ids']}`"
                    )
                    for _, row in heldout_df.iterrows()
                ],
                "",
                "## Family Means",
                "",
                *[
                    (
                        f"- {row['family']}: n={int(row['n_targets'])}, "
                        f"mean overall={format_score(row['mean_overall_score'])}, "
                        f"mean best learned={format_score(row['mean_best_learned_score'])}"
                    )
                    for _, row in family_df.iterrows()
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote benchmark summary to {out_root / 'benchmark_summary.csv'}")
    print(f"Wrote ablation summary to {out_root / 'ablation_summary.csv'}")
    print(f"Wrote family summary to {out_root / 'family_summary.csv'}")
    print(f"Wrote held-out family summary to {out_root / 'heldout_family_units.csv'}")
    if not neural_df.empty:
        print(f"Wrote neural comparison summary to {out_root / 'neural_comparison_summary.csv'}")
    print(f"Wrote benchmark protocol manifest to {out_root / 'benchmark_protocol_manifest.json'}")


if __name__ == "__main__":
    main()
