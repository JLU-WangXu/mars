from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"
BENCHMARK_CSV = OUTPUTS_DIR / "benchmark_twelvepack" / "benchmark_summary.csv"
NEURAL_COMPARISON_CSV = OUTPUTS_DIR / "benchmark_twelvepack" / "neural_comparison_summary.csv"
PAPER_BUNDLE_DIR = OUTPUTS_DIR / "paper_bundle_v1"
FIGURES_DIR = PAPER_BUNDLE_DIR / "figures"

CASE_STUDIES = [
    {
        "case_id": "case_1lbt",
        "figure_label": "Figure 4",
        "primary_target": "1LBT",
        "companion_targets": [],
        "narrative_role": "Compact benchmark case for field construction, decoder behavior, and final winner calibration.",
    },
    {
        "case_id": "case_tem1",
        "figure_label": "Figure 5",
        "primary_target": "tem1_1btl",
        "companion_targets": [],
        "narrative_role": "Robust multi-site engineering case that highlights selector stability and engineering consistency.",
    },
    {
        "case_id": "case_petase",
        "figure_label": "Figure 6",
        "primary_target": "petase_5xh3",
        "companion_targets": ["petase_5xfy"],
        "narrative_role": "Family-transfer case across related cutinase/PETase structures with aligned mutation logic.",
    },
    {
        "case_id": "case_cld",
        "figure_label": "Figure 7",
        "primary_target": "CLD_3Q09_TOPIC",
        "companion_targets": ["CLD_3Q09_NOTOPIC"],
        "narrative_role": "Ancestry-aware case study linking oxidative shell redesign to lineage-conditioned evidence.",
    },
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _to_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pipeline_dir_for_target(target: str) -> Path:
    return OUTPUTS_DIR / f"{target.lower()}_pipeline"


def _asset_record(target: str) -> dict[str, str]:
    pipeline_dir = _pipeline_dir_for_target(target)
    viz_dir = pipeline_dir / "viz_bundle"
    return {
        "target": target,
        "pipeline_dir": str(pipeline_dir),
        "pipeline_summary": str(pipeline_dir / "pipeline_summary.md"),
        "viz_manifest": str(viz_dir / "viz_manifest.json"),
        "scene_pml": str(viz_dir / "scene.pml"),
        "palette_json": str(viz_dir / "palette.json"),
        "position_fields": str(pipeline_dir / "position_fields.json"),
        "pairwise_tensor": str(pipeline_dir / "pairwise_energy_tensor.json"),
        "retrieval_hits": str(pipeline_dir / "retrieval_memory_hits.json"),
        "ancestral_field": str(pipeline_dir / "ancestral_field.json"),
        "combined_candidates": str(pipeline_dir / "combined_ranked_candidates.csv"),
    }


def _exists_record(asset_record: dict[str, str]) -> dict[str, object]:
    result = dict(asset_record)
    for key, value in list(asset_record.items()):
        if key in {"target"}:
            continue
        result[f"{key}_exists"] = Path(value).exists()
    return result


def build_bundle() -> None:
    rows = _read_rows(BENCHMARK_CSV)
    PAPER_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_overview = []
    decoder_summary = []
    family_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    row_by_target = {row["target"]: row for row in rows}

    for row in rows:
        family_groups[row["family"]].append(row)
        benchmark_overview.append(
            {
                "target": row["target"],
                "family": row["family"],
                "overall_source": row["overall_source"],
                "overall_mutations": row["overall_mutations"],
                "overall_score": _to_float(row["overall_score"]),
                "overall_mars_score": _to_float(row["overall_mars_score"]),
                "best_learned_source": row["best_learned_source"],
                "best_learned_mutations": row["best_learned_mutations"],
                "best_learned_score": _to_float(row["best_learned_score"]),
                "best_learned_mars_score": _to_float(row["best_learned_mars_score"]),
                "decoder_novel_count": _to_int(row["decoder_novel_count"]),
                "decoder_rejected_count": _to_int(row["decoder_rejected_count"]),
                "accepted_homologs": _to_int(row["accepted_homologs"]),
                "accepted_asr": _to_int(row["accepted_asr"]),
                "asr_prior_enabled": _to_bool(row["asr_prior_enabled"]),
                "family_prior_enabled": _to_bool(row["family_prior_enabled"]),
                "family_dataset_id": row["family_dataset_id"] or "NA",
            }
        )
        decoder_summary.append(
            {
                "target": row["target"],
                "decoder_enabled": _to_bool(row["decoder_enabled"]),
                "decoder_injected": _to_bool(row["decoder_injected"]),
                "decoder_novel_count": _to_int(row["decoder_novel_count"]),
                "decoder_rejected_count": _to_int(row["decoder_rejected_count"]),
                "best_decoder_candidate": row["best_decoder_candidate"] or "NA",
                "best_decoder_ranking_score": _to_float(row["best_decoder_ranking_score"]),
                "overall_source": row["overall_source"],
                "overall_mutations": row["overall_mutations"],
                "overall_score": _to_float(row["overall_score"]),
                "overall_mars_score": _to_float(row["overall_mars_score"]),
                "best_learned_source": row["best_learned_source"],
                "best_learned_score": _to_float(row["best_learned_score"]),
            }
        )

    family_summary = []
    for family, members in sorted(family_groups.items()):
        family_summary.append(
            {
                "family": family,
                "target_count": len(members),
                "mean_overall_score": round(
                    sum(_to_float(item["overall_score"]) for item in members) / len(members), 6
                ),
                "mean_best_learned_score": round(
                    sum(_to_float(item["best_learned_score"]) for item in members) / len(members), 6
                ),
                "family_prior_targets": sum(_to_bool(item["family_prior_enabled"]) for item in members),
                "asr_active_targets": sum(_to_bool(item["asr_prior_enabled"]) for item in members),
                "decoder_injected_targets": sum(_to_bool(item["decoder_injected"]) for item in members),
            }
        )

    asset_inventory = [_exists_record(_asset_record(row["target"])) for row in rows]

    case_rows = []
    figure_manifest = [
        {
            "figure_label": "Figure 1",
            "title": "MARS-FIELD architecture",
            "source_doc": str(REPO_ROOT / "docs" / "mars_field_figure1_spec_v2.md"),
            "rendered_figure": str(FIGURES_DIR / "figure1_mars_field_architecture_v1.svg"),
            "caption": str(FIGURES_DIR / "figure1_mars_field_architecture_v1_caption.md"),
            "figure_type": "principle_diagram",
        },
        {
            "figure_label": "Figure 2",
            "title": "Twelvepack benchmark overview",
            "source_table": str(PAPER_BUNDLE_DIR / "figure2_benchmark_overview.csv"),
            "family_table": str(PAPER_BUNDLE_DIR / "figure2_family_summary.csv"),
            "rendered_figure": str(FIGURES_DIR / "figure2_benchmark_overview_v3.svg"),
            "figure_type": "benchmark_overview",
        },
        {
            "figure_label": "Figure 3",
            "title": "Decoder and calibration analysis",
            "source_table": str(PAPER_BUNDLE_DIR / "figure3_decoder_summary.csv"),
            "rendered_figure": str(FIGURES_DIR / "figure3_decoder_calibration_v3.svg"),
            "figure_type": "calibration_analysis",
        },
        {
            "figure_label": "Figure 3S",
            "title": "Neural comparison analysis",
            "source_table": str(PAPER_BUNDLE_DIR / "neural_comparison_summary.csv"),
            "rendered_figure": str(FIGURES_DIR / "figure_neural_comparison_v1.svg"),
            "figure_type": "neural_comparison",
        },
        {
            "figure_label": "Figure 3T",
            "title": "Neural branch diagnostics",
            "rendered_figure": str(FIGURES_DIR / "figure_neural_branch_diagnostics_v1.svg"),
            "figure_type": "neural_branch_diagnostics",
        },
        {
            "figure_label": "Figure 3U",
            "title": "Neural policy comparison",
            "rendered_figure": str(FIGURES_DIR / "figure_policy_compare_v1.svg"),
            "figure_type": "neural_policy_comparison",
        },
    ]

    for case in CASE_STUDIES:
        primary = row_by_target[case["primary_target"]]
        companions = [row_by_target[target] for target in case["companion_targets"]]
        all_targets = [case["primary_target"], *case["companion_targets"]]
        assets = [_asset_record(target) for target in all_targets]
        case_rows.append(
            {
                "figure_label": case["figure_label"],
                "case_id": case["case_id"],
                "primary_target": case["primary_target"],
                "companion_targets": ";".join(case["companion_targets"]) or "NA",
                "family": primary["family"],
                "narrative_role": case["narrative_role"],
                "overall_source": primary["overall_source"],
                "overall_mutations": primary["overall_mutations"],
                "overall_score": _to_float(primary["overall_score"]),
                "overall_mars_score": _to_float(primary["overall_mars_score"]),
                "best_learned_source": primary["best_learned_source"],
                "best_learned_mutations": primary["best_learned_mutations"],
                "best_learned_score": _to_float(primary["best_learned_score"]),
            }
        )
        figure_manifest.append(
            {
                "figure_label": case["figure_label"],
                "title": case["primary_target"],
                "figure_type": "case_study",
                "primary_target": case["primary_target"],
                "companion_targets": case["companion_targets"],
                "narrative_role": case["narrative_role"],
                "asset_paths": assets,
                "rendered_figure": str(
                    FIGURES_DIR
                    / {
                        "Figure 4": "figure4_case_1lbt_v2.svg",
                        "Figure 5": "figure5_case_tem1_v2.svg",
                        "Figure 6": "figure6_case_petase_v2.svg",
                        "Figure 7": "figure7_case_cld_v1.svg",
                    }[case["figure_label"]]
                ),
                "primary_overall_mutations": primary["overall_mutations"],
                "primary_overall_source": primary["overall_source"],
                "primary_overall_score": _to_float(primary["overall_score"]),
                "companion_count": len(companions),
            }
        )

    _write_csv(
        PAPER_BUNDLE_DIR / "figure2_benchmark_overview.csv",
        benchmark_overview,
        [
            "target",
            "family",
            "overall_source",
            "overall_mutations",
            "overall_score",
            "overall_mars_score",
            "best_learned_source",
            "best_learned_mutations",
            "best_learned_score",
            "best_learned_mars_score",
            "decoder_novel_count",
            "decoder_rejected_count",
            "accepted_homologs",
            "accepted_asr",
            "asr_prior_enabled",
            "family_prior_enabled",
            "family_dataset_id",
        ],
    )
    _write_csv(
        PAPER_BUNDLE_DIR / "figure2_family_summary.csv",
        family_summary,
        [
            "family",
            "target_count",
            "mean_overall_score",
            "mean_best_learned_score",
            "family_prior_targets",
            "asr_active_targets",
            "decoder_injected_targets",
        ],
    )
    _write_csv(
        PAPER_BUNDLE_DIR / "figure3_decoder_summary.csv",
        decoder_summary,
        [
            "target",
            "decoder_enabled",
            "decoder_injected",
            "decoder_novel_count",
            "decoder_rejected_count",
            "best_decoder_candidate",
            "best_decoder_ranking_score",
            "overall_source",
            "overall_mutations",
            "overall_score",
            "overall_mars_score",
            "best_learned_source",
            "best_learned_score",
        ],
    )
    _write_csv(
        PAPER_BUNDLE_DIR / "case_study_targets.csv",
        case_rows,
        [
            "figure_label",
            "case_id",
            "primary_target",
            "companion_targets",
            "family",
            "narrative_role",
            "overall_source",
            "overall_mutations",
            "overall_score",
            "overall_mars_score",
            "best_learned_source",
            "best_learned_mutations",
            "best_learned_score",
        ],
    )
    _write_csv(
        PAPER_BUNDLE_DIR / "asset_inventory.csv",
        asset_inventory,
        [
            "target",
            "pipeline_dir",
            "pipeline_dir_exists",
            "pipeline_summary",
            "pipeline_summary_exists",
            "viz_manifest",
            "viz_manifest_exists",
            "scene_pml",
            "scene_pml_exists",
            "palette_json",
            "palette_json_exists",
            "position_fields",
            "position_fields_exists",
            "pairwise_tensor",
            "pairwise_tensor_exists",
            "retrieval_hits",
            "retrieval_hits_exists",
            "ancestral_field",
            "ancestral_field_exists",
            "combined_candidates",
            "combined_candidates_exists",
        ],
    )
    if NEURAL_COMPARISON_CSV.exists():
        neural_rows = _read_rows(NEURAL_COMPARISON_CSV)
        _write_csv(
            PAPER_BUNDLE_DIR / "neural_comparison_summary.csv",
            neural_rows,
            list(neural_rows[0].keys()) if neural_rows else [],
        )

    with (PAPER_BUNDLE_DIR / "figure_panel_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(figure_manifest, handle, indent=2)

    summary_lines = [
        "# MARS-FIELD paper bundle v1",
        "",
        "## Included figure data",
        "",
        f"- Figure 1 spec: `{REPO_ROOT / 'docs' / 'mars_field_figure1_spec_v2.md'}`",
        f"- Figure 1 render: `{FIGURES_DIR / 'figure1_mars_field_architecture_v1.svg'}`",
        f"- Figure 1 caption: `{FIGURES_DIR / 'figure1_mars_field_architecture_v1_caption.md'}`",
        f"- Figure 2 benchmark table: `{PAPER_BUNDLE_DIR / 'figure2_benchmark_overview.csv'}`",
        f"- Figure 2 family table: `{PAPER_BUNDLE_DIR / 'figure2_family_summary.csv'}`",
        f"- Figure 2 render: `{FIGURES_DIR / 'figure2_benchmark_overview_v3.svg'}`",
        f"- Figure 3 decoder table: `{PAPER_BUNDLE_DIR / 'figure3_decoder_summary.csv'}`",
        f"- Figure 3 render: `{FIGURES_DIR / 'figure3_decoder_calibration_v3.svg'}`",
        f"- Neural comparison table: `{PAPER_BUNDLE_DIR / 'neural_comparison_summary.csv'}`",
        f"- Neural comparison render: `{FIGURES_DIR / 'figure_neural_comparison_v1.svg'}`",
        f"- Neural branch diagnostics render: `{FIGURES_DIR / 'figure_neural_branch_diagnostics_v1.svg'}`",
        f"- Neural policy comparison render: `{FIGURES_DIR / 'figure_policy_compare_v1.svg'}`",
        f"- Derived benchmark metrics: `{PAPER_BUNDLE_DIR / 'benchmark_derived_metrics.csv'}`",
        f"- Data figure summary: `{PAPER_BUNDLE_DIR / 'data_figure_summary.md'}`",
        f"- Case-study manifest: `{PAPER_BUNDLE_DIR / 'case_study_targets.csv'}`",
        f"- Asset inventory: `{PAPER_BUNDLE_DIR / 'asset_inventory.csv'}`",
        f"- Figure manifest: `{PAPER_BUNDLE_DIR / 'figure_panel_manifest.json'}`",
        "",
        "## Selected case studies",
        "",
    ]
    for row in case_rows:
        summary_lines.append(
            f"- {row['figure_label']}: `{row['primary_target']}` "
            f"({row['family']}) overall `{row['overall_mutations']}` from {row['overall_source']} "
            f"score={row['overall_score']:.6f}, mars={row['overall_mars_score']:.3f}"
        )
    summary_lines.extend(
        [
            "",
            "## Bundle scope",
            "",
            f"- targets in benchmark_twelvepack: {len(rows)}",
            f"- families represented: {len(family_summary)}",
            f"- primary case studies: {len(case_rows)}",
        ]
    )
    (PAPER_BUNDLE_DIR / "bundle_summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build_bundle()
