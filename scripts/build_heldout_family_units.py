from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def text_or_na(value: object) -> str:
    if pd.isna(value):
        return "NA"
    text = str(value)
    return text if text else "NA"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--family-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    families = pd.read_csv(args.family_csv)

    if "family" not in summary.columns:
        summary["family"] = summary["target"]

    rows: list[dict[str, object]] = []
    for family in sorted(summary["family"].astype(str).unique().tolist()):
        heldout_family = summary[summary["family"] == family].copy()
        context = summary[summary["family"] != family].copy()
        best_overall = heldout_family.sort_values(["overall_score", "target"], ascending=[False, True]).iloc[0]
        best_learned = heldout_family.sort_values(["best_learned_score", "target"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "heldout_family": family,
                "split_type": summarize_family_prior(heldout_family["family_prior_enabled"]),
                "heldout_targets": join_unique(heldout_family["target"]),
                "family_dataset_ids": join_unique(heldout_family["family_dataset_id"]),
                "heldout_n_targets": int(len(heldout_family)),
                "context_n_targets": int(len(context)),
                "heldout_mean_overall_score": round(float(heldout_family["overall_score"].mean()), 3),
                "heldout_mean_best_learned_score": round(float(heldout_family["best_learned_score"].mean()), 3),
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
    heldout = pd.DataFrame(rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(args.out_csv, index=False)

    lines = [
        "# Held-Out Family Units",
        "",
        "Current benchmark interpretation:",
        "",
        "- each explicit family label is treated as one held-out evaluation unit",
        "- targets sharing a family label are aggregated before the held-out vs context comparison",
        "- family-level means are stored separately in `family_summary.csv`",
        "",
        "## Held-Out Families",
        "",
    ]
    for _, row in heldout.iterrows():
        lines.append(
            f"- {row['heldout_family']} [{row['split_type']}]: held-out targets `{row['heldout_targets']}`; "
            f"held-out mean overall={format_score(row['heldout_mean_overall_score'])}, "
            f"held-out mean best learned={format_score(row['heldout_mean_best_learned_score'])}; "
            f"context mean overall={format_score(row['context_mean_overall_score'])}, "
            f"context mean best learned={format_score(row['context_mean_best_learned_score'])}; "
            f"best overall `{row['best_heldout_overall_mutations']}` ({row['best_heldout_overall_source']}, {format_score(row['best_heldout_overall_score'])}); "
            f"best learned `{row['best_heldout_learned_mutations']}` ({row['best_heldout_learned_source']}, {format_score(row['best_heldout_learned_score'])}); "
            f"family prior `{text_or_na(row['family_dataset_ids'])}`"
        )
    lines.extend(["", "## Family Means", ""])
    for _, row in families.sort_values("family").iterrows():
        lines.append(
            f"- {row['family']}: n={row['n_targets']}, mean overall={format_score(row['mean_overall_score'])}, mean best learned={format_score(row['mean_best_learned_score'])}"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote held-out family units to {args.out_csv}")
    print(f"Wrote held-out family markdown to {args.out_md}")


if __name__ == "__main__":
    main()
