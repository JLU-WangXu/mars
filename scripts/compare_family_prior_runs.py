from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-summary", type=Path, required=True)
    parser.add_argument("--without-summary", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    with_df = pd.read_csv(args.with_summary)
    without_df = pd.read_csv(args.without_summary)

    keep_cols = [
        "target",
        "overall_source",
        "overall_mutations",
        "overall_score",
        "best_learned_source",
        "best_learned_mutations",
        "best_learned_score",
        "family_prior_enabled",
        "family_dataset_id",
        "accepted_positive",
        "accepted_negative",
    ]
    with_df = with_df[keep_cols].rename(columns={col: f"{col}_with" if col != "target" else col for col in keep_cols})
    without_df = without_df[keep_cols].rename(columns={col: f"{col}_without" if col != "target" else col for col in keep_cols})

    merged = with_df.merge(without_df, on="target", how="inner")
    merged["overall_score_delta"] = (merged["overall_score_with"] - merged["overall_score_without"]).round(3)
    merged["best_learned_score_delta"] = (merged["best_learned_score_with"] - merged["best_learned_score_without"]).round(3)
    merged["overall_changed"] = merged["overall_mutations_with"] != merged["overall_mutations_without"]
    merged["best_learned_changed"] = merged["best_learned_mutations_with"] != merged["best_learned_mutations_without"]
    merged = merged.sort_values("target").reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)

    lines = [
        "# Family Prior Comparison",
        "",
        "## Targets",
        "",
    ]
    for _, row in merged.iterrows():
        lines.append(
            f"- {row['target']}: overall with `{row['overall_mutations_with']}` ({row['overall_score_with']}) vs without `{row['overall_mutations_without']}` ({row['overall_score_without']}), "
            f"delta={row['overall_score_delta']}; best learned with `{row['best_learned_mutations_with']}` ({row['best_learned_source_with']}, {row['best_learned_score_with']}) "
            f"vs without `{row['best_learned_mutations_without']}` ({row['best_learned_source_without']}, {row['best_learned_score_without']}), "
            f"delta={row['best_learned_score_delta']}"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote family prior comparison to {args.out_csv}")
    print(f"Wrote family prior comparison markdown to {args.out_md}")


if __name__ == "__main__":
    main()
