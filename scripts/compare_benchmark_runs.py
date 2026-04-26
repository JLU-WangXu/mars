from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEEP_COLUMNS = [
    "target",
    "ranking_model",
    "decoder_enabled",
    "decoder_injected",
    "decoder_novel_count",
    "decoder_rejected_count",
    "selection_policy",
    "policy_resolution",
    "policy_source",
    "policy_mutations",
    "policy_selection_score",
    "policy_engineering_score",
    "neural_rerank_enabled",
    "neural_top_source",
    "neural_top_mutations",
    "neural_top_energy",
    "neural_top_mars_score",
    "overall_source",
    "overall_mutations",
    "overall_score",
    "overall_mars_score",
    "best_learned_source",
    "best_learned_mutations",
    "best_learned_score",
    "best_learned_mars_score",
]


def prepare(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    keep = [col for col in KEEP_COLUMNS if col in df.columns]
    renamed = {
        col: f"{col}_{suffix}" if col != "target" else col
        for col in keep
    }
    return df[keep].rename(columns=renamed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-a", type=Path, required=True)
    parser.add_argument("--summary-b", type=Path, required=True)
    parser.add_argument("--label-a", type=str, default="a")
    parser.add_argument("--label-b", type=str, default="b")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    df_a = pd.read_csv(args.summary_a)
    df_b = pd.read_csv(args.summary_b)

    left = prepare(df_a, args.label_a)
    right = prepare(df_b, args.label_b)
    merged = left.merge(right, on="target", how="inner")

    merged[f"overall_score_delta_{args.label_b}_minus_{args.label_a}"] = (
        merged[f"overall_score_{args.label_b}"] - merged[f"overall_score_{args.label_a}"]
    ).round(6)
    merged[f"best_learned_score_delta_{args.label_b}_minus_{args.label_a}"] = (
        merged[f"best_learned_score_{args.label_b}"] - merged[f"best_learned_score_{args.label_a}"]
    ).round(6)
    if f"policy_selection_score_{args.label_a}" in merged.columns and f"policy_selection_score_{args.label_b}" in merged.columns:
        merged[f"policy_selection_score_delta_{args.label_b}_minus_{args.label_a}"] = (
            merged[f"policy_selection_score_{args.label_b}"] - merged[f"policy_selection_score_{args.label_a}"]
        ).round(6)
        merged["policy_changed"] = merged[f"policy_mutations_{args.label_a}"] != merged[f"policy_mutations_{args.label_b}"]
    merged["overall_changed"] = merged[f"overall_mutations_{args.label_a}"] != merged[f"overall_mutations_{args.label_b}"]
    merged["best_learned_changed"] = merged[f"best_learned_mutations_{args.label_a}"] != merged[f"best_learned_mutations_{args.label_b}"]
    merged = merged.sort_values("target").reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)

    lines = [
        f"# Benchmark Run Comparison: {args.label_a} vs {args.label_b}",
        "",
        "## Targets",
        "",
    ]
    for _, row in merged.iterrows():
        policy_text = ""
        if f"policy_selection_score_{args.label_a}" in row.index and f"policy_selection_score_{args.label_b}" in row.index:
            policy_text = (
                f"policy {args.label_a}=`{row[f'policy_mutations_{args.label_a}']}` ({row[f'policy_selection_score_{args.label_a}']}) "
                f"vs {args.label_b}=`{row[f'policy_mutations_{args.label_b}']}` ({row[f'policy_selection_score_{args.label_b}']}), "
                f"policy_delta={row.get(f'policy_selection_score_delta_{args.label_b}_minus_{args.label_a}', 'NA')}; "
            )
        lines.append(
            f"- {row['target']}: overall {args.label_a}=`{row[f'overall_mutations_{args.label_a}']}` ({row[f'overall_score_{args.label_a}']}) "
            f"vs {args.label_b}=`{row[f'overall_mutations_{args.label_b}']}` ({row[f'overall_score_{args.label_b}']}), "
            f"delta={row[f'overall_score_delta_{args.label_b}_minus_{args.label_a}']}; "
            f"{policy_text}"
            f"best learned {args.label_a}=`{row[f'best_learned_mutations_{args.label_a}']}` ({row[f'best_learned_score_{args.label_a}']}) "
            f"vs {args.label_b}=`{row[f'best_learned_mutations_{args.label_b}']}` ({row[f'best_learned_score_{args.label_b}']}); "
            f"decoder {args.label_a}={'on' if row.get(f'decoder_enabled_{args.label_a}', False) else 'off'} "
            f"vs {args.label_b}={'on' if row.get(f'decoder_enabled_{args.label_b}', False) else 'off'}"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote benchmark comparison to {args.out_csv}")
    print(f"Wrote benchmark comparison markdown to {args.out_md}")


if __name__ == "__main__":
    main()
