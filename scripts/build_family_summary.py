from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_family_map(targets: list[Path]) -> dict[str, str]:
    family_map: dict[str, str] = {}
    for config_path in targets:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        protein = cfg["protein"]["name"]
        family = cfg.get("benchmark", {}).get("family", protein)
        family_map[protein] = family
    return family_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--outpath", type=Path, required=True)
    args = parser.parse_args()

    bench_cfg = yaml.safe_load(args.benchmark_config.read_text(encoding="utf-8"))
    target_configs = [Path(p) if Path(p).is_absolute() else (ROOT / p) for p in bench_cfg["targets"]]
    family_map = load_family_map(target_configs)

    df = pd.read_csv(args.summary_csv)
    df["family"] = df["target"].map(family_map)

    family_df = (
        df.groupby("family", as_index=False)
        .agg(
            n_targets=("target", "count"),
            mean_overall_score=("overall_score", "mean"),
            mean_best_learned_score=("best_learned_score", "mean"),
        )
        .sort_values("family")
        .reset_index(drop=True)
    )
    family_df.to_csv(args.outpath, index=False)
    print(f"Wrote family summary to {args.outpath}")


if __name__ == "__main__":
    main()
