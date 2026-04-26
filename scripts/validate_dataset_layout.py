from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"

BENCHMARK_COLUMNS = {
    "target_id",
    "family",
    "target_name",
    "role",
    "status",
    "freeze_validation_tier",
    "assay",
    "meta_path",
    "wt_fasta_path",
    "structure_path",
    "homologs_path",
    "notes",
}

EVOLUTION_COLUMNS = {
    "dataset_id",
    "family",
    "adaptation_axis",
    "record_type",
    "status",
    "positive_group",
    "negative_group",
    "manifest_path",
    "notes",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def check_columns(path: Path, expected: set[str]) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        found = set(reader.fieldnames or [])
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    msgs = []
    if missing:
        msgs.append(f"{path.name}: missing columns: {', '.join(missing)}")
    if extra:
        msgs.append(f"{path.name}: extra columns: {', '.join(extra)}")
    return msgs


def resolve_dataset_path(rel_path: str) -> Path:
    return (DATASETS / rel_path).resolve()


def validate_benchmark(strict_ready: bool) -> list[str]:
    path = DATASETS / "benchmark_pool" / "index.csv"
    msgs = check_columns(path, BENCHMARK_COLUMNS)
    rows = read_csv_rows(path)
    for row in rows:
        status = row["status"].strip().lower()
        required = ["meta_path", "wt_fasta_path", "structure_path"] if status == "ready" else []
        optional = ["meta_path", "wt_fasta_path", "structure_path", "homologs_path"]
        for key in optional:
            rel = row[key].strip()
            if not rel:
                if key in required and strict_ready:
                    msgs.append(f"benchmark_pool {row['target_id']}: missing required {key}")
                continue
            if not resolve_dataset_path(rel).exists():
                msgs.append(f"benchmark_pool {row['target_id']}: path not found for {key}: {rel}")
    return msgs


def validate_evolution(strict_ready: bool) -> list[str]:
    path = DATASETS / "evolution_train" / "index.csv"
    msgs = check_columns(path, EVOLUTION_COLUMNS)
    rows = read_csv_rows(path)
    for row in rows:
        status = row["status"].strip().lower()
        rel = row["manifest_path"].strip()
        if not rel:
            if status == "ready" and strict_ready:
                msgs.append(f"evolution_train {row['dataset_id']}: missing manifest_path")
            continue
        if status == "ready" and not resolve_dataset_path(rel).exists():
            msgs.append(f"evolution_train {row['dataset_id']}: manifest not found: {rel}")
    return msgs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-ready", action="store_true", help="fail if ready entries are missing required files")
    args = parser.parse_args()

    msgs = []
    msgs.extend(validate_benchmark(strict_ready=args.strict_ready))
    msgs.extend(validate_evolution(strict_ready=args.strict_ready))

    if msgs:
        print("Dataset layout issues:")
        for msg in msgs:
            print(f"- {msg}")
        return 1

    print("Dataset layout looks consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
