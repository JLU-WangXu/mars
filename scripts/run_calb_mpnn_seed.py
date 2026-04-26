from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendors" / "ProteinMPNN"
DEFAULT_PDB = Path(r"D:\Codex\Work\4-12 Mars protein\designs\calb_poc\1LBT.pdb")


def run(cmd: list[str], cwd: Path) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def normalize_parsed_names(parsed_jsonl: Path) -> None:
    lines = []
    with parsed_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            obj["name"] = Path(obj["name"]).stem
            lines.append(obj)
    with parsed_jsonl.open("w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--design-positions", default="249 251 298")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "calb_mpnn_seed" / "run_01")
    parser.add_argument("--num-seq-per-target", type=int, default=24)
    parser.add_argument("--sampling-temp", default="0.1 0.2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--use-soluble-model", action="store_true", default=True)
    args = parser.parse_args()

    input_dir = args.out_dir.parent / "input_pdbs"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied_pdb = input_dir / args.pdb.name
    copied_pdb.write_bytes(args.pdb.read_bytes())

    parsed_jsonl = args.out_dir.parent / "parsed_pdbs.jsonl"
    assigned_jsonl = args.out_dir.parent / "assigned_pdbs.jsonl"
    fixed_jsonl = args.out_dir.parent / "fixed_pdbs.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "helper_scripts/parse_multiple_chains.py",
            "--input_path",
            str(input_dir),
            "--output_path",
            str(parsed_jsonl),
        ],
        cwd=VENDOR,
    )
    normalize_parsed_names(parsed_jsonl)

    run(
        [
            sys.executable,
            "helper_scripts/assign_fixed_chains.py",
            "--input_path",
            str(parsed_jsonl),
            "--output_path",
            str(assigned_jsonl),
            "--chain_list",
            args.chain,
        ],
        cwd=VENDOR,
    )

    run(
        [
            sys.executable,
            "helper_scripts/make_fixed_positions_dict.py",
            "--input_path",
            str(parsed_jsonl),
            "--output_path",
            str(fixed_jsonl),
            "--chain_list",
            args.chain,
            "--position_list",
            args.design_positions,
            "--specify_non_fixed",
        ],
        cwd=VENDOR,
    )

    cmd = [
        sys.executable,
        "protein_mpnn_run.py",
        "--path_to_model_weights",
        str(VENDOR / ("soluble_model_weights" if args.use_soluble_model else "vanilla_model_weights")),
        "--jsonl_path",
        str(parsed_jsonl),
        "--chain_id_jsonl",
        str(assigned_jsonl),
        "--fixed_positions_jsonl",
        str(fixed_jsonl),
        "--out_folder",
        str(args.out_dir),
        "--num_seq_per_target",
        str(args.num_seq_per_target),
        "--sampling_temp",
        args.sampling_temp,
        "--seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
        "--suppress_print",
        "0",
    ]
    if args.use_soluble_model:
        cmd.append("--use_soluble_model")
    run(cmd, cwd=VENDOR)

    print("DONE", args.out_dir)


if __name__ == "__main__":
    main()
