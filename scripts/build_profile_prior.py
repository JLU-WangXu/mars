from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marsstack.evolution import build_profile_from_homologs, load_fasta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wt-fasta", type=Path, required=True)
    parser.add_argument("--homologs-fasta", type=Path, required=True)
    parser.add_argument("--aligned-out", type=Path, required=True)
    parser.add_argument("--profile-out", type=Path, required=True)
    parser.add_argument("--min-identity", type=float, default=0.20)
    args = parser.parse_args()

    wt_entries = load_fasta(args.wt_fasta)
    if not wt_entries:
        raise ValueError(f"No WT sequence found in {args.wt_fasta}")
    wt_seq = wt_entries[0][1]
    homolog_entries = load_fasta(args.homologs_fasta)

    aligned_entries, profile = build_profile_from_homologs(
        wt_seq=wt_seq,
        homolog_entries=homolog_entries,
        aligned_out=args.aligned_out,
        min_identity=args.min_identity,
    )

    coverage = [
        sum(1 for _, seq in aligned_entries[1:] if seq[i] != "-")
        for i in range(len(wt_seq))
    ]
    payload = {
        "wt_length": len(wt_seq),
        "input_homologs": len(homolog_entries),
        "accepted_homologs": max(0, len(aligned_entries) - 1),
        "mean_coverage": round(sum(coverage) / max(1, len(coverage)), 3),
        "profile": profile,
    }
    args.profile_out.parent.mkdir(parents=True, exist_ok=True)
    args.profile_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote aligned homologs to {args.aligned_out}")
    print(f"Wrote profile summary to {args.profile_out}")


if __name__ == "__main__":
    main()
