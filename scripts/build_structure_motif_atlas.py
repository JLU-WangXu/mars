from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marsstack.retrieval_memory import (
    build_structure_memory_bank,
    build_structure_motif_atlas,
    serialize_motif_atlas,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-radius", type=float, default=0.42)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--out-json", type=Path, default=ROOT / "outputs" / "structure_motif_atlas_v1.json")
    parser.add_argument("--out-summary", type=Path, default=ROOT / "outputs" / "structure_motif_atlas_v1_summary.json")
    args = parser.parse_args()

    entries = build_structure_memory_bank(ROOT)
    atlas = build_structure_motif_atlas(
        entries=entries,
        cluster_radius=float(args.cluster_radius),
        min_cluster_size=int(args.min_cluster_size),
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(serialize_motif_atlas(atlas), indent=2),
        encoding="utf-8",
    )

    summary = {
        "entry_count": len(entries),
        "prototype_count": len(atlas),
        "cluster_radius": float(args.cluster_radius),
        "min_cluster_size": int(args.min_cluster_size),
        "max_support_count": max((item.support_count for item in atlas), default=0),
        "mean_support_count": round(sum(item.support_count for item in atlas) / max(1, len(atlas)), 3),
    }
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote motif atlas to {args.out_json}")
    print(f"Wrote motif atlas summary to {args.out_summary}")


if __name__ == "__main__":
    main()
