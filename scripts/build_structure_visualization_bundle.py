from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

PALETTES = {
    "paper_ember": {
        "protein": "slate",
        "mut_hot": "tv_red",
        "mut_warm": "orange",
        "mut_cool": "marine",
        "support": "cyan",
        "design": "yelloworange",
        "protected": "deepteal",
        "surface": "lightorange",
        "background": "white",
    },
    "paper_lagoon": {
        "protein": "gray70",
        "mut_hot": "raspberry",
        "mut_warm": "brightorange",
        "mut_cool": "aquamarine",
        "support": "deepsalmon",
        "design": "tv_blue",
        "protected": "forest",
        "surface": "palecyan",
        "background": "white",
    },
}


def mutation_positions(mutation_text: str) -> list[int]:
    if mutation_text == "WT":
        return []
    positions = []
    for part in mutation_text.split(";"):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            positions.append(int(digits))
    return positions


def pick_rows(df: pd.DataFrame, top_n: int) -> list[dict[str, object]]:
    selected = []
    if not df.empty:
        selected.append(df.iloc[0].to_dict())
    learned = df[df["source_group"] == "learned"]
    if not learned.empty:
        selected.append(learned.iloc[0].to_dict())
    selected.extend(df.head(max(0, top_n - len(selected))).to_dict(orient="records"))

    deduped = []
    seen = set()
    for row in selected:
        key = str(row["candidate_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= top_n:
            break
    return deduped


def build_pymol_script(
    pdb_path: Path,
    chain_id: str,
    rows: list[dict[str, object]],
    protected_positions: list[int],
    design_positions: list[int],
    palette: dict[str, str],
) -> str:
    lines = [
        "reinitialize",
        f"load {pdb_path.as_posix()}, target",
        f"bg_color {palette['background']}",
        "hide everything, all",
        "show cartoon, target",
        f"color {palette['protein']}, target",
        "set cartoon_transparency, 0.10",
        f"select protected_resi, target and chain {chain_id} and resi {'+'.join(str(x) for x in protected_positions)}" if protected_positions else "select protected_resi, none",
        f"select design_resi, target and chain {chain_id} and resi {'+'.join(str(x) for x in design_positions)}" if design_positions else "select design_resi, none",
        "show sticks, protected_resi",
        f"color {palette['protected']}, protected_resi",
        "show spheres, design_resi",
        "set sphere_scale, 0.28, design_resi",
        f"color {palette['design']}, design_resi",
        "",
    ]

    for idx, row in enumerate(rows, start=1):
        candidate_name = f"cand_{idx:02d}"
        mut_positions = mutation_positions(str(row["mutations"]))
        selection = f"target and chain {chain_id} and resi {'+'.join(str(x) for x in mut_positions)}" if mut_positions else "none"
        lines.extend(
            [
                f"# {row['candidate_id']} | {row['mutations']} | {row['source']}",
                f"select {candidate_name}, {selection}",
                f"show sticks, {candidate_name}",
                f"color {palette['mut_hot'] if row['source'] == 'fusion_decoder' else palette['mut_warm'] if row['source_group'] == 'learned' else palette['mut_cool']}, {candidate_name}",
                f"label {candidate_name} and name CA, \"{row['candidate_id']}:{row['mutations']}\"",
                "",
            ]
        )

    lines.extend(
        [
            "set ray_trace_mode, 1",
            "set antialias, 2",
            "set depth_cue, 0",
            "set orthoscopic, on",
            "orient target",
            "zoom target, 12",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--palette", type=str, default="paper_ember", choices=sorted(PALETTES))
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else (ROOT / args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protein = cfg["protein"]
    chain_id = str(protein["chain"])
    pdb_path = Path(protein["pdb_path"])
    if not pdb_path.is_absolute():
        pdb_path = (ROOT / pdb_path).resolve()

    target_name = str(protein["name"]).lower()
    pipeline_dir = ROOT / "outputs" / f"{target_name}_pipeline"
    ranked = pd.read_csv(pipeline_dir / "combined_ranked_candidates.csv")
    rows = pick_rows(ranked, top_n=int(args.top_n))

    viz_dir = pipeline_dir / "viz_bundle"
    viz_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "target": protein["name"],
        "pdb_path": str(pdb_path),
        "chain_id": chain_id,
        "palette": args.palette,
        "protected_positions": protein.get("protected_positions", []),
        "design_positions": cfg["generation"].get("design_positions", []),
        "candidates": rows,
    }
    (viz_dir / "viz_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (viz_dir / "palette.json").write_text(json.dumps(PALETTES[args.palette], indent=2), encoding="utf-8")
    (viz_dir / "scene.pml").write_text(
        build_pymol_script(
            pdb_path=pdb_path,
            chain_id=chain_id,
            rows=rows,
            protected_positions=[int(x) for x in protein.get("protected_positions", [])],
            design_positions=[int(x) for x in cfg["generation"].get("design_positions", [])],
            palette=PALETTES[args.palette],
        ),
        encoding="utf-8",
    )

    print(f"Wrote visualization bundle to {viz_dir}")


if __name__ == "__main__":
    main()
