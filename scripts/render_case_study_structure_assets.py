from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_BUNDLE_DIR = ROOT / "outputs" / "paper_bundle_v1"
CASE_CSV = PAPER_BUNDLE_DIR / "case_study_targets.csv"
STRUCTURE_DIR = PAPER_BUNDLE_DIR / "structure_panels"


PYMOL_COLORS = {
    "mf_backbone": (0.15, 0.20, 0.28),
    "mf_backbone_soft": (0.57, 0.63, 0.71),
    "mf_context_line": (0.72, 0.78, 0.85),
    "mf_mut": (0.86, 0.36, 0.19),
    "mf_mut_alt": (0.96, 0.66, 0.18),
    "mf_design": (0.95, 0.74, 0.22),
    "mf_protected": (0.04, 0.50, 0.53),
    "mf_context": (0.25, 0.58, 0.84),
    "mf_label": (0.10, 0.14, 0.21),
    "mf_surface": (0.80, 0.88, 0.96),
}


def read_case_rows() -> list[dict[str, str]]:
    with CASE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def target_pipeline_dir(target: str) -> Path:
    return ROOT / "outputs" / f"{target.lower()}_pipeline"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mutation_positions(mutation_text: str) -> list[int]:
    if not mutation_text or mutation_text == "WT" or mutation_text == "NA":
        return []
    positions = []
    for part in mutation_text.split(";"):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            positions.append(int(digits))
    return positions


def path_posix(path: Path) -> str:
    return path.resolve().as_posix()


def selection_from_positions(object_name: str, chain_id: str, positions: list[int]) -> str:
    if not positions:
        return "none"
    joined = "+".join(str(p) for p in positions)
    return f"{object_name} and chain {chain_id} and resi {joined}"


def define_colors() -> list[str]:
    lines = []
    for name, rgb in PYMOL_COLORS.items():
        values = ", ".join(f"{channel:.3f}" for channel in rgb)
        lines.append(f"set_color {name}, [{values}]")
    return lines


def build_render_pml(
    pdb_path: Path,
    chain_id: str,
    protected_positions: list[int],
    design_positions: list[int],
    overall_mutations: str,
    best_learned_mutations: str,
    out_dir: Path,
    target_label: str,
) -> str:
    overview_png = path_posix(out_dir / "overview.png")
    window_png = path_posix(out_dir / "design_window.png")
    session_pse = path_posix(out_dir / "figure_session.pse")

    design_sel = selection_from_positions("target", chain_id, design_positions)
    protected_sel = selection_from_positions("target", chain_id, protected_positions)
    overall_pos = mutation_positions(overall_mutations)
    learned_pos = mutation_positions(best_learned_mutations)
    overall_sel = selection_from_positions("target", chain_id, overall_pos)
    learned_sel = selection_from_positions("target", chain_id, learned_pos)

    label_text = target_label.replace('"', "'")

    lines = [
        "reinitialize",
        *define_colors(),
        f"load {path_posix(pdb_path)}, target",
        f"remove not (target and chain {chain_id})",
        "remove solvent",
        "hide everything, all",
        "bg_color white",
        "set ray_opaque_background, off",
        "set depth_cue, 0",
        "set orthoscopic, on",
        "set antialias, 2",
        "set specular, 0.15",
        "set ambient, 0.50",
        "set direct, 0.18",
        "set ray_trace_mode, 1",
        "set ray_trace_gain, 0.08",
        "set cartoon_fancy_helices, 1",
        "set cartoon_smooth_loops, 1",
        "set cartoon_transparency, 0.00",
        "set stick_radius, 0.16",
        "set sphere_scale, 0.30",
        "set dash_gap, 0.28",
        "set label_font_id, 7",
        "set label_size, -0.45",
        "set label_color, mf_label",
        "show cartoon, target",
        "color mf_backbone, target",
        f"select design_resi, {design_sel}",
        f"select protected_resi, {protected_sel}",
        f"select overall_resi, {overall_sel}",
        f"select learned_resi, {learned_sel}",
        "show lines, protected_resi",
        "color mf_protected, protected_resi",
        "show spheres, design_resi",
        "color mf_design, design_resi",
        "show sticks, overall_resi",
        "show spheres, overall_resi",
        "color mf_mut, overall_resi",
        "show sticks, learned_resi and not overall_resi",
        "color mf_mut_alt, learned_resi and not overall_resi",
        "select design_window, byres (target within 8 of design_resi)",
        "show cartoon, design_window",
        "show lines, design_window and not (overall_resi or learned_resi or protected_resi or design_resi)",
        "color mf_context_line, design_window and not (overall_resi or learned_resi or protected_resi or design_resi)",
        "color mf_backbone_soft, design_window and not (overall_resi or learned_resi or protected_resi)",
        "set cartoon_transparency, 0.10, design_window",
        "set sphere_transparency, 0.08, overall_resi",
        "set sphere_transparency, 0.20, design_resi and not overall_resi",
        "label overall_resi and name CA, resn + resi",
        "disable learned_resi",
        f"pseudoatom title_anchor, pos=[0,0,0], label=\"{label_text}\"",
        "hide everything, title_anchor",
        "delete title_anchor",
        "",
        "# overview scene",
        "hide labels, all",
        "hide lines, design_window and not (overall_resi or protected_resi)",
        "hide surface, all",
        "hide spheres, design_resi and not overall_resi",
        "set sphere_scale, 0.26, overall_resi",
        "set stick_radius, 0.14, overall_resi",
        "set line_width, 1.6, protected_resi",
        "orient target",
        "turn y, -10",
        "turn x, 8",
        "zoom target, 10",
        "scene overview, store",
        f"png {overview_png}, width=2400, height=1800, dpi=300, ray=1",
        "",
        "# design window scene",
        "disable target",
        "enable target",
        "show cartoon, target",
        "hide labels, all",
        "label overall_resi and name CA, resn + resi",
        "show spheres, design_resi",
        "show sticks, overall_resi",
        "show lines, design_window and not (overall_resi or learned_resi or protected_resi or design_resi)",
        "show lines, protected_resi",
        "orient design_window",
        "zoom design_window, 6",
        "turn y, -12",
        "turn x, 8",
        "show surface, design_window",
        "set transparency, 0.90, design_window",
        "color mf_surface, design_window",
        "scene design_window, store",
        f"png {window_png}, width=2400, height=1800, dpi=300, ray=1",
        "",
        f"save {session_pse}",
        "quit",
    ]
    return "\n".join(lines) + "\n"


def render_target(target: str, overall_mutations: str, best_learned_mutations: str, render: bool) -> dict[str, object]:
    pipeline_dir = target_pipeline_dir(target)
    viz_manifest_path = pipeline_dir / "viz_bundle" / "viz_manifest.json"
    viz_manifest = read_json(viz_manifest_path)
    out_dir = STRUCTURE_DIR / target
    out_dir.mkdir(parents=True, exist_ok=True)

    pml_path = out_dir / "render_scene.pml"
    pml_path.write_text(
        build_render_pml(
            pdb_path=Path(viz_manifest["pdb_path"]),
            chain_id=str(viz_manifest["chain_id"]),
            protected_positions=[int(x) for x in viz_manifest.get("protected_positions", [])],
            design_positions=[int(x) for x in viz_manifest.get("design_positions", [])],
            overall_mutations=overall_mutations,
            best_learned_mutations=best_learned_mutations,
            out_dir=out_dir,
            target_label=target,
        ),
        encoding="utf-8",
    )

    rendered = False
    if render:
        subprocess.run(
            ["cmd", "/c", "pymol", "-cq", str(pml_path)],
            cwd=str(ROOT),
            check=True,
        )
        rendered = True

    manifest = {
        "target": target,
        "pipeline_dir": str(pipeline_dir),
        "viz_manifest": str(viz_manifest_path),
        "render_scene_pml": str(pml_path),
        "overview_png": str(out_dir / "overview.png"),
        "design_window_png": str(out_dir / "design_window.png"),
        "figure_session_pse": str(out_dir / "figure_session.pse"),
        "overall_mutations": overall_mutations,
        "best_learned_mutations": best_learned_mutations,
        "rendered": rendered,
    }
    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--include-companions", action="store_true")
    args = parser.parse_args()

    rows = read_case_rows()
    rendered = []
    seen = set()
    for row in rows:
        targets = [row["primary_target"]]
        if args.include_companions and row["companion_targets"] and row["companion_targets"] != "NA":
            targets.extend(part.strip() for part in row["companion_targets"].split(";") if part.strip())
        for target in targets:
            if target in seen:
                continue
            seen.add(target)
            overall_mut = row["overall_mutations"] if target == row["primary_target"] else row["best_learned_mutations"]
            best_learned_mut = row["best_learned_mutations"]
            rendered.append(render_target(target, overall_mut, best_learned_mut, args.render))

    summary = {
        "target_count": len(rendered),
        "rendered": args.render,
        "targets": rendered,
    }
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    (STRUCTURE_DIR / "structure_panel_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
