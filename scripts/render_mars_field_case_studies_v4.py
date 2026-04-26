from __future__ import annotations

from pathlib import Path
import json
import re
import textwrap

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PAPER = OUT / "paper_bundle_v1"
FIG = PAPER / "figures"
STRUCT = PAPER / "structure_panels"

COL = {
    "ink": "#111827",
    "muted": "#667085",
    "edge": "#d0d5dd",
    "accent1": "#1d4ed8",
    "accent2": "#0f766e",
    "accent3": "#b45309",
    "accent4": "#7c3aed",
    "white": "#ffffff",
}

CASES = [
    ("1LBT", "lipase_b", COL["accent1"], "Safety-preserving controller"),
    ("tem1_1btl", "beta_lactamase", COL["accent2"], "Stable incumbent with neural alternative"),
    ("petase_5xh3", "cutinase", COL["accent3"], "Reproducible redesign across related PETase backbones"),
    ("CLD_3Q09_TOPIC", "cld", COL["accent4"], "Calibration stress-test under topic conditioning"),
]


def style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def cl(text: str) -> str:
    return str(text).replace("_", " ")


def crop_image(path: Path):
    image = mpimg.imread(path)
    if image.ndim == 3:
        rgb = image[..., :3]
        mask = (rgb < 0.985).any(axis=2)
        coords = mask.nonzero()
        if len(coords[0]) > 0:
            y0, y1 = coords[0].min(), coords[0].max() + 1
            x0, x1 = coords[1].min(), coords[1].max() + 1
            py = max(4, int((y1 - y0) * 0.03))
            px = max(4, int((x1 - x0) * 0.03))
            image = image[max(0, y0 - py): min(image.shape[0], y1 + py), max(0, x0 - px): min(image.shape[1], x1 + px)]
    return image


def load_summary_lines(target: str) -> list[str]:
    path = OUT / f"{target.lower()}_pipeline" / "pipeline_summary.md"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def extract_text(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return "NA"


def extract_mutation_only(text: str) -> str:
    match = re.search(r"`([^`]+)`", text)
    return match.group(1) if match else text.replace("`", "")


def compact_mutations(text: str, max_len: int = 28) -> str:
    text = str(text or "NA")
    if text == "NA":
        return text
    compact = text.replace(";", "/")
    if len(compact) <= max_len:
        return compact
    parts = compact.split("/")
    kept = []
    for part in parts:
        proposal = "/".join(kept + [part])
        if len(proposal) > max_len - 1:
            break
        kept.append(part)
    if kept:
        return "/".join(kept) + "…"
    return compact[: max_len - 1] + "…"


def best_neural_decoder(target: str) -> str:
    path = OUT / f"{target.lower()}_pipeline" / "learned_fusion_summary.json"
    if not path.exists():
        return "NA"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("best_neural_decoder_candidate", "") or "NA")


def wrapped_lines(lines: list[str], width: int = 28) -> list[str]:
    out: list[str] = []
    for line in lines:
        out.extend(textwrap.wrap(line, width=width) or [""])
    return out


def small_box(ax, xy, wh, title, lines, accent):
    x, y = xy
    w, h = wh
    patch = mpl.patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=COL["edge"],
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.add_patch(mpl.patches.Rectangle((x, y + h - 0.014), w, 0.014, color=accent, transform=ax.transAxes, lw=0))
    ax.text(x + 0.02, y + h - 0.04, title, transform=ax.transAxes, fontsize=8.9, fontweight="bold", color=COL["ink"], ha="left", va="top")
    yy = y + h - 0.095
    for line in wrapped_lines(lines, width=28):
        ax.text(x + 0.02, yy, line, transform=ax.transAxes, fontsize=7.1, color=COL["muted"], ha="left", va="top")
        yy -= 0.060


def render():
    style()
    FIG.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.4, 10.4))
    gs = fig.add_gridspec(2, 2, wspace=0.16, hspace=0.16)
    letters = ["a", "b", "c", "d"]

    fig.text(0.03, 0.985, "Figure 4 | Representative case studies reveal distinct controller regimes", fontsize=16.5, fontweight="bold", ha="left", va="top", color=COL["ink"])
    fig.text(0.03, 0.962, "PSE-derived overview and design-window renders are paired with compact controller annotations for four representative targets.", fontsize=9.8, ha="left", va="top", color=COL["muted"])

    for idx, (target, family, accent, message) in enumerate(CASES):
        cell = gs[idx // 2, idx % 2].subgridspec(2, 2, height_ratios=[1.0, 0.72], width_ratios=[1.0, 1.0], wspace=0.06, hspace=0.06)
        ax_over = fig.add_subplot(cell[0, 0])
        ax_zoom = fig.add_subplot(cell[0, 1])
        ax_text = fig.add_subplot(cell[1, :])

        over = crop_image(STRUCT / target / "overview.png")
        zoom = crop_image(STRUCT / target / "design_window.png")
        ax_over.imshow(over)
        ax_zoom.imshow(zoom)
        for ax in [ax_over, ax_zoom]:
            ax.axis("off")

        ax_over.text(0.00, 1.02, letters[idx], transform=ax_over.transAxes, fontsize=11.5, fontweight="bold", color=COL["ink"], ha="left", va="bottom")
        ax_over.text(0.08, 1.02, cl(target), transform=ax_over.transAxes, fontsize=10.5, fontweight="bold", color=COL["ink"], ha="left", va="bottom")
        ax_over.text(0.08, 0.99, family, transform=ax_over.transAxes, fontsize=7.8, color=accent, ha="left", va="top")
        ax_zoom.text(0.00, 1.02, "design window", transform=ax_zoom.transAxes, fontsize=8.2, fontweight="bold", color=COL["muted"], ha="left", va="bottom")

        lines = load_summary_lines(target)
        policy = extract_mutation_only(extract_text(lines, "- policy winner"))
        best_learned = extract_mutation_only(extract_text(lines, "- best learned winner"))
        neural_top = extract_text(lines, "- neural top candidate")
        retained = extract_text(lines, "- neural field decoder novel candidates injected")
        neural_decoder = best_neural_decoder(target)

        ax_text.axis("off")
        small_box(
            ax_text,
            (0.00, 0.05),
            (0.48, 0.84),
            "Controller state",
            [
                message,
                f"policy: {compact_mutations(policy)}",
                f"best learned: {compact_mutations(best_learned)}",
            ],
            accent,
        )
        small_box(
            ax_text,
            (0.52, 0.05),
            (0.48, 0.84),
            "Neural branch",
            [
                f"neural top: {compact_mutations(neural_top)}",
                f"best decoder: {compact_mutations(neural_decoder)}",
                f"retained decoded candidates: {retained}",
            ],
            accent,
        )

    fig.savefig(FIG / "figure4_case_studies_master_v4.svg")
    fig.savefig(FIG / "figure4_case_studies_master_v4.png", dpi=300)


if __name__ == "__main__":
    render()
