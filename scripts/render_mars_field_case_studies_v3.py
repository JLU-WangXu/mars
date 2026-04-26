from __future__ import annotations

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PAPER = OUT / "paper_bundle_v1"
FIG = PAPER / "figures"
STRUCT = PAPER / "structure_panels"

COL = {
    "ink": "#111827",
    "muted": "#667085",
    "grid": "#e5e7eb",
    "box": "#d0d5dd",
    "accent1": "#1d4ed8",
    "accent2": "#0f766e",
    "accent3": "#b45309",
    "accent4": "#7c3aed",
    "good": "#0f766e",
    "warn": "#b42318",
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
            "font.size": 9.5,
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


def load_pipeline_summary(target: str) -> list[str]:
    path = OUT / f"{target.lower()}_pipeline" / "pipeline_summary.md"
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def extract_line(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return "NA"


def top_candidates(target: str, n: int = 3) -> pd.DataFrame:
    path = OUT / f"{target.lower()}_pipeline" / "combined_ranked_candidates.csv"
    df = pd.read_csv(path)
    cols = ["source", "mutations", "ranking_score", "mars_score"]
    return df[cols].head(n).copy()


def best_neural_decoder(target: str) -> str:
    path = OUT / f"{target.lower()}_pipeline" / "learned_fusion_summary.json"
    if not path.exists():
        return "NA"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("best_neural_decoder_candidate", "") or "NA")


def text_box(ax, xy, wh, title, lines, edge, accent):
    x, y = xy
    w, h = wh
    patch = mpl.patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.9,
        edgecolor=edge,
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.add_patch(mpl.patches.Rectangle((x, y + h - 0.018), w, 0.018, color=accent, transform=ax.transAxes, lw=0))
    ax.text(x + 0.02, y + h - 0.05, title, transform=ax.transAxes, fontsize=9.8, fontweight="bold", color=COL["ink"], ha="left", va="top")
    yy = y + h - 0.10
    for line in lines:
        ax.text(x + 0.022, yy, line, transform=ax.transAxes, fontsize=7.8, color=COL["muted"], ha="left", va="top")
        yy -= 0.04


def render():
    style()
    FIG.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.6, 11.2))
    gs = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.20)

    letters = ["a", "b", "c", "d"]
    for idx, (target, family, accent, claim) in enumerate(CASES):
        sub = gs[idx // 2, idx % 2].subgridspec(2, 2, height_ratios=[1.0, 0.95], width_ratios=[1.0, 1.0], wspace=0.08, hspace=0.10)
        ax_over = fig.add_subplot(sub[0, 0])
        ax_zoom = fig.add_subplot(sub[0, 1])
        ax_text = fig.add_subplot(sub[1, :])

        over_path = STRUCT / target / "overview.png"
        zoom_path = STRUCT / target / "design_window.png"
        ax_over.imshow(crop_image(over_path))
        ax_zoom.imshow(crop_image(zoom_path))
        ax_over.axis("off")
        ax_zoom.axis("off")

        ax_over.text(0.00, 1.03, letters[idx], transform=ax_over.transAxes, fontsize=12, fontweight="bold", ha="left", va="bottom", color=COL["ink"])
        ax_over.text(0.08, 1.03, f"{cl(target)}", transform=ax_over.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom", color=COL["ink"])
        ax_over.text(0.08, 0.99, family, transform=ax_over.transAxes, fontsize=8.4, ha="left", va="top", color=accent)
        ax_zoom.text(0.00, 1.03, "design window", transform=ax_zoom.transAxes, fontsize=8.6, fontweight="bold", ha="left", va="bottom", color=COL["muted"])

        summary = load_pipeline_summary(target)
        policy = extract_line(summary, "- policy winner")
        best_overall = extract_line(summary, "- best overall winner")
        best_learned = extract_line(summary, "- best learned winner")
        neural_top = extract_line(summary, "- neural top candidate")
        neural_decoder = best_neural_decoder(target)
        decoder_novel = extract_line(summary, "- neural field decoder novel candidates injected")

        ax_text.axis("off")
        text_box(
            ax_text,
            (0.00, 0.02),
            (0.48, 0.88),
            "Case-specific message",
            [
                claim,
                f"policy: {policy}",
                f"overall: {best_overall}",
                f"best learned: {best_learned}",
            ],
            COL["box"],
            accent,
        )
        text_box(
            ax_text,
            (0.52, 0.02),
            (0.48, 0.88),
            "Neural branch readout",
            [
                f"neural top: {neural_top}",
                f"best neural decoder: {neural_decoder}",
                f"retained neural-decoder candidates: {decoder_novel}",
                "structural panel rendered from PSE-derived overview and close-up images",
            ],
            COL["box"],
            accent,
        )

    fig.suptitle("Figure 4 | Representative case studies reveal distinct controller regimes", x=0.03, y=0.995, ha="left", fontsize=17, fontweight="bold", color=COL["ink"])
    fig.text(
        0.03,
        0.972,
        "PSE-derived overview and design-window panels are paired with concise controller-level annotations rather than crowded dashboard subplots.",
        ha="left",
        fontsize=10.1,
        color=COL["muted"],
    )

    fig.savefig(FIG / "figure4_case_studies_master_v3.svg")
    fig.savefig(FIG / "figure4_case_studies_master_v3.png", dpi=300)


if __name__ == "__main__":
    render()
