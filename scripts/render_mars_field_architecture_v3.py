from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "paper_bundle_v1" / "figures"

COL = {
    "ink": "#111827",
    "muted": "#667085",
    "edge": "#d0d5dd",
    "geom": "#3b82f6",
    "phylo": "#22c55e",
    "asr": "#14b8a6",
    "retr": "#f59e0b",
    "env": "#f97316",
    "field": "#7c3aed",
    "decoder": "#db2777",
    "selector": "#334155",
    "soft": "#f8fafc",
}


def style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def card(ax, x, y, w, h, title, lines, face, edge, accent, title_size=12.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.add_patch(Rectangle((x, y + h - 0.014), w, 0.014, color=accent, lw=0))
    ax.text(x + 0.018, y + h - 0.040, title, fontsize=title_size, fontweight="bold", color=COL["ink"], ha="left", va="top")
    yy = y + h - 0.082
    for line in lines:
        ax.text(x + 0.020, yy, line, fontsize=8.2, color=COL["muted"], ha="left", va="top")
        yy -= 0.030


def arrow(ax, x1, y1, x2, y2, color, lw=2.0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            alpha=0.9,
        )
    )


def mini_heatmap(ax, x, y, w, h):
    rng = np.random.default_rng(4)
    mat = rng.random((5, 10))
    cmap = mpl.colormaps["viridis"]
    cell_w = w / 10
    cell_h = h / 5
    for i in range(5):
        for j in range(10):
            ax.add_patch(
                Rectangle(
                    (x + j * cell_w, y + (4 - i) * cell_h),
                    cell_w * 0.95,
                    cell_h * 0.92,
                    color=cmap(0.15 + 0.75 * mat[i, j]),
                    lw=0,
                )
            )


def pair_panel(ax, x, y, w, h):
    pts = [
        (x + 0.18 * w, y + 0.66 * h),
        (x + 0.44 * w, y + 0.82 * h),
        (x + 0.70 * w, y + 0.62 * h),
        (x + 0.56 * w, y + 0.28 * h),
    ]
    for a, b in [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color="#c084fc", lw=2.1, alpha=0.8)
    for i, (px, py) in enumerate(pts, start=1):
        ax.scatter([px], [py], s=52, color="white", edgecolor=COL["field"], linewidth=1.6, zorder=3)
        ax.text(px, py - 0.035 * h, f"i{i}", fontsize=7.5, ha="center", color=COL["muted"])


def render():
    style()
    FIG.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.2, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.035, 0.97, "Figure 1", fontsize=10, color=COL["muted"], ha="left", va="top")
    ax.text(0.035, 0.935, "MARS-FIELD integrates heterogeneous evidence into a shared residue field", fontsize=16.5, fontweight="bold", color=COL["ink"], ha="left", va="top")
    ax.text(0.035, 0.903, "The method operates on a shared residue-level decision space rather than on a generator vote.", fontsize=10.0, color=COL["muted"], ha="left", va="top")

    # panel labels
    ax.text(0.035, 0.83, "a", fontsize=12, fontweight="bold", color=COL["ink"])
    ax.text(0.36, 0.83, "b", fontsize=12, fontweight="bold", color=COL["ink"])
    ax.text(0.76, 0.83, "c", fontsize=12, fontweight="bold", color=COL["ink"])
    ax.text(0.36, 0.23, "d", fontsize=12, fontweight="bold", color=COL["ink"])

    # left cards
    ys = [0.70, 0.57, 0.44, 0.31, 0.18]
    specs = [
        ("Geometric encoder", ["backbone graph", "local geometry", "design/protected masks"], "#eff6ff", "#bfdbfe", COL["geom"]),
        ("Phylo-sequence encoder", ["homolog profile", "family differential prior", "template-aware weighting"], "#f0fdf4", "#bbf7d0", COL["phylo"]),
        ("Ancestral lineage encoder", ["ASR posterior", "posterior entropy", "lineage confidence"], "#ecfeff", "#99f6e4", COL["asr"]),
        ("Retrieval memory encoder", ["motif atlas", "prototype memory", "structural motif support"], "#fffbeb", "#fde68a", COL["retr"]),
        ("Environment hypernetwork", ["oxidation pressure", "flexible surface burden", "engineering context"], "#fff7ed", "#fdba74", COL["env"]),
    ]
    for y, (title, lines, face, edge, accent) in zip(ys, specs):
        card(ax, 0.04, y, 0.24, 0.095, title, lines, face, edge, accent)

    # center field region
    field_box = FancyBboxPatch(
        (0.36, 0.28), 0.32, 0.49,
        boxstyle="round,pad=0.016,rounding_size=0.03",
        linewidth=1.0,
        edgecolor="#d8b4fe",
        facecolor="#faf5ff",
    )
    ax.add_patch(field_box)
    ax.text(0.38, 0.73, "Shared residue field", fontsize=14.2, fontweight="bold", color=COL["ink"], ha="left")
    ax.text(0.38, 0.705, "Site-wise residue energies and pairwise couplings define the decision manifold.", fontsize=8.8, color=COL["muted"], ha="left")

    eq_box = FancyBboxPatch(
        (0.39, 0.61), 0.26, 0.08,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        edgecolor="#e9d5ff",
        facecolor="white",
    )
    ax.add_patch(eq_box)
    ax.text(0.52, 0.65, "E(x) = Σ_i U(i, x_i) + Σ_(i,j) C(i, j, x_i, x_j)", fontsize=12.2, fontweight="bold", color=COL["field"], ha="center", va="center")

    ax.text(0.40, 0.57, "U(i, a)", fontsize=9.5, fontweight="bold", color=COL["ink"], ha="left")
    ax.text(0.53, 0.57, "C(i, j, a, b)", fontsize=9.5, fontweight="bold", color=COL["ink"], ha="left")
    mini_heatmap(ax, 0.40, 0.38, 0.15, 0.13)
    pair_panel(ax, 0.53, 0.35, 0.11, 0.18)
    ax.text(0.405, 0.34, "design position × residue", fontsize=7.6, color=COL["muted"], ha="left")
    ax.text(0.53, 0.34, "pairwise compatibility", fontsize=7.6, color=COL["muted"], ha="left")

    # bottom field inset
    ax.text(0.38, 0.215, "Example target-level field instantiation", fontsize=10.3, fontweight="bold", color=COL["ink"], ha="left")
    ax.text(0.38, 0.188, "1LBT residue preferences at 249/251/298 and one pairwise edge.", fontsize=8.4, color=COL["muted"], ha="left")
    x0, y0 = 0.40, 0.07
    positions = ["249", "251", "298"]
    residues = [["Q", "R", "L"], ["E", "S", "A"], ["L", "I", "V"]]
    vals = [[1.96, 1.19, 0.91], [1.77, 1.30, 1.07], [2.36, 1.13, 0.48]]
    cmap = mpl.colormaps["viridis"]
    for i, pos in enumerate(positions):
        ax.text(x0 + i * 0.085, y0 + 0.09, pos, fontsize=8, fontweight="bold", color=COL["muted"], ha="center")
        for j in range(3):
            v = min(1.0, vals[i][j] / 2.5)
            ax.add_patch(Rectangle((x0 + i * 0.085 - 0.016, y0 + 0.06 - j * 0.023), 0.032, 0.018, color=cmap(0.18 + 0.72 * v), lw=0))
            ax.text(x0 + i * 0.085 - 0.022, y0 + 0.068 - j * 0.023, residues[i][j], fontsize=7.1, color=COL["ink"], ha="right", va="center")
    ax.plot([0.57, 0.605], [0.11, 0.125], color=COL["field"], lw=2.0)
    ax.scatter([0.57, 0.605], [0.11, 0.125], s=24, color="white", edgecolor=COL["field"], linewidth=1.1)
    ax.text(0.61, 0.13, "pairwise edge", fontsize=7.2, color=COL["muted"], ha="left")

    # right side
    card(ax, 0.75, 0.56, 0.20, 0.15, "Structured decoder", ["field-to-sequence search", "constrained beam decoding", "energy-guided generation"], "#fdf4ff", "#f5d0fe", COL["decoder"])
    card(ax, 0.75, 0.36, 0.20, 0.15, "Calibrated selector", ["target-wise normalization", "prior consistency", "safety gating"], "#f8fafc", "#dbe4f0", COL["selector"])
    card(ax, 0.75, 0.18, 0.20, 0.11, "Outputs", ["ranked designs", "benchmark tables", "case-study bundles"], "#fafafa", "#e5e7eb", "#64748b", title_size=11.8)

    # arrows
    mids = [y + 0.047 for y in ys]
    field_targets = [0.68, 0.62, 0.56, 0.50, 0.44]
    for sy, ty, c in zip(mids, field_targets, [COL["geom"], COL["phylo"], COL["asr"], COL["retr"], COL["env"]]):
        arrow(ax, 0.28, sy, 0.36, ty, c, lw=2.1)
    arrow(ax, 0.68, 0.60, 0.75, 0.63, COL["decoder"], lw=2.4)
    arrow(ax, 0.68, 0.49, 0.75, 0.43, COL["selector"], lw=2.4)
    arrow(ax, 0.85, 0.56, 0.85, 0.51, COL["decoder"], lw=1.8)
    arrow(ax, 0.85, 0.36, 0.85, 0.29, COL["selector"], lw=1.8)

    fig.savefig(FIG / "figure1_mars_field_architecture_v3.svg")
    fig.savefig(FIG / "figure1_mars_field_architecture_v3.png", dpi=300)


if __name__ == "__main__":
    render()
