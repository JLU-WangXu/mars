from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "paper_bundle_v1" / "figures"

COL = {
    "ink": "#111827",
    "muted": "#667085",
    "line": "#cbd5e1",
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


def card(ax, x, y, w, h, title, lines, face, edge, accent=None, title_size=12.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    if accent is not None:
        ax.add_patch(Rectangle((x, y + h - 0.018), w, 0.018, color=accent, lw=0))
    ax.text(x + 0.02, y + h - 0.05, title, fontsize=title_size, fontweight="bold", color=COL["ink"], ha="left", va="top")
    yy = y + h - 0.105
    for line in lines:
        ax.text(x + 0.025, yy, line, fontsize=8.8, color=COL["muted"], ha="left", va="top")
        yy -= 0.04


def arrow(ax, x1, y1, x2, y2, color, rad=0.0, lw=2.2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            alpha=0.92,
        )
    )


def field_matrix(ax, x, y, w, h):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        edgecolor="#d8b4fe",
        facecolor="#faf5ff",
    )
    ax.add_patch(patch)
    ax.text(x + 0.02, y + h - 0.045, "Shared residue field", fontsize=13, fontweight="bold", color=COL["ink"], ha="left", va="top")
    ax.text(x + 0.02, y + h - 0.085, "site-wise residue energies  U(i, a)", fontsize=9, color=COL["muted"], ha="left")
    ax.text(x + 0.02, y + h - 0.115, "pairwise couplings  C(i, j, a, b)", fontsize=9, color=COL["muted"], ha="left")

    # heatmap slab
    nx, ny = 10, 5
    rng = np.random.default_rng(7)
    mat = rng.random((ny, nx))
    cell_w = w * 0.42 / nx
    cell_h = h * 0.28 / ny
    x0 = x + 0.03
    y0 = y + 0.10
    cmap = mpl.cm.get_cmap("viridis")
    for i in range(ny):
        for j in range(nx):
            color = cmap(0.15 + 0.75 * mat[i, j])
            ax.add_patch(Rectangle((x0 + j * cell_w, y0 + i * cell_h), cell_w * 0.96, cell_h * 0.94, color=color, lw=0))

    ax.text(x0, y0 - 0.025, "design position", fontsize=7.5, color=COL["muted"])
    ax.text(x0 - 0.022, y0 + ny * cell_h * 0.45, "amino acid", fontsize=7.5, color=COL["muted"], rotation=90, va="center")

    # pairwise mini graph
    px = x + w * 0.60
    py = y + 0.14
    pts = [(px, py + 0.12), (px + 0.10, py + 0.20), (px + 0.22, py + 0.15), (px + 0.14, py + 0.04)]
    for a, b in [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color="#c084fc", lw=2.2, alpha=0.75)
    for i, (pxi, pyi) in enumerate(pts):
        ax.scatter([pxi], [pyi], s=75, color="#ffffff", edgecolor="#7c3aed", linewidth=1.8, zorder=3)
        ax.text(pxi, pyi - 0.036, f"i{i+1}", fontsize=7.5, ha="center", color=COL["muted"])
    ax.text(x + w * 0.56, y + 0.31, "pairwise coupling layer", fontsize=8.5, color=COL["muted"])

    # equation box
    eq = FancyBboxPatch(
        (x + 0.03, y + h - 0.19), w - 0.06, 0.08,
        boxstyle="round,pad=0.01,rounding_size=0.018",
        linewidth=0.8,
        edgecolor="#e9d5ff",
        facecolor="white",
    )
    ax.add_patch(eq)
    ax.text(
        x + w / 2,
        y + h - 0.15,
        "E(x) = Σ_i U(i, x_i) + Σ_(i,j) C(i, j, x_i, x_j)",
        fontsize=13,
        fontweight="bold",
        color=COL["field"],
        ha="center",
        va="center",
    )


def render():
    style()
    FIG.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.2, 7.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.965, "Figure 1", fontsize=10.5, color=COL["muted"], ha="left", va="top")
    ax.text(0.03, 0.93, "MARS-FIELD integrates heterogeneous evidence into a shared residue field", fontsize=17, fontweight="bold", color=COL["ink"], ha="left", va="top")
    ax.text(0.03, 0.895, "The native decision object of the method is a residue-level field, not a generator vote.", fontsize=10.2, color=COL["muted"], ha="left", va="top")

    # left evidence cards
    ys = [0.71, 0.58, 0.45, 0.32, 0.19]
    entries = [
        ("Geometric encoder", ["backbone graph", "local geometry", "design/protected masks"], "#eff6ff", "#bfdbfe", COL["geom"]),
        ("Phylo-sequence encoder", ["homolog MSA", "conservation", "family differential priors"], "#f0fdf4", "#bbf7d0", COL["phylo"]),
        ("Ancestral lineage encoder", ["ASR posterior", "posterior entropy", "lineage confidence"], "#ecfeff", "#99f6e4", COL["asr"]),
        ("Retrieval memory encoder", ["motif atlas", "prototype memory", "structural motif support"], "#fffbeb", "#fde68a", COL["retr"]),
        ("Environment hypernetwork", ["oxidation pressure", "flexible-surface burden", "engineering context"], "#fff7ed", "#fdba74", COL["env"]),
    ]
    for y, (title, lines, face, edge, accent) in zip(ys, entries):
        card(ax, 0.04, y, 0.25, 0.10, title, lines, face, edge, accent=accent)

    # central field
    field_matrix(ax, 0.35, 0.26, 0.34, 0.50)

    # right modules
    card(
        ax,
        0.75,
        0.56,
        0.20,
        0.18,
        "Structured decoder",
        ["neural field decoding", "constrained beam search", "energy-guided sequence generation"],
        "#fdf4ff",
        "#f5d0fe",
        accent=COL["decoder"],
    )
    card(
        ax,
        0.75,
        0.32,
        0.20,
        0.18,
        "Calibrated selector",
        ["target-wise normalization", "prior consistency", "safety gating and final policy"],
        "#f8fafc",
        "#dbe4f0",
        accent=COL["selector"],
    )
    card(
        ax,
        0.75,
        0.12,
        0.20,
        0.13,
        "Outputs",
        ["ranked designs", "benchmark tables", "case-study structure bundles"],
        "#fafafa",
        "#e5e7eb",
        accent="#64748b",
        title_size=12.0,
    )

    # arrows from evidence to field
    colors = [COL["geom"], COL["phylo"], COL["asr"], COL["retr"], COL["env"]]
    ymid = [y + 0.05 for y in ys]
    targets = [0.68, 0.62, 0.56, 0.50, 0.44]
    for sy, ty, c in zip(ymid, targets, colors):
        arrow(ax, 0.29, sy, 0.35, ty, c, rad=0.0, lw=2.2)

    # arrows to right
    arrow(ax, 0.69, 0.58, 0.75, 0.65, COL["decoder"], rad=0.0, lw=2.6)
    arrow(ax, 0.69, 0.48, 0.75, 0.41, COL["selector"], rad=0.0, lw=2.6)
    arrow(ax, 0.85, 0.56, 0.85, 0.50, COL["decoder"], rad=0.0, lw=2.0)
    arrow(ax, 0.85, 0.32, 0.85, 0.25, COL["selector"], rad=0.0, lw=2.0)

    # small panel letters
    ax.text(0.03, 0.83, "a", fontsize=12, fontweight="bold", color=COL["ink"])
    ax.text(0.35, 0.80, "b", fontsize=12, fontweight="bold", color=COL["ink"])
    ax.text(0.75, 0.80, "c", fontsize=12, fontweight="bold", color=COL["ink"])
    ax.text(0.35, 0.22, "d", fontsize=12, fontweight="bold", color=COL["ink"])
    ax.text(0.38, 0.22, "Example target-level field instantiation", fontsize=10.5, fontweight="bold", color=COL["ink"])
    ax.text(0.38, 0.18, "1LBT: residue preferences at 249/251/298 and one pairwise edge illustrate the field object in practice.", fontsize=8.8, color=COL["muted"])

    # tiny illustrative inset for target-level field
    x0, y0 = 0.38, 0.07
    labels = ["249", "251", "298"]
    residues = [["Q", "R", "L"], ["E", "S", "A"], ["L", "I", "V"]]
    scores = [[1.96, 1.19, 0.91], [1.77, 1.30, 1.07], [2.36, 1.13, 0.48]]
    for i, pos in enumerate(labels):
        ax.text(x0 + i * 0.095, y0 + 0.08, pos, fontsize=8.2, fontweight="bold", color=COL["muted"], ha="center")
        for j in range(3):
            val = min(1.0, max(0.0, scores[i][j] / 2.5))
            color = mpl.cm.get_cmap("viridis")(0.20 + 0.70 * val)
            ax.add_patch(Rectangle((x0 + i * 0.095 - 0.022, y0 + 0.05 - j * 0.022), 0.044, 0.018, color=color, lw=0))
            ax.text(x0 + i * 0.095 - 0.03, y0 + 0.059 - j * 0.022, residues[i][j], fontsize=7.2, color=COL["ink"], ha="right", va="center")
    ax.plot([0.57, 0.61], [0.10, 0.13], color=COL["field"], lw=2.0)
    ax.scatter([0.57, 0.61], [0.10, 0.13], s=24, color="white", edgecolor=COL["field"], linewidth=1.2)
    ax.text(0.615, 0.135, "pairwise edge", fontsize=7.2, color=COL["muted"], ha="left")

    fig.savefig(FIG / "figure1_mars_field_architecture_v2.svg")
    fig.savefig(FIG / "figure1_mars_field_architecture_v2.png", dpi=300)


if __name__ == "__main__":
    render()
