from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "outputs" / "paper_bundle_v1"
FIG = PAPER / "figures"

BENCH = pd.read_csv(PAPER / "figure2_benchmark_overview.csv")
FAM = pd.read_csv(PAPER / "figure2_family_summary.csv")

COL = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "axis": "#cbd5e1",
    "overall": "#24364b",
    "learned": "#dc6803",
    "blue": "#2563eb",
    "teal": "#0f766e",
    "rose": "#e11d48",
    "case1": "#1d4ed8",
    "case2": "#0f766e",
    "case3": "#b45309",
    "case4": "#7c3aed",
}
SRC = {
    "local_proposal": "#334155",
    "esm_if": "#2563eb",
    "mars_mpnn": "#0f766e",
    "fusion_decoder": "#c2410c",
}
CASE_COLORS = {
    "1LBT": COL["case1"],
    "tem1_1btl": COL["case2"],
    "petase_5xh3": COL["case3"],
    "petase_5xfy": COL["case3"],
    "CLD_3Q09_TOPIC": COL["case4"],
    "CLD_3Q09_NOTOPIC": COL["case4"],
}


def style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": COL["axis"],
            "axes.linewidth": 0.8,
            "xtick.color": COL["muted"],
            "ytick.color": COL["muted"],
            "text.color": COL["ink"],
            "axes.labelcolor": COL["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def title(ax, tag, txt):
    ax.text(-0.02, 1.05, tag, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="left", va="bottom")
    ax.text(0.05, 1.05, txt, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom")


def cl(x: str) -> str:
    return x.replace("_", " ")


def render():
    style()
    FIG.mkdir(parents=True, exist_ok=True)

    df = BENCH.copy()
    df["gap"] = df["overall_score"] - df["best_learned_score"]
    df["overall_color"] = df["overall_source"].map(SRC).fillna(COL["overall"])

    fig = plt.figure(figsize=(13.2, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.34)

    # A: target-level dumbbell
    ax = fig.add_subplot(gs[:, 0])
    plot = df.sort_values("overall_score").reset_index(drop=True)
    y = np.arange(len(plot))
    ax.hlines(y, plot["best_learned_score"], plot["overall_score"], color=COL["axis"], lw=2.0)
    ax.scatter(plot["best_learned_score"], y, s=50, color=COL["learned"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(plot["overall_score"], y, s=58, color=COL["overall"], edgecolor="white", linewidth=0.8, zorder=4)
    for i, row in plot.iterrows():
        flags = []
        if bool(row["asr_prior_enabled"]):
            flags.append("ASR")
        if bool(row["family_prior_enabled"]):
            flags.append("FP")
        suffix = f" [{' / '.join(flags)}]" if flags else ""
        ax.text(row["overall_score"] + 0.18, i, f"{cl(row['target'])}{suffix}", va="center", fontsize=8.5)
    ax.axvline(0, color=COL["axis"], lw=0.9)
    ax.set_yticks([])
    ax.set_xlabel("Ranking score")
    ax.grid(axis="x", color=COL["grid"], lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    title(ax, "A", "Target-level winner stabilization")
    ax.legend(
        handles=[
            mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["overall"], markeredgecolor="white", markersize=8.5, label="overall winner"),
            mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["learned"], markeredgecolor="white", markersize=8.0, label="best learned"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=9,
    )

    # B: family lollipop gap
    ax2 = fig.add_subplot(gs[0, 1])
    fam = FAM.copy()
    fam["gap"] = fam["mean_overall_score"] - fam["mean_best_learned_score"]
    fam = fam.sort_values("gap", ascending=True).reset_index(drop=True)
    y2 = np.arange(len(fam))
    ax2.hlines(y2, fam["mean_best_learned_score"], fam["mean_overall_score"], color=COL["axis"], lw=2.0)
    ax2.scatter(fam["mean_best_learned_score"], y2, s=40, color=COL["learned"], edgecolor="white", linewidth=0.8, zorder=3)
    ax2.scatter(fam["mean_overall_score"], y2, s=46, color=COL["blue"], edgecolor="white", linewidth=0.8, zorder=4)
    ax2.axvline(0, color=COL["axis"], lw=0.9)
    ax2.set_yticks(y2)
    ax2.set_yticklabels([v.replace("_", " ") for v in fam["family"]], fontsize=8.1)
    ax2.set_xlabel("Mean score")
    ax2.grid(axis="x", color=COL["grid"], lw=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    title(ax2, "B", "Family-level transfer gap")

    # C: engineering consistency with highlighted case studies
    ax3 = fig.add_subplot(gs[1, 1])
    others = df[~df["target"].isin(CASE_COLORS.keys())]
    ax3.scatter(
        others["overall_score"],
        others["overall_mars_score"],
        s=36,
        color="#94a3b8",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.85,
        zorder=2,
    )
    for _, row in df[df["target"].isin(CASE_COLORS.keys())].iterrows():
        color = CASE_COLORS[row["target"]]
        ax3.scatter(
            row["overall_score"],
            row["overall_mars_score"],
            s=86,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )
        ax3.text(
            row["overall_score"] + 0.07,
            row["overall_mars_score"] + 0.06,
            cl(row["target"]).replace(" ", "\n"),
            fontsize=7.3,
            color=color,
        )
    ax3.axvline(0, color=COL["axis"], lw=0.9)
    ax3.axhline(0, color=COL["axis"], lw=0.9)
    ax3.set_xlabel("Overall ranking score")
    ax3.set_ylabel("Overall MARS score")
    ax3.grid(color=COL["grid"], lw=0.8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    title(ax3, "C", "Engineering-consistent winners")

    fig.text(0.01, 1.015, "MARS-FIELD benchmark overview", fontsize=17, fontweight="bold", ha="left")
    fig.text(
        0.01,
        0.988,
        "Final winners remain engineering-consistent while family-level rescue gaps expose where calibration matters most.",
        fontsize=10.5,
        ha="left",
        color=COL["muted"],
    )
    fig.savefig(FIG / "figure2_benchmark_overview_v3.svg")
    fig.savefig(FIG / "figure2_benchmark_overview_v3.png", dpi=300)


if __name__ == "__main__":
    render()
