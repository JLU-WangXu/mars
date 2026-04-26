from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "outputs" / "paper_bundle_v1"
FIG = PAPER / "figures"

BENCH = pd.read_csv(PAPER / "figure2_benchmark_overview.csv")
DEC = pd.read_csv(PAPER / "figure3_decoder_summary.csv")

COL = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "axis": "#cbd5e1",
    "overall": "#24364b",
    "learned": "#dc6803",
    "decoder": "#ea580c",
    "teal": "#0f766e",
    "rose": "#e11d48",
    "light": "#f8fafc",
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


def cl(x):
    return x.replace("_", " ")


def render():
    style()
    FIG.mkdir(parents=True, exist_ok=True)

    df = BENCH.copy()
    df["gap"] = df["overall_score"] - df["best_learned_score"]
    df["decoder_injected"] = DEC["decoder_injected"].astype(bool).values
    df["decoder_pressure"] = DEC["decoder_rejected_count"].values - DEC["decoder_novel_count"].values

    fig = plt.figure(figsize=(13.2, 8.2))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.34)

    # A: acceptance gate
    ax = fig.add_subplot(gs[0, 0])
    d = DEC.sort_values(["decoder_rejected_count", "decoder_novel_count"], ascending=[True, False]).reset_index(drop=True)
    y = np.arange(len(d))
    ax.barh(y, d["decoder_rejected_count"], color="#cbd5e1", height=0.62, label="rejected")
    ax.barh(y, d["decoder_novel_count"], color=COL["decoder"], height=0.62, label="novel injected")
    ax.set_yticks(y)
    ax.set_yticklabels([cl(v) for v in d["target"]], fontsize=8.2)
    ax.set_xlabel("Decoder candidate count")
    ax.grid(axis="x", color=COL["grid"], lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    title(ax, "A", "Decoder acceptance is tightly gated")

    # B: rescue gap bars
    ax2 = fig.add_subplot(gs[0, 1])
    gap = df.sort_values("gap", ascending=False).reset_index(drop=True)
    x = np.arange(len(gap))
    ax2.bar(x, gap["gap"], color=[COL["teal"] if v > 0 else COL["rose"] for v in gap["gap"]], width=0.66)
    ax2.axhline(0, color=COL["axis"], lw=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([cl(v).replace(" ", "\n") for v in gap["target"]], fontsize=7.7)
    ax2.set_ylabel("Overall - best learned")
    ax2.grid(axis="y", color=COL["grid"], lw=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    title(ax2, "B", "Calibration rescue gap")

    # C: selection rescue map
    ax3 = fig.add_subplot(gs[1, 0])
    sizes = 70 + np.clip(np.abs(df["gap"]) * 10.0, 0, 170)
    colors = [COL["decoder"] if injected else COL["overall"] for injected in df["decoder_injected"]]
    ax3.scatter(df["best_learned_score"], df["overall_score"], s=sizes, c=colors, edgecolor="white", linewidth=1.0, alpha=0.92, zorder=3)
    lim_low = min(df["best_learned_score"].min(), df["overall_score"].min()) - 1.0
    lim_high = max(df["best_learned_score"].max(), df["overall_score"].max()) + 1.0
    ax3.plot([lim_low, lim_high], [lim_low, lim_high], color=COL["axis"], lw=1.2, linestyle="--", zorder=1)
    ax3.fill_between([lim_low, lim_high], [lim_low, lim_high], [lim_high, lim_high], color="#ecfdf5", alpha=0.4, zorder=0)
    annotate = df[(df["gap"].abs() > 1.5) | (df["decoder_injected"]) | (df["target"].isin(["1LBT", "CLD_3Q09_TOPIC", "esterase_7b4q"]))]
    for _, row in annotate.iterrows():
        ax3.text(
            row["best_learned_score"] + 0.10,
            row["overall_score"] + 0.05,
            cl(row["target"]).replace(" ", "\n"),
            fontsize=7.1,
            color=COL["muted"],
        )
    ax3.set_xlim(lim_low, lim_high)
    ax3.set_ylim(-0.6, lim_high)
    ax3.set_xlabel("Best learned score")
    ax3.set_ylabel("Final overall score")
    ax3.grid(color=COL["grid"], lw=0.8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    title(ax3, "C", "Selection rescue map")
    ax3.text(
        0.02,
        0.96,
        "Above diagonal = calibration rescues weak learned winners",
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=COL["muted"],
    )

    # D: source-shift dumbbell
    ax4 = fig.add_subplot(gs[1, 1])
    order = ["local_proposal", "esm_if", "mars_mpnn", "fusion_decoder"]
    labels = ["local", "esm_if", "mars_mpnn", "decoder"]
    overall_counts = df["overall_source"].value_counts().reindex(order, fill_value=0)
    learned_counts = df["best_learned_source"].value_counts().reindex(order, fill_value=0)
    y2 = np.arange(len(order))
    ax4.hlines(y2, learned_counts.values, overall_counts.values, color=COL["axis"], lw=2.2, zorder=1)
    ax4.scatter(learned_counts.values, y2, s=86, color=COL["learned"], edgecolor="white", linewidth=0.9, zorder=3, label="best learned")
    ax4.scatter(overall_counts.values, y2, s=92, color=COL["overall"], edgecolor="white", linewidth=0.9, zorder=4, label="overall winner")
    for i, (lv, ov) in enumerate(zip(learned_counts.values, overall_counts.values)):
        ax4.text(lv - 0.10, i - 0.17, str(int(lv)), ha="right", va="center", fontsize=8.6, color=COL["learned"])
        ax4.text(ov + 0.10, i + 0.17, str(int(ov)), ha="left", va="center", fontsize=8.6, color=COL["overall"])
    ax4.set_yticks(y2)
    ax4.set_yticklabels(labels, fontsize=8.8)
    ax4.invert_yaxis()
    ax4.set_xlabel("Target count")
    ax4.grid(axis="x", color=COL["grid"], lw=0.8)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.legend(frameon=False, fontsize=9, loc="lower right")
    title(ax4, "D", "Source-shift after calibration")

    fig.text(0.01, 1.015, "MARS-FIELD decoder and calibration analysis", fontsize=17, fontweight="bold", ha="left")
    fig.text(
        0.01,
        0.988,
        "Decoder proposals enlarge the search space, but calibration and engineering priors redirect final selection toward stable engineering winners.",
        fontsize=10.5,
        ha="left",
        color=COL["muted"],
    )
    fig.savefig(FIG / "figure3_decoder_calibration_v3.svg")
    fig.savefig(FIG / "figure3_decoder_calibration_v3.png", dpi=300)


if __name__ == "__main__":
    render()
