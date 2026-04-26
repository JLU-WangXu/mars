from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "benchmark_twelvepack" / "neural_comparison_summary.csv"
OUT_DIR = ROOT / "outputs" / "paper_bundle_v1" / "figures"

COL = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "axis": "#cbd5e1",
    "overall": "#24364b",
    "neural": "#dc6803",
    "good": "#0f766e",
    "bad": "#e11d48",
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


def cl(x: str) -> str:
    return x.replace("_", " ")


def title(ax, tag, text):
    ax.text(-0.02, 1.05, tag, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="left", va="bottom")
    ax.text(0.05, 1.05, text, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom")


def main():
    style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY)
    df["match_any"] = df["neural_matches_overall"] | df["neural_matches_best_learned"]
    df["preferred"] = df["neural_mars_delta_vs_overall"] > 0

    fig = plt.figure(figsize=(11.6, 5.6))
    gs = fig.add_gridspec(1, 2, wspace=0.32)

    # A: delta map
    ax = fig.add_subplot(gs[0, 0])
    plot = df.sort_values("neural_mars_delta_vs_overall", ascending=True).reset_index(drop=True)
    y = range(len(plot))
    colors = [COL["good"] if value >= 0 else COL["bad"] for value in plot["neural_mars_delta_vs_overall"]]
    ax.barh(y, plot["neural_mars_delta_vs_overall"], color=colors, height=0.62)
    ax.axvline(0, color=COL["axis"], lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([cl(v) for v in plot["target"]], fontsize=8.2)
    ax.set_xlabel("Neural MARS delta vs final winner")
    ax.grid(axis="x", color=COL["grid"], lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    title(ax, "A", "Neural engineering agreement")

    # B: match map
    ax2 = fig.add_subplot(gs[0, 1])
    x = plot["neural_top_mars_score"]
    y2 = plot["overall_mars_score"]
    colors2 = [COL["good"] if match else COL["neural"] for match in plot["match_any"]]
    sizes = [90 if match else 70 for match in plot["match_any"]]
    ax2.scatter(x, y2, s=sizes, c=colors2, edgecolor="white", linewidth=0.8)
    lim_low = min(float(x.min()), float(y2.min())) - 0.7
    lim_high = max(float(x.max()), float(y2.max())) + 0.7
    ax2.plot([lim_low, lim_high], [lim_low, lim_high], color=COL["axis"], lw=1.2, linestyle="--")
    for _, row in plot.iterrows():
        ax2.text(
            float(row["neural_top_mars_score"]) + 0.05,
            float(row["overall_mars_score"]) + 0.05,
            cl(row["target"]).replace(" ", "\n"),
            fontsize=7.1,
            color=COL["muted"],
        )
    ax2.set_xlabel("Neural top candidate MARS score")
    ax2.set_ylabel("Final overall MARS score")
    ax2.set_xlim(lim_low, lim_high)
    ax2.set_ylim(lim_low, lim_high)
    ax2.grid(color=COL["grid"], lw=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    title(ax2, "B", "Neural vs final engineering outcome")

    fig.text(0.01, 1.03, "MARS-FIELD neural comparison overview", fontsize=16, fontweight="bold", ha="left")
    fig.text(
        0.01,
        0.995,
        "The current neural reranker sometimes matches the final engineering winner, but still diverges strongly on several targets.",
        fontsize=10.2,
        ha="left",
        color=COL["muted"],
    )
    fig.savefig(OUT_DIR / "figure_neural_comparison_v1.svg")
    fig.savefig(OUT_DIR / "figure_neural_comparison_v1.png", dpi=300)


if __name__ == "__main__":
    main()
