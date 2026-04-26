from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs" / "benchmark_twelvepack_neural_default" / "compare_current_vs_neural.csv"
FIG = ROOT / "outputs" / "paper_bundle_v1" / "figures"

COL = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "axis": "#cbd5e1",
    "current": "#24364b",
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


def title(ax, tag, txt):
    ax.text(-0.02, 1.05, tag, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="left", va="bottom")
    ax.text(0.05, 1.05, txt, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom")


def cl(x):
    return x.replace("_", " ")


def main():
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)

    fig = plt.figure(figsize=(11.8, 5.8))
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    plot = df.sort_values("policy_selection_score_delta_neural_minus_current", ascending=True).reset_index(drop=True)
    y = range(len(plot))
    colors = [COL["good"] if v >= 0 else COL["bad"] for v in plot["policy_selection_score_delta_neural_minus_current"]]
    ax.barh(y, plot["policy_selection_score_delta_neural_minus_current"], color=colors, height=0.62)
    ax.axvline(0, color=COL["axis"], lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([cl(v) for v in plot["target"]], fontsize=8.1)
    ax.set_xlabel("Neural policy selection-score delta")
    ax.grid(axis="x", color=COL["grid"], lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    title(ax, "A", "Neural-default policy delta")

    ax2 = fig.add_subplot(gs[0, 1])
    changed = plot[plot["policy_changed"] == True].copy()
    if changed.empty:
        changed = plot.copy()
    y2 = range(len(changed))
    ax2.hlines(y2, changed["policy_selection_score_neural"], changed["policy_selection_score_current"], color=COL["axis"], lw=2.0)
    ax2.scatter(changed["policy_selection_score_current"], y2, s=70, color=COL["current"], edgecolor="white", linewidth=0.8, label="current")
    ax2.scatter(changed["policy_selection_score_neural"], y2, s=70, color=COL["neural"], edgecolor="white", linewidth=0.8, label="neural-default")
    ax2.set_yticks(y2)
    ax2.set_yticklabels([cl(v) for v in changed["target"]], fontsize=8.1)
    ax2.set_xlabel("Policy selection score")
    ax2.grid(axis="x", color=COL["grid"], lw=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    title(ax2, "B", "Targets whose policy selection changes")

    fig.text(0.01, 1.03, "MARS-FIELD policy comparison", fontsize=16, fontweight="bold", ha="left")
    fig.text(
        0.01,
        0.995,
        "Current and neural-default policies mostly agree on final winners today, but the policy deltas reveal where neural selection still degrades engineering quality.",
        fontsize=10.2,
        ha="left",
        color=COL["muted"],
    )
    fig.savefig(FIG / "figure_policy_compare_v1.svg")
    fig.savefig(FIG / "figure_policy_compare_v1.png", dpi=300)


if __name__ == "__main__":
    main()
