from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "outputs" / "benchmark_twelvepack" / "benchmark_summary.csv"
FIG = ROOT / "outputs" / "paper_bundle_v1" / "figures"

COL = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "axis": "#cbd5e1",
    "geom": "#2563eb",
    "phylo": "#0f766e",
    "asr": "#7c3aed",
    "retrieval": "#dc6803",
    "environment": "#e11d48",
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
    df = pd.read_csv(BENCH)
    df = df[df["neural_rerank_enabled"] == True].copy()
    if df.empty:
        return

    fig = plt.figure(figsize=(12.4, 5.8))
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    # A: stacked gate usage
    ax = fig.add_subplot(gs[0, 0])
    plot = df.sort_values("neural_gate_retrieval", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot))
    left = np.zeros(len(plot))
    gate_specs = [
        ("neural_gate_geom", COL["geom"], "geom"),
        ("neural_gate_phylo", COL["phylo"], "phylo"),
        ("neural_gate_asr", COL["asr"], "asr"),
        ("neural_gate_retrieval", COL["retrieval"], "retrieval"),
        ("neural_gate_environment", COL["environment"], "env"),
    ]
    for field, color, label in gate_specs:
        values = plot[field].astype(float).to_numpy()
        ax.barh(y, values, left=left, color=color, height=0.62, label=label)
        left = left + values
    ax.set_yticks(y)
    ax.set_yticklabels([cl(v) for v in plot["target"]], fontsize=8.2)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean neural gate weight")
    ax.grid(axis="x", color=COL["grid"], lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=3)
    title(ax, "A", "Neural branch usage profile")

    # B: neural delta vs retrieval/asr activity
    ax2 = fig.add_subplot(gs[0, 1])
    df["neural_delta"] = df["neural_top_mars_score"] - df["overall_mars_score"]
    colors = [COL["good"] if value >= 0 else COL["bad"] for value in df["neural_delta"]]
    sizes = 70 + np.abs(df["neural_delta"]) * 25.0
    ax2.scatter(df["neural_gate_retrieval"], df["neural_delta"], s=sizes, c=colors, edgecolor="white", linewidth=0.8)
    for _, row in df.iterrows():
        tag = cl(row["target"]).replace(" ", "\n")
        if bool(row["asr_prior_enabled"]):
            tag += "\n[ASR]"
        elif bool(row["family_prior_enabled"]):
            tag += "\n[FP]"
        ax2.text(
            float(row["neural_gate_retrieval"]) + 0.005,
            float(row["neural_delta"]) + 0.05,
            tag,
            fontsize=7.0,
            color=COL["muted"],
        )
    ax2.axhline(0, color=COL["axis"], lw=0.9)
    ax2.set_xlabel("Retrieval gate weight")
    ax2.set_ylabel("Neural MARS delta vs final winner")
    ax2.grid(color=COL["grid"], lw=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    title(ax2, "B", "Retrieval-heavy neural behavior")

    fig.text(0.01, 1.03, "MARS-FIELD neural branch diagnostics", fontsize=16, fontweight="bold", ha="left")
    fig.text(
        0.01,
        0.995,
        "Neural gating currently leans heavily on retrieval-rich signals, with uneven gains across targets.",
        fontsize=10.2,
        ha="left",
        color=COL["muted"],
    )
    fig.savefig(FIG / "figure_neural_branch_diagnostics_v1.svg")
    fig.savefig(FIG / "figure_neural_branch_diagnostics_v1.png", dpi=300)


if __name__ == "__main__":
    main()
