from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "benchmark_twelvepack_final"
FIG = ROOT / "outputs" / "paper_bundle_v1" / "figures"

BENCH = pd.read_csv(OUT / "benchmark_summary.csv")
COMPARE = pd.read_csv(OUT / "compare_current_vs_final.csv")
ABL = pd.read_csv(OUT / "ablation_summary.csv")
NEURAL = pd.read_csv(OUT / "neural_comparison_summary.csv")

COL = {
    "ink": "#0f172a",
    "muted": "#667085",
    "grid": "#e2e8f0",
    "axis": "#cbd5e1",
    "oxidation": "#b42318",
    "surface": "#1d4ed8",
    "evolution": "#0f766e",
    "good": "#0f766e",
    "bad": "#b42318",
    "neutral": "#64748b",
}


def style() -> None:
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


def panel_title(ax, tag: str, title: str) -> None:
    ax.text(-0.02, 1.05, tag, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="left", va="bottom")
    ax.text(0.06, 1.05, title, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom")


def cl(text: str) -> str:
    return str(text).replace("_", " ")


def ablation_summary_rows() -> list[tuple[str, int, float]]:
    full = ABL[ABL["ablation"] == "full"][["target", "top_mutations", "ablation_score"]].rename(columns={"top_mutations": "full_mut", "ablation_score": "full_score"})
    rows = []
    for ab in ["no_oxidation", "no_surface", "no_evolution"]:
        sub = ABL[ABL["ablation"] == ab][["target", "top_mutations", "ablation_score"]].rename(columns={"top_mutations": "mut", "ablation_score": "score"})
        merged = full.merge(sub, on="target")
        changed = int((merged["full_mut"] != merged["mut"]).sum())
        mean_drop = float((merged["full_score"] - merged["score"]).mean())
        rows.append((ab, changed, mean_drop))
    return rows


def render() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.6, 8.6))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.34)

    # A: ablation summary
    ax = fig.add_subplot(gs[0, 0])
    rows = ablation_summary_rows()
    labels = [r[0].replace("_", " ") for r in rows]
    changed = [r[1] for r in rows]
    mean_drop = [r[2] for r in rows]
    x = np.arange(len(labels))
    bar_colors = [COL["oxidation"], COL["surface"], COL["evolution"]]
    bars = ax.bar(x, changed, color=bar_colors, width=0.56)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Targets with changed top candidate")
    ax.set_ylim(0, max(changed) + 2)
    ax.grid(axis="y", color=COL["grid"], lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_title(ax, "A", "Constraint sensitivity")
    ax2 = ax.twinx()
    ax2.plot(x, mean_drop, color=COL["ink"], marker="o", lw=1.8)
    ax2.set_ylabel("Mean full-minus-ablation score")
    ax2.tick_params(axis="y", colors=COL["muted"])
    for bar, val in zip(bars, changed):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(val), ha="center", va="bottom", fontsize=8.2)
    for xi, val in zip(x, mean_drop):
        ax2.text(xi, val + 0.15, f"{val:.2f}", ha="center", va="bottom", fontsize=7.8, color=COL["ink"])

    # B: gate heatmap
    axb = fig.add_subplot(gs[0, 1])
    gate_cols = [
        ("neural_gate_geom", "geom"),
        ("neural_gate_phylo", "phylo"),
        ("neural_gate_asr", "asr"),
        ("neural_gate_retrieval", "retr."),
        ("neural_gate_environment", "env"),
    ]
    gate_df = BENCH[["target"] + [c[0] for c in gate_cols]].copy().sort_values("target").reset_index(drop=True)
    mat = gate_df[[c[0] for c in gate_cols]].to_numpy(dtype=float)
    im = axb.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.4, float(mat.max())))
    axb.set_yticks(np.arange(len(gate_df)))
    axb.set_yticklabels([cl(v) for v in gate_df["target"]], fontsize=7.8)
    axb.set_xticks(np.arange(len(gate_cols)))
    axb.set_xticklabels([c[1] for c in gate_cols], fontsize=8.2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axb.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7.0, color="white" if mat[i, j] > mat.max() * 0.55 else COL["ink"])
    for spine in axb.spines.values():
        spine.set_color(COL["axis"])
        spine.set_linewidth(0.8)
    panel_title(axb, "B", "Target-specific neural gate profile")

    # C: limitation slope plot
    axc = fig.add_subplot(gs[1, 0])
    delta_col = "policy_selection_score_delta_final_minus_current"
    neg = COMPARE[COMPARE[delta_col] < 0].copy().sort_values(delta_col, ascending=True).reset_index(drop=True)
    y = np.arange(len(neg))
    for i, row in neg.iterrows():
        axc.plot(
            [0, 1],
            [float(row["policy_selection_score_current"]), float(row["policy_selection_score_final"])],
            color=COL["bad"],
            lw=2.4,
            alpha=0.9,
        )
        axc.scatter([0, 1], [float(row["policy_selection_score_current"]), float(row["policy_selection_score_final"])], s=[56, 56], color=[COL["neutral"], COL["bad"]], edgecolor="white", linewidth=0.8, zorder=3)
        axc.text(-0.04, float(row["policy_selection_score_current"]), cl(row["target"]), ha="right", va="center", fontsize=8.0)
        axc.text(1.04, float(row["policy_selection_score_final"]), f"{float(row[delta_col]):+.2f}", ha="left", va="center", fontsize=8.0, color=COL["bad"])
    axc.set_xlim(-0.18, 1.18)
    axc.set_xticks([0, 1])
    axc.set_xticklabels(["incumbent", "final"])
    axc.set_ylabel("Policy selection score")
    axc.grid(axis="y", color=COL["grid"], lw=0.8)
    axc.spines["top"].set_visible(False)
    axc.spines["right"].set_visible(False)
    panel_title(axc, "C", "Concentrated limitation cases")

    # D: decoder selectivity
    axd = fig.add_subplot(gs[1, 1])
    dd = BENCH.copy()
    dd["retain_rate"] = dd["neural_decoder_novel_count"] / dd["neural_decoder_generated_count"].replace(0, np.nan)
    dd["retain_rate"] = dd["retain_rate"].fillna(0.0)
    colors = [COL["good"] if int(v) > 0 else COL["neutral"] for v in dd["neural_decoder_novel_count"]]
    sizes = 70 + dd["neural_decoder_novel_count"].to_numpy(dtype=float) * 18.0
    axd.scatter(dd["neural_decoder_generated_count"], dd["retain_rate"], s=sizes, c=colors, edgecolor="white", linewidth=0.9)
    for _, row in dd.iterrows():
        if int(row["neural_decoder_novel_count"]) > 0 or row["target"] in {"1LBT", "CLD_3Q09_TOPIC"}:
            axd.text(
                float(row["neural_decoder_generated_count"]) + 0.25,
                float(row["retain_rate"]) + 0.005,
                cl(row["target"]).replace(" ", "\n"),
                fontsize=7.1,
                color=COL["muted"],
            )
    axd.set_xlabel("Neural decoder preview count")
    axd.set_ylabel("Retained novel / preview")
    axd.set_xlim(0, max(35, float(dd["neural_decoder_generated_count"].max()) + 2))
    axd.set_ylim(-0.01, max(0.55, float(dd["retain_rate"].max()) + 0.04))
    axd.grid(color=COL["grid"], lw=0.8)
    axd.spines["top"].set_visible(False)
    axd.spines["right"].set_visible(False)
    panel_title(axd, "D", "Decoder selectivity rather than blind novelty")

    fig.text(0.01, 1.015, "MARS-FIELD mechanism and limitation figure", fontsize=17, fontweight="bold", ha="left")
    fig.text(
        0.01,
        0.988,
        "Ablations identify dominant constraints, neural diagnostics reveal target-dependent evidence use, and failures remain concentrated rather than diffuse.",
        fontsize=10.3,
        ha="left",
        color=COL["muted"],
    )

    fig.savefig(FIG / "figure3_mechanism_limitations_v4.svg")
    fig.savefig(FIG / "figure3_mechanism_limitations_v4.png", dpi=300)


if __name__ == "__main__":
    render()
