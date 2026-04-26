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

COL = {
    "ink": "#111827",
    "muted": "#667085",
    "grid": "#e5e7eb",
    "axis": "#cbd5e1",
    "oxidation": "#b42318",
    "surface": "#1d4ed8",
    "evolution": "#0f766e",
    "good": "#0f766e",
    "bad": "#b42318",
    "neutral": "#475467",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
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


def panel_title(ax, tag: str, text: str) -> None:
    ax.text(0.0, 1.04, tag, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="left", va="bottom")
    ax.text(0.08, 1.04, text, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom")


def cl(text: str) -> str:
    return str(text).replace("_", " ")


def build_ablation_panel():
    full = ABL[ABL["ablation"] == "full"][["target", "top_mutations", "ablation_score"]].rename(columns={"top_mutations": "full_mut", "ablation_score": "full_score"})
    rows = []
    for ab in ["no_oxidation", "no_surface", "no_evolution"]:
        sub = ABL[ABL["ablation"] == ab][["target", "top_mutations", "ablation_score"]].rename(columns={"top_mutations": "mut", "ablation_score": "score"})
        merged = full.merge(sub, on="target")
        rows.append(
            {
                "ablation": ab,
                "changed": int((merged["full_mut"] != merged["mut"]).sum()),
                "mean_drop": float((merged["full_score"] - merged["score"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def render() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.8, 8.2))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.36)

    # A: ablation summary
    ax = fig.add_subplot(gs[0, 0])
    ab = build_ablation_panel()
    x = np.arange(len(ab))
    colors = [COL["oxidation"], COL["surface"], COL["evolution"]]
    bars = ax.bar(x, ab["changed"], color=colors, width=0.54)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", " ") for v in ab["ablation"]], fontsize=8.5)
    ax.set_ylabel("Targets with changed top candidate")
    ax.grid(axis="y", color=COL["grid"], lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_title(ax, "a", "Ablation sensitivity")
    for bar, val in zip(bars, ab["changed"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, str(int(val)), ha="center", va="bottom", fontsize=8.1)
    ax2 = ax.twinx()
    ax2.plot(x, ab["mean_drop"], color=COL["ink"], marker="o", lw=1.7)
    ax2.set_ylabel("Mean full-minus-ablation score")
    for xi, val in zip(x, ab["mean_drop"]):
        ax2.text(xi, val + 0.12, f"{val:.2f}", ha="center", va="bottom", fontsize=7.8, color=COL["ink"])

    # B: gate heatmap
    axb = fig.add_subplot(gs[0, 1])
    gate_cols = [
        ("neural_gate_geom", "geom"),
        ("neural_gate_phylo", "phylo"),
        ("neural_gate_asr", "asr"),
        ("neural_gate_retrieval", "retr."),
        ("neural_gate_environment", "env"),
    ]
    gate_df = BENCH[["target"] + [g[0] for g in gate_cols]].copy().sort_values("target").reset_index(drop=True)
    mat = gate_df[[g[0] for g in gate_cols]].to_numpy(dtype=float)
    axb.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.4, float(mat.max())))
    axb.set_yticks(np.arange(len(gate_df)))
    axb.set_yticklabels([cl(v) for v in gate_df["target"]], fontsize=7.7)
    axb.set_xticks(np.arange(len(gate_cols)))
    axb.set_xticklabels([g[1] for g in gate_cols], fontsize=8.0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axb.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7.0, color="white" if mat[i, j] > mat.max() * 0.55 else COL["ink"])
    for spine in axb.spines.values():
        spine.set_color(COL["axis"])
        spine.set_linewidth(0.8)
    panel_title(axb, "b", "Target-specific neural gate profile")

    # C: concentrated limitation cases
    axc = fig.add_subplot(gs[1, 0])
    delta_col = "policy_selection_score_delta_final_minus_current"
    neg = COMPARE[COMPARE[delta_col] < 0].copy().sort_values(delta_col, ascending=True).reset_index(drop=True)
    y = np.arange(len(neg))
    current_vals = neg["policy_selection_score_current"].to_numpy(dtype=float)
    final_vals = neg["policy_selection_score_final"].to_numpy(dtype=float)
    axc.hlines(y, final_vals, current_vals, color="#e4e7ec", lw=3.0, zorder=1)
    axc.scatter(current_vals, y, s=68, color=COL["neutral"], edgecolor="white", linewidth=0.8, zorder=3)
    axc.scatter(final_vals, y, s=68, color=COL["bad"], edgecolor="white", linewidth=0.8, zorder=4)
    axc.set_yticks(y)
    axc.set_yticklabels([cl(v) for v in neg["target"]], fontsize=8.1)
    axc.set_xlabel("Policy selection score")
    axc.grid(axis="x", color=COL["grid"], lw=0.8)
    axc.spines["top"].set_visible(False)
    axc.spines["right"].set_visible(False)
    panel_title(axc, "c", "Concentrated limitation cases")
    for i, row in neg.iterrows():
        axc.text(final_vals[i] - 0.05, i - 0.15, f"{float(row[delta_col]):+.2f}", ha="right", va="center", fontsize=8.0, color=COL["bad"])

    # D: decoder selectivity
    axd = fig.add_subplot(gs[1, 1])
    dd = BENCH.copy()
    dd["retain_rate"] = dd["neural_decoder_novel_count"] / dd["neural_decoder_generated_count"].replace(0, np.nan)
    dd["retain_rate"] = dd["retain_rate"].fillna(0.0)
    colors = [COL["good"] if int(v) > 0 else COL["neutral"] for v in dd["neural_decoder_novel_count"]]
    sizes = 80 + dd["neural_decoder_novel_count"].to_numpy(dtype=float) * 18.0
    axd.scatter(dd["neural_decoder_generated_count"], dd["retain_rate"], s=sizes, c=colors, edgecolor="white", linewidth=0.9)
    for _, row in dd.iterrows():
        if int(row["neural_decoder_novel_count"]) > 0 or row["target"] in {"1LBT", "CLD_3Q09_TOPIC"}:
            axd.text(
                float(row["neural_decoder_generated_count"]) + 0.20,
                float(row["retain_rate"]) + 0.005,
                cl(row["target"]).replace(" ", "\n"),
                fontsize=7.0,
                color=COL["muted"],
            )
    axd.set_xlabel("Neural decoder preview count")
    axd.set_ylabel("Retained novel / preview")
    axd.grid(color=COL["grid"], lw=0.8)
    axd.spines["top"].set_visible(False)
    axd.spines["right"].set_visible(False)
    panel_title(axd, "d", "Decoder selectivity")

    fig.savefig(FIG / "figure3_mechanism_limitations_v5.svg")
    fig.savefig(FIG / "figure3_mechanism_limitations_v5.png", dpi=300)


if __name__ == "__main__":
    render()
