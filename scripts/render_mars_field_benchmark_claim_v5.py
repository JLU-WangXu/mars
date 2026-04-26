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
FAMILY = pd.read_csv(OUT / "family_summary.csv")

COL = {
    "ink": "#111827",
    "muted": "#667085",
    "grid": "#e5e7eb",
    "axis": "#cbd5e1",
    "pos": "#0f766e",
    "neg": "#b42318",
    "neutral": "#475467",
    "card_edge": "#d0d5dd",
    "retained": "#c2410c",
    "rejected": "#cbd5e1",
    "overlap": "#98a2b3",
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


def metric_card(ax, xy, wh, value: str, label: str, color: str) -> None:
    x, y = xy
    w, h = wh
    patch = mpl.patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.016,rounding_size=0.03",
        linewidth=0.9,
        edgecolor=COL["card_edge"],
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + 0.06, y + h * 0.62, value, transform=ax.transAxes, fontsize=18, fontweight="bold", color=color, ha="left", va="center")
    ax.text(x + 0.06, y + h * 0.28, label, transform=ax.transAxes, fontsize=8.8, color=COL["muted"], ha="left", va="center")


def build_family_matrix() -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    fam = FAMILY.copy()
    cols = ["mean_overall_score", "mean_best_learned_score", "family_prior_targets"]
    raw = fam[cols].to_numpy(dtype=float)
    mat = raw.copy()
    for j in range(mat.shape[1]):
        col = mat[:, j]
        rng = col.max() - col.min()
        mat[:, j] = 0.5 if rng < 1e-9 else (col - col.min()) / rng
    return mat, [cl(v) for v in fam["family"]], ["mean\nfinal", "mean\nlearned", "family\nprior"], raw


def render() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)

    delta_col = "policy_selection_score_delta_final_minus_current"
    df = COMPARE[["target", delta_col]].copy()
    df["family"] = df["target"].map(BENCH.set_index("target")["family"].to_dict())
    df = df.sort_values(delta_col, ascending=True).reset_index(drop=True)

    bench = BENCH.copy()
    bench["decoder_overlap_count"] = (
        bench["neural_decoder_generated_count"]
        - bench["neural_decoder_novel_count"]
        - bench["neural_decoder_rejected_count"]
    ).clip(lower=0)
    util = bench.sort_values(["neural_decoder_novel_count", "neural_decoder_generated_count"], ascending=[True, False]).reset_index(drop=True)

    positive = int((COMPARE[delta_col] > 1e-9).sum())
    negative = int((COMPARE[delta_col] < -1e-9).sum())
    mean_delta = float(COMPARE[delta_col].mean())
    decoder_enabled = int(bench["neural_decoder_enabled"].sum())
    retained_targets = int(bench["neural_decoder_injected"].sum())
    retained_total = int(bench["neural_decoder_novel_count"].sum())

    fig = plt.figure(figsize=(12.6, 8.2))
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.45, 1.0],
        height_ratios=[0.85, 1.15, 1.15],
        wspace=0.30,
        hspace=0.38,
    )

    # Panel A
    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(df))
    colors = [COL["pos"] if v > 0 else COL["neg"] for v in df[delta_col]]
    ax.hlines(y, 0, df[delta_col], color=colors, lw=2.6, alpha=0.9, zorder=2)
    ax.scatter(df[delta_col], y, s=70, color=colors, edgecolor="white", linewidth=0.9, zorder=3)
    ax.axvline(0, color=COL["axis"], lw=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([cl(v) for v in df["target"]], fontsize=8.3)
    ax.set_xlabel("Paired policy delta, final minus incumbent")
    ax.grid(axis="x", color=COL["grid"], lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_title(ax, "a", "Target-wise benchmark shifts")
    ax.text(
        0.0,
        -0.09,
        "Green: targets improved under the final controller. Red: targets with lower final policy score.",
        transform=ax.transAxes,
        fontsize=8.4,
        color=COL["muted"],
        ha="left",
        va="top",
    )

    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    panel_title(ax2, "b", "Headline benchmark metrics")
    metric_card(ax2, (0.00, 0.46), (0.46, 0.40), f"{positive}/12", "targets improved", COL["pos"])
    metric_card(ax2, (0.52, 0.46), (0.46, 0.40), f"{negative}/12", "targets decreased", COL["neg"])
    metric_card(ax2, (0.00, 0.02), (0.46, 0.32), f"{mean_delta:+.3f}", "mean paired delta", COL["ink"])
    metric_card(ax2, (0.52, 0.02), (0.46, 0.32), f"{retained_total}", "retained neural-decoder candidates", COL["retained"])

    # Panel C
    ax3 = fig.add_subplot(gs[1, 1])
    y3 = np.arange(len(util))
    ax3.barh(y3, util["decoder_overlap_count"], color=COL["overlap"], height=0.62, label="existing overlap")
    ax3.barh(y3, util["neural_decoder_rejected_count"], left=util["decoder_overlap_count"], color=COL["rejected"], height=0.62, label="rejected")
    ax3.barh(
        y3,
        util["neural_decoder_novel_count"],
        left=util["decoder_overlap_count"] + util["neural_decoder_rejected_count"],
        color=COL["retained"],
        height=0.62,
        label="retained novel",
    )
    ax3.set_yticks(y3)
    ax3.set_yticklabels([cl(v) for v in util["target"]], fontsize=7.7)
    ax3.set_xlabel("Neural decoder candidates")
    ax3.grid(axis="x", color=COL["grid"], lw=0.8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    panel_title(ax3, "c", "Neural decoder utilization")
    ax3.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=1)
    ax3.text(
        0.0,
        -0.18,
        f"Neural decoder enabled on {decoder_enabled}/12 targets and retained candidates on {retained_targets}/12 targets.",
        transform=ax3.transAxes,
        fontsize=8.0,
        color=COL["muted"],
        ha="left",
        va="top",
    )

    # Panel D
    ax4 = fig.add_subplot(gs[2, 1])
    mat, row_labels, col_labels, raw = build_family_matrix()
    ax4.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax4.set_yticks(np.arange(len(row_labels)))
    ax4.set_yticklabels(row_labels, fontsize=7.6)
    ax4.set_xticks(np.arange(len(col_labels)))
    ax4.set_xticklabels(col_labels, fontsize=7.8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text_color = "white" if mat[i, j] > 0.55 else COL["ink"]
            val = raw[i, j]
            label = f"{val:.1f}" if j < 2 else str(int(val))
            ax4.text(j, i, label, ha="center", va="center", fontsize=7.2, color=text_color)
    for spine in ax4.spines.values():
        spine.set_color(COL["axis"])
        spine.set_linewidth(0.8)
    panel_title(ax4, "d", "Family-level summary")

    fig.savefig(FIG / "figure2_benchmark_claim_v5.svg")
    fig.savefig(FIG / "figure2_benchmark_claim_v5.png", dpi=300)


if __name__ == "__main__":
    render()
