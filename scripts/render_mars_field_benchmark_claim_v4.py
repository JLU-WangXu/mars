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
    "ink": "#0f172a",
    "muted": "#667085",
    "grid": "#e2e8f0",
    "axis": "#cbd5e1",
    "pos": "#0f766e",
    "neg": "#b42318",
    "neutral": "#64748b",
    "decoder": "#c2410c",
    "reject": "#cbd5e1",
    "overlap": "#94a3b8",
    "heat_hi": "#0f766e",
    "heat_mid": "#f59e0b",
    "heat_lo": "#e5e7eb",
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


def metric_card(ax, x: float, y: float, w: float, h: float, value: str, label: str, color: str) -> None:
    box = mpl.patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=0.9, edgecolor="#d0d5dd", facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(x + 0.04, y + h * 0.60, value, transform=ax.transAxes, fontsize=16, fontweight="bold", color=color, ha="left", va="center")
    ax.text(x + 0.04, y + h * 0.28, label, transform=ax.transAxes, fontsize=8.7, color=COL["muted"], ha="left", va="center")


def render() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)

    delta_col = "policy_selection_score_delta_final_minus_current"
    df = COMPARE[["target", delta_col]].copy()
    df["family"] = df["target"].map(BENCH.set_index("target")["family"].to_dict())
    df = df.sort_values(delta_col, ascending=True).reset_index(drop=True)

    final = BENCH.copy()
    final["decoder_overlap_count"] = (
        final["neural_decoder_generated_count"]
        - final["neural_decoder_novel_count"]
        - final["neural_decoder_rejected_count"]
    ).clip(lower=0)

    fig = plt.figure(figsize=(13.6, 8.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.33)

    # A: paired delta headline panel
    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(df))
    colors = [COL["pos"] if v > 0 else COL["neg"] for v in df[delta_col]]
    ax.hlines(y, 0, df[delta_col], color=colors, lw=2.8, alpha=0.88, zorder=2)
    ax.scatter(df[delta_col], y, s=72, color=colors, edgecolor="white", linewidth=0.9, zorder=3)
    ax.axvline(0, color=COL["axis"], lw=1.1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{cl(t)}  |  {cl(f)}" for t, f in zip(df["target"], df["family"])], fontsize=8.1)
    ax.set_xlabel("Paired policy delta: final minus incumbent")
    ax.grid(axis="x", color=COL["grid"], lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_title(ax, "A", "Panel-level policy shifts")
    ax.text(
        0.02,
        0.03,
        "Positive values indicate a stronger final controller policy score.\nNegative values identify calibration-limited targets.",
        transform=ax.transAxes,
        fontsize=8.6,
        color=COL["muted"],
        ha="left",
        va="bottom",
    )

    # B: metric summary cards
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    positive = int((COMPARE[delta_col] > 1e-9).sum())
    negative = int((COMPARE[delta_col] < -1e-9).sum())
    mean_delta = float(COMPARE[delta_col].mean())
    enabled = int(final["neural_decoder_enabled"].sum())
    injected = int(final["neural_decoder_injected"].sum())
    retained = int(final["neural_decoder_novel_count"].sum())
    metric_card(ax2, 0.00, 0.56, 0.46, 0.30, f"{positive}/12", "targets improved", COL["pos"])
    metric_card(ax2, 0.52, 0.56, 0.46, 0.30, f"{negative}/12", "targets decreased", COL["neg"])
    metric_card(ax2, 0.00, 0.16, 0.46, 0.30, f"{mean_delta:+.3f}", "mean paired delta", COL["ink"])
    metric_card(ax2, 0.52, 0.16, 0.46, 0.30, f"{retained}", "retained neural-decoder candidates", COL["decoder"])
    ax2.text(0.00, 0.99, "B", transform=ax2.transAxes, fontsize=13, fontweight="bold", va="top")
    ax2.text(0.07, 0.99, "Headline benchmark metrics", transform=ax2.transAxes, fontsize=11, fontweight="bold", va="top")
    ax2.text(
        0.00,
        0.02,
        f"Neural decoder enabled on {enabled}/12 targets and retained novel candidates on {injected}/12 targets.",
        transform=ax2.transAxes,
        fontsize=8.6,
        color=COL["muted"],
        ha="left",
        va="bottom",
    )

    # C: neural decoder utilization
    ax3 = fig.add_subplot(gs[1, 1])
    util = final.sort_values(["neural_decoder_novel_count", "neural_decoder_generated_count"], ascending=[True, False]).reset_index(drop=True)
    y3 = np.arange(len(util))
    ax3.barh(y3, util["decoder_overlap_count"], color=COL["overlap"], height=0.64, label="existing overlap")
    ax3.barh(y3, util["neural_decoder_rejected_count"], left=util["decoder_overlap_count"], color=COL["reject"], height=0.64, label="rejected")
    ax3.barh(
        y3,
        util["neural_decoder_novel_count"],
        left=util["decoder_overlap_count"] + util["neural_decoder_rejected_count"],
        color=COL["decoder"],
        height=0.64,
        label="retained novel",
    )
    ax3.set_yticks(y3)
    ax3.set_yticklabels([cl(v) for v in util["target"]], fontsize=8.0)
    ax3.set_xlabel("Neural decoder preview decomposition")
    ax3.grid(axis="x", color=COL["grid"], lw=0.8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(frameon=False, fontsize=8.2, loc="lower right")
    panel_title(ax3, "C", "Neural decoder utilization")
    for i, row in util.iterrows():
        if int(row["neural_decoder_novel_count"]) > 0:
            ax3.text(
                row["decoder_overlap_count"] + row["neural_decoder_rejected_count"] + row["neural_decoder_novel_count"] + 0.4,
                i,
                str(int(row["neural_decoder_novel_count"])),
                fontsize=8.0,
                color=COL["decoder"],
                va="center",
            )

    # D: family heatmap
    ax4 = fig.add_subplot(gs[0, 1], frame_on=False)
    # create inset heatmap underneath summary cards
    box = gs[0, 1].get_position(fig)
    heat_ax = fig.add_axes([box.x0, box.y0 - 0.03, box.width, box.height * 0.46])
    fam = FAMILY.copy()
    matrix = np.column_stack(
        [
            fam["mean_overall_score"].to_numpy(dtype=float),
            fam["mean_best_learned_score"].to_numpy(dtype=float),
            fam["family_prior_targets"].to_numpy(dtype=float),
        ]
    )
    display = matrix.copy()
    for col_idx in range(display.shape[1]):
        col = display[:, col_idx]
        rng = col.max() - col.min()
        display[:, col_idx] = 0.5 if rng < 1e-9 else (col - col.min()) / rng
    im = heat_ax.imshow(display, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    heat_ax.set_yticks(np.arange(len(fam)))
    heat_ax.set_yticklabels([cl(v) for v in fam["family"]], fontsize=7.6)
    heat_ax.set_xticks([0, 1, 2])
    heat_ax.set_xticklabels(["mean\nfinal", "mean\nlearned", "family\nprior"], fontsize=7.8)
    for i in range(len(fam)):
        for j in range(3):
            val = matrix[i, j]
            heat_ax.text(j, i, f"{val:.1f}" if j < 2 else f"{int(val)}", ha="center", va="center", fontsize=7.3, color="white" if display[i, j] > 0.55 else COL["ink"])
    for spine in heat_ax.spines.values():
        spine.set_color(COL["axis"])
        spine.set_linewidth(0.8)
    heat_ax.set_title("D  Family-level summary", loc="left", fontsize=11, fontweight="bold", pad=8, color=COL["ink"])

    fig.text(0.01, 1.01, "MARS-FIELD benchmark claim figure", fontsize=17, fontweight="bold", ha="left")
    fig.text(
        0.01,
        0.985,
        "The final controller remains panel-stable while enabling decode-time neural generation and retaining novel candidates on a subset of targets.",
        fontsize=10.3,
        ha="left",
        color=COL["muted"],
    )

    fig.savefig(FIG / "figure2_benchmark_claim_v4.svg")
    fig.savefig(FIG / "figure2_benchmark_claim_v4.png", dpi=300)


if __name__ == "__main__":
    render()
