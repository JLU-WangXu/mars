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
DEC = pd.read_csv(PAPER / "figure3_decoder_summary.csv")

COL = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "axis": "#cbd5e1",
    "overall": "#24364b",
    "learned": "#dc6803",
    "teal": "#0f766e",
    "blue": "#2563eb",
    "decoder": "#ea580c",
    "rose": "#e11d48",
    "fill": "#f8fafc",
}
SRC = {
    "local_proposal": "#334155",
    "esm_if": "#2563eb",
    "mars_mpnn": "#0f766e",
    "fusion_decoder": "#c2410c",
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


def render_fig2():
    df = BENCH.copy()
    df["gap"] = df["overall_score"] - df["best_learned_score"]
    df["src_c"] = df["overall_source"].map(SRC).fillna(COL["overall"])
    df["learn_c"] = df["best_learned_source"].map(SRC).fillna(COL["learned"])
    fig = plt.figure(figsize=(13.2, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.32)

    ax = fig.add_subplot(gs[:, 0])
    plot = df.sort_values("overall_score").reset_index(drop=True)
    y = np.arange(len(plot))
    ax.hlines(y, plot["best_learned_score"], plot["overall_score"], color=COL["axis"], lw=2.0)
    ax.scatter(plot["best_learned_score"], y, s=52, color=COL["learned"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(plot["overall_score"], y, s=58, color=COL["overall"], edgecolor="white", linewidth=0.8, zorder=4)
    for i, row in plot.iterrows():
        flags = []
        if bool(row["asr_prior_enabled"]):
            flags.append("ASR")
        if bool(row["family_prior_enabled"]):
            flags.append("FP")
        suffix = f" [{' / '.join(flags)}]" if flags else ""
        ax.text(row["overall_score"] + 0.18, i, f"{cl(row['target'])}{suffix}", va="center", fontsize=8.6)
    ax.axvline(0, color=COL["axis"], lw=0.9)
    ax.set_yticks([])
    ax.set_xlabel("Ranking score")
    ax.grid(axis="x", color=COL["grid"], lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    title(ax, "A", "Target-level winner stabilization")

    ax2 = fig.add_subplot(gs[0, 1])
    fam = FAM.sort_values("mean_overall_score", ascending=False).reset_index(drop=True)
    x = np.arange(len(fam))
    ax2.bar(x, fam["mean_overall_score"], color=COL["blue"], width=0.64, alpha=0.88)
    ax2.scatter(x, fam["mean_best_learned_score"], s=44, color=COL["rose"], edgecolor="white", linewidth=0.8, zorder=3)
    ax2.axhline(0, color=COL["axis"], lw=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([v.replace("_", "\n") for v in fam["family"]], fontsize=7.8)
    ax2.set_ylabel("Mean score")
    ax2.grid(axis="y", color=COL["grid"], lw=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    title(ax2, "B", "Family-level transfer profile")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(df["overall_score"], df["overall_mars_score"], s=70, c=df["src_c"], edgecolor="white", linewidth=0.8)
    for _, row in df.iterrows():
        ax3.text(row["overall_score"] + 0.05, row["overall_mars_score"] + 0.05, cl(row["target"]).replace(" ", "\n"), fontsize=7.1, color=COL["muted"])
    ax3.axvline(0, color=COL["axis"], lw=0.9)
    ax3.axhline(0, color=COL["axis"], lw=0.9)
    ax3.set_xlabel("Overall ranking score")
    ax3.set_ylabel("Overall MARS score")
    ax3.grid(color=COL["grid"], lw=0.8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    title(ax3, "C", "Engineering consistency of final winners")

    fig.text(0.01, 1.015, "MARS-FIELD benchmark overview", fontsize=17, fontweight="bold", ha="left")
    fig.text(0.01, 0.988, "Final winners remain engineering-consistent while the learned branch varies sharply across held-out families.", fontsize=10.5, ha="left", color=COL["muted"])
    fig.savefig(FIG / "figure2_benchmark_overview_v2.svg")
    fig.savefig(FIG / "figure2_benchmark_overview_v2.png", dpi=300)
    plt.close(fig)


def render_fig3():
    df = BENCH.copy()
    df["gap"] = df["overall_score"] - df["best_learned_score"]
    fig = plt.figure(figsize=(13.2, 8.2))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.34)

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

    ax3 = fig.add_subplot(gs[1, 0])
    injected = DEC.copy()
    injected["decoder_injected"] = injected["decoder_injected"].astype(bool)
    ax3.scatter(injected["best_decoder_ranking_score"], injected["overall_score"], s=72, c=injected["decoder_injected"].map({True: COL["decoder"], False: "#cbd5e1"}), edgecolor="white", linewidth=0.8)
    for _, row in injected.iterrows():
        ax3.text(row["best_decoder_ranking_score"] + 0.06, row["overall_score"] + 0.03, cl(row["target"]).replace(" ", "\n"), fontsize=7.0, color=COL["muted"])
    ax3.axvline(0, color=COL["axis"], lw=0.9)
    ax3.axhline(0, color=COL["axis"], lw=0.9)
    ax3.set_xlabel("Best decoder ranking score")
    ax3.set_ylabel("Final overall score")
    ax3.grid(color=COL["grid"], lw=0.8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    title(ax3, "C", "Decoder proposals are not blindly promoted")

    ax4 = fig.add_subplot(gs[1, 1])
    order = ["local_proposal", "esm_if", "mars_mpnn", "fusion_decoder"]
    ov = df["overall_source"].value_counts().reindex(order, fill_value=0)
    bl = df["best_learned_source"].value_counts().reindex(order, fill_value=0)
    x2 = np.arange(len(order))
    ax4.bar(x2 - 0.16, ov.values, width=0.32, color=COL["overall"], label="overall")
    ax4.bar(x2 + 0.16, bl.values, width=0.32, color=COL["learned"], label="best learned")
    ax4.set_xticks(x2)
    ax4.set_xticklabels(["local", "esm_if", "mars_mpnn", "decoder"], fontsize=8.8)
    ax4.set_ylabel("Target count")
    ax4.grid(axis="y", color=COL["grid"], lw=0.8)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.legend(frameon=False, fontsize=9)
    title(ax4, "D", "Selection suppresses unstable learned branches")

    fig.text(0.01, 1.015, "MARS-FIELD decoder and calibration analysis", fontsize=17, fontweight="bold", ha="left")
    fig.text(0.01, 0.988, "Decoder proposals enlarge the search space, but calibration and engineering priors prevent unstable learned winners from taking over the benchmark.", fontsize=10.5, ha="left", color=COL["muted"])
    fig.savefig(FIG / "figure3_decoder_calibration_v2.svg")
    fig.savefig(FIG / "figure3_decoder_calibration_v2.png", dpi=300)
    plt.close(fig)


def main():
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    BENCH.assign(
        overall_minus_best_learned=BENCH["overall_score"] - BENCH["best_learned_score"],
        mars_minus_best_learned_mars=BENCH["overall_mars_score"] - BENCH["best_learned_mars_score"],
        decoder_pressure=BENCH["decoder_rejected_count"] - BENCH["decoder_novel_count"],
    ).to_csv(PAPER / "benchmark_derived_metrics.csv", index=False)
    render_fig2()
    render_fig3()
    summary = [
        "# MARS-FIELD data figure summary",
        "",
        f"- Twelvepack targets: {len(BENCH)}",
        f"- Families represented: {len(FAM)}",
        f"- Decoder-injected targets: {int(DEC['decoder_injected'].sum())}",
        f"- Figure 2 v2: `{FIG / 'figure2_benchmark_overview_v2.svg'}`",
        f"- Figure 3 v2: `{FIG / 'figure3_decoder_calibration_v2.svg'}`",
        f"- Derived metrics: `{PAPER / 'benchmark_derived_metrics.csv'}`",
    ]
    (PAPER / "data_figure_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
