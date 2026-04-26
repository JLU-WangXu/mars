from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_BUNDLE = ROOT / "outputs" / "paper_bundle_v1"
FIGURES_DIR = PAPER_BUNDLE / "figures"


BENCHMARK_CSV = PAPER_BUNDLE / "figure2_benchmark_overview.csv"
FAMILY_CSV = PAPER_BUNDLE / "figure2_family_summary.csv"
DECODER_CSV = PAPER_BUNDLE / "figure3_decoder_summary.csv"
DERIVED_CSV = PAPER_BUNDLE / "benchmark_derived_metrics.csv"
SUMMARY_MD = PAPER_BUNDLE / "data_figure_summary.md"


COLORS = {
    "navy": "#23364d",
    "slate": "#5b6b7f",
    "blue": "#3b82f6",
    "teal": "#0f766e",
    "amber": "#d97706",
    "orange": "#ea580c",
    "rose": "#e11d48",
    "green": "#15803d",
    "gray": "#cbd5e1",
    "light": "#f8fafc",
    "ink": "#111827",
}

SOURCE_COLORS = {
    "local_proposal": "#334155",
    "esm_if": "#2563eb",
    "mars_mpnn": "#0f766e",
    "fusion_decoder": "#c2410c",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.8,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark = pd.read_csv(BENCHMARK_CSV)
    family = pd.read_csv(FAMILY_CSV)
    decoder = pd.read_csv(DECODER_CSV)
    benchmark["overall_minus_best_learned"] = benchmark["overall_score"] - benchmark["best_learned_score"]
    benchmark["mars_minus_best_learned_mars"] = benchmark["overall_mars_score"] - benchmark["best_learned_mars_score"]
    benchmark["decoder_pressure"] = benchmark["decoder_rejected_count"] - benchmark["decoder_novel_count"]
    benchmark["asr_or_family"] = benchmark["asr_prior_enabled"] | benchmark["family_prior_enabled"]
    benchmark["overall_source_color"] = benchmark["overall_source"].map(SOURCE_COLORS).fillna(COLORS["slate"])
    benchmark["best_learned_source_color"] = benchmark["best_learned_source"].map(SOURCE_COLORS).fillna(COLORS["slate"])
    return benchmark, family, decoder


def write_derived_metrics(benchmark: pd.DataFrame) -> None:
    columns = [
        "target",
        "family",
        "overall_source",
        "overall_score",
        "overall_mars_score",
        "best_learned_source",
        "best_learned_score",
        "best_learned_mars_score",
        "overall_minus_best_learned",
        "mars_minus_best_learned_mars",
        "decoder_novel_count",
        "decoder_rejected_count",
        "decoder_pressure",
        "accepted_homologs",
        "accepted_asr",
        "asr_prior_enabled",
        "family_prior_enabled",
        "family_dataset_id",
    ]
    benchmark[columns].to_csv(DERIVED_CSV, index=False)


def render_figure2(benchmark: pd.DataFrame, family: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.6, 8.8))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.75, 1.0],
        height_ratios=[1.25, 1.0],
        wspace=0.28,
        hspace=0.3,
    )

    # Panel A: target-level overall vs best learned
    ax = fig.add_subplot(gs[:, 0])
    bench = benchmark.sort_values("overall_score", ascending=True).reset_index(drop=True)
    y = np.arange(len(bench))
    ax.barh(
        y,
        bench["overall_score"],
        color=bench["overall_source_color"],
        alpha=0.94,
        height=0.68,
        edgecolor="none",
    )
    ax.scatter(
        bench["best_learned_score"],
        y,
        s=54,
        color=bench["best_learned_source_color"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for idx, row in bench.iterrows():
        flags = []
        if bool(row["asr_prior_enabled"]):
            flags.append("ASR")
        if bool(row["family_prior_enabled"]):
            flags.append("FP")
        flag_text = f" [{' / '.join(flags)}]" if flags else ""
        ax.text(
            row["overall_score"] + 0.08,
            idx,
            f"{row['target']}{flag_text}",
            va="center",
            ha="left",
            fontsize=9.2,
            color=COLORS["ink"],
        )
    ax.axvline(0, color="#94a3b8", lw=0.9)
    ax.set_yticks([])
    ax.set_xlabel("Ranking score")
    ax.set_title("Figure 2A  Twelvepack target-level ranking performance", loc="left", fontweight="bold", pad=12)
    ax.grid(axis="x", color="#e2e8f0", lw=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            mpl.lines.Line2D([0], [0], color=COLORS["navy"], lw=8, label="overall winner"),
            mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["amber"], markeredgecolor="white", markersize=8, label="best learned"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9,
    )

    # Panel B: family means
    ax2 = fig.add_subplot(gs[0, 1])
    fam = family.sort_values("mean_overall_score", ascending=False).reset_index(drop=True)
    x = np.arange(len(fam))
    ax2.bar(
        x,
        fam["mean_overall_score"],
        color=COLORS["blue"],
        alpha=0.88,
        width=0.68,
    )
    ax2.scatter(
        x,
        fam["mean_best_learned_score"],
        color=COLORS["rose"],
        s=44,
        zorder=3,
        edgecolor="white",
        linewidth=0.7,
    )
    for idx, row in fam.iterrows():
        ax2.text(
            idx,
            row["mean_overall_score"] + 0.15,
            str(int(row["target_count"])),
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["slate"],
        )
    ax2.axhline(0, color="#94a3b8", lw=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [name.replace("_", "\n") for name in fam["family"]],
        fontsize=8.2,
    )
    ax2.set_ylabel("Mean score")
    ax2.set_title("Figure 2B  Family-level transfer profile", loc="left", fontweight="bold", pad=12)
    ax2.grid(axis="y", color="#e2e8f0", lw=0.8, alpha=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel C: engineering consistency scatter
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(
        benchmark["overall_score"],
        benchmark["overall_mars_score"],
        s=80,
        color=benchmark["overall_source_color"],
        edgecolor="white",
        linewidth=0.9,
        alpha=0.95,
    )
    for _, row in benchmark.iterrows():
        ax3.text(
            row["overall_score"] + 0.05,
            row["overall_mars_score"] + 0.05,
            row["target"].replace("_", "\n"),
            fontsize=7.8,
            color=COLORS["slate"],
        )
    ax3.axvline(0, color="#94a3b8", lw=0.9)
    ax3.axhline(0, color="#94a3b8", lw=0.9)
    ax3.set_xlabel("Overall ranking score")
    ax3.set_ylabel("Overall MARS score")
    ax3.set_title("Figure 2C  Engineering consistency of final winners", loc="left", fontweight="bold", pad=12)
    ax3.grid(color="#e2e8f0", lw=0.8, alpha=0.8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.suptitle("MARS-FIELD benchmark overview", x=0.07, y=0.995, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.07,
        0.965,
        "Overall winners remain engineering-consistent while the learned branch varies strongly across held-out families.",
        ha="left",
        va="top",
        fontsize=10.5,
        color=COLORS["slate"],
    )

    svg_path = FIGURES_DIR / "figure2_benchmark_overview_v1.svg"
    png_path = FIGURES_DIR / "figure2_benchmark_overview_v1.png"
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def render_figure3(benchmark: pd.DataFrame, decoder: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.6, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)

    # Panel A: decoder novel vs rejected
    ax = fig.add_subplot(gs[0, 0])
    dec = decoder.sort_values("decoder_rejected_count", ascending=True).reset_index(drop=True)
    y = np.arange(len(dec))
    ax.barh(y, dec["decoder_rejected_count"], color=COLORS["gray"], height=0.66, label="rejected")
    ax.barh(y, dec["decoder_novel_count"], color=COLORS["orange"], height=0.66, label="novel injected")
    ax.set_yticks(y)
    ax.set_yticklabels(dec["target"], fontsize=8.7)
    ax.set_xlabel("Candidate count")
    ax.set_title("Figure 3A  Decoder acceptance is tightly gated", loc="left", fontweight="bold", pad=12)
    ax.grid(axis="x", color="#e2e8f0", lw=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    # Panel B: calibration rescue gap
    ax2 = fig.add_subplot(gs[0, 1])
    gap = benchmark.sort_values("overall_minus_best_learned", ascending=False).reset_index(drop=True)
    x = np.arange(len(gap))
    colors = [COLORS["teal"] if value > 0 else COLORS["rose"] for value in gap["overall_minus_best_learned"]]
    ax2.bar(x, gap["overall_minus_best_learned"], color=colors, width=0.72)
    ax2.axhline(0, color="#94a3b8", lw=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(gap["target"], rotation=40, ha="right", fontsize=8.4)
    ax2.set_ylabel("Overall - best learned")
    ax2.set_title("Figure 3B  Calibration rescue gap", loc="left", fontweight="bold", pad=12)
    ax2.grid(axis="y", color="#e2e8f0", lw=0.8, alpha=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel C: decoder score vs final score
    ax3 = fig.add_subplot(gs[1, 0])
    injected = decoder.copy()
    injected["decoder_injected"] = injected["decoder_injected"].astype(bool)
    color_values = injected["decoder_injected"].map({True: COLORS["orange"], False: COLORS["gray"]})
    ax3.scatter(
        injected["best_decoder_ranking_score"],
        injected["overall_score"],
        s=82,
        c=color_values,
        edgecolor="white",
        linewidth=0.9,
    )
    for _, row in injected.iterrows():
        ax3.text(
            row["best_decoder_ranking_score"] + 0.08,
            row["overall_score"] + 0.04,
            row["target"].replace("_", "\n"),
            fontsize=7.6,
            color=COLORS["slate"],
        )
    ax3.axvline(0, color="#94a3b8", lw=0.9)
    ax3.axhline(0, color="#94a3b8", lw=0.9)
    ax3.set_xlabel("Best decoder ranking score")
    ax3.set_ylabel("Final overall score")
    ax3.set_title("Figure 3C  Decoder proposals are not blindly promoted", loc="left", fontweight="bold", pad=12)
    ax3.grid(color="#e2e8f0", lw=0.8, alpha=0.8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Panel D: source summary
    ax4 = fig.add_subplot(gs[1, 1])
    source_order = ["local_proposal", "esm_if", "mars_mpnn", "fusion_decoder"]
    overall_counts = benchmark["overall_source"].value_counts().reindex(source_order, fill_value=0)
    learned_counts = benchmark["best_learned_source"].value_counts().reindex(source_order, fill_value=0)
    x = np.arange(len(source_order))
    width = 0.36
    ax4.bar(x - width / 2, overall_counts.values, width=width, color=COLORS["navy"], label="overall winner")
    ax4.bar(x + width / 2, learned_counts.values, width=width, color=COLORS["amber"], label="best learned")
    ax4.set_xticks(x)
    ax4.set_xticklabels(["local", "esm_if", "mars_mpnn", "decoder"], fontsize=9)
    ax4.set_ylabel("Target count")
    ax4.set_title("Figure 3D  Selection suppresses unstable learned branches", loc="left", fontweight="bold", pad=12)
    ax4.grid(axis="y", color="#e2e8f0", lw=0.8, alpha=0.8)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.legend(frameon=False, fontsize=9)

    fig.suptitle("MARS-FIELD decoder and calibration analysis", x=0.07, y=0.995, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.07,
        0.965,
        "Decoder proposals expand candidate space, but calibration and engineering priors prevent unstable learned winners from taking over the benchmark.",
        ha="left",
        va="top",
        fontsize=10.5,
        color=COLORS["slate"],
    )

    svg_path = FIGURES_DIR / "figure3_decoder_calibration_v1.svg"
    png_path = FIGURES_DIR / "figure3_decoder_calibration_v1.png"
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def write_summary(benchmark: pd.DataFrame, family: pd.DataFrame, decoder: pd.DataFrame) -> None:
    top_gap = benchmark.sort_values("overall_minus_best_learned", ascending=False).iloc[0]
    worst_learned = benchmark.sort_values("best_learned_score").iloc[0]
    asr_targets = benchmark[benchmark["asr_prior_enabled"]]["target"].tolist()
    family_prior_targets = benchmark[benchmark["family_prior_enabled"]]["target"].tolist()
    decoder_injected = decoder[decoder["decoder_injected"]]["target"].tolist()

    lines = [
        "# MARS-FIELD data figure summary",
        "",
        "## Main quantitative observations",
        "",
        f"- Twelvepack targets: {len(benchmark)}",
        f"- Families represented: {len(family)}",
        f"- ASR-active targets: {len(asr_targets)} ({', '.join(asr_targets) if asr_targets else 'NA'})",
        f"- Family-prior targets: {len(family_prior_targets)} ({', '.join(family_prior_targets) if family_prior_targets else 'NA'})",
        f"- Decoder-injected targets: {len(decoder_injected)} ({', '.join(decoder_injected) if decoder_injected else 'NA'})",
        "",
        "## Strongest calibration rescue",
        "",
        f"- Largest overall minus best-learned gap: `{top_gap['target']}` "
        f"({top_gap['overall_minus_best_learned']:.3f})",
        "",
        "## Weakest learned branch",
        "",
        f"- Lowest best-learned score: `{worst_learned['target']}` "
        f"from `{worst_learned['best_learned_source']}` "
        f"({worst_learned['best_learned_score']:.3f})",
        "",
        "## Rendered data figures",
        "",
        f"- Figure 2: `{FIGURES_DIR / 'figure2_benchmark_overview_v1.svg'}`",
        f"- Figure 3: `{FIGURES_DIR / 'figure3_decoder_calibration_v1.svg'}`",
        f"- Derived metrics: `{DERIVED_CSV}`",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    benchmark, family, decoder = load_tables()
    write_derived_metrics(benchmark)
    render_figure2(benchmark, family)
    render_figure3(benchmark, decoder)
    write_summary(benchmark, family, decoder)


if __name__ == "__main__":
    main()
