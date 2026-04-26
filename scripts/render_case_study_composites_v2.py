from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "outputs" / "paper_bundle_v1"
FIG = PAPER / "figures"
STRUCT = PAPER / "structure_panels"

COL = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "overall": "#24364b",
    "learned": "#dc6803",
    "teal": "#0f766e",
    "red": "#e11d48",
}
SRC = {
    "local_proposal": "#334155",
    "esm_if": "#2563eb",
    "mars_mpnn": "#0f766e",
    "fusion_decoder": "#c2410c",
}

CASE_CFG = {
    "1LBT": ("figure4_case_1lbt_v2", "1LBT case study", "Compact three-site oxidation hardening with convergent geometric and learned support."),
    "tem1_1btl": ("figure5_case_tem1_v2", "TEM-1 case study", "Calibration preserves a high-confidence multi-site engineering winner against unstable learned alternatives."),
    "petase_5xh3": ("figure6_case_petase_v2", "PETase family-transfer case study", "Related cutinase backbones show convergent design logic across paired structures."),
    "CLD_3Q09_TOPIC": ("figure7_case_cld_v2", "CLD ancestry-aware case study", "Topic / no-topic comparison exposes the lineage-conditioned branch in an oxidative shell redesign problem."),
}


def style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def t(ax, tag, label):
    ax.text(-0.02, 1.05, tag, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="left", va="bottom", color=COL["ink"])
    ax.text(0.05, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom", color=COL["ink"])


def cl(x):
    return x.replace("_", " ")


def wrap_mutations(text, every=2):
    parts = str(text).split(";")
    rows = [";".join(parts[i:i + every]) for i in range(0, len(parts), every)]
    return "\n".join(rows)


def img(target, kind):
    image = mpimg.imread(STRUCT / target / f"{kind}.png")
    if image.ndim == 3:
        rgb = image[..., :3]
        mask = np.any(rgb < 0.985, axis=2)
        coords = np.argwhere(mask)
        if coords.size:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            pad_y = max(4, int((y1 - y0) * 0.03))
            pad_x = max(4, int((x1 - x0) * 0.03))
            y0 = max(0, y0 - pad_y)
            y1 = min(image.shape[0], y1 + pad_y)
            x0 = max(0, x0 - pad_x)
            x1 = min(image.shape[1], x1 + pad_x)
            return image[y0:y1, x0:x1]
    return image


def candidates(target, n=4):
    path = ROOT / "outputs" / f"{target.lower()}_pipeline" / "combined_ranked_candidates.csv"
    cols = ["source", "mutations", "ranking_score", "mars_score", "score_oxidation", "score_surface", "score_manual", "score_evolution"]
    return pd.read_csv(path)[cols].head(n).copy()


def draw_component(ax, row, tag, title_text):
    vals = [float(row["score_oxidation"]), float(row["score_surface"]), float(row["score_manual"]), float(row["score_evolution"])]
    names = ["oxidation", "surface", "manual", "evolution"]
    ax.barh(range(4), vals, color=[COL["teal"] if v >= 0 else COL["red"] for v in vals], height=0.62)
    ax.axvline(0, color="#cbd5e1", lw=0.9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(names, fontsize=8.4)
    ax.invert_yaxis()
    ax.set_xlabel("Component score")
    ax.grid(axis="x", color=COL["grid"], lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    t(ax, tag, title_text)


def draw_lollipop(ax, df, tag, title_text, group_labels=False):
    work = df.copy().reset_index(drop=True)
    work["color"] = work["source"].map(SRC).fillna(COL["overall"])
    y = np.arange(len(work))[::-1]
    ax.hlines(y, 0, work["ranking_score"], color="#cbd5e1", lw=1.6, zorder=1)
    ax.scatter(work["ranking_score"], y, s=70, color=work["color"], edgecolor="white", linewidth=0.8, zorder=3)
    for yy, (_, row) in zip(y, work.iterrows()):
        label = wrap_mutations(row["mutations"], every=2)
        left = f"{row['source'].replace('_', ' ')}"
        ax.text(-0.02, yy, left, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.0, color=COL["muted"])
        ax.text(row["ranking_score"] + 0.08, yy, label, ha="left", va="center", fontsize=7.7, color=COL["ink"])
    ax.set_yticks([])
    ax.set_xlabel("Ranking score")
    ax.grid(axis="x", color=COL["grid"], lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    t(ax, tag, title_text)


def single_case(target):
    fig_name, title_text, subtitle = CASE_CFG[target]
    df = candidates(target)
    fig = plt.figure(figsize=(12.8, 7.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 0.95, 1.0], height_ratios=[0.95, 1.05], wspace=0.22, hspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(img(target, "overview")); ax1.axis("off"); t(ax1, "A", f"{cl(target)} overview")
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(img(target, "design_window")); ax2.axis("off"); t(ax2, "B", "Design-window close-up")
    ax3 = fig.add_subplot(gs[:, 2]); draw_component(ax3, df.iloc[0], "C", "Score composition of final winner")
    ax4 = fig.add_subplot(gs[1, 0:2]); draw_lollipop(ax4, df, "D", "Top-ranked candidate strip")

    fig.text(0.01, 1.015, title_text, fontsize=17, fontweight="bold", ha="left", color=COL["ink"])
    fig.text(0.01, 0.988, subtitle, fontsize=10.5, ha="left", color=COL["muted"])
    fig.savefig(FIG / f"{fig_name}.svg")
    fig.savefig(FIG / f"{fig_name}.png", dpi=300)


def paired_case(primary, companion):
    fig_name, title_text, subtitle = CASE_CFG[primary]
    p = candidates(primary, 3)
    c = candidates(companion, 3)
    pair = pd.concat(
        [
            p.assign(group=cl(primary)),
            c.assign(group=cl(companion)),
        ],
        ignore_index=True,
    )

    fig = plt.figure(figsize=(13.4, 8.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 0.95, 1.0], height_ratios=[0.95, 1.05], wspace=0.22, hspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(img(primary, "overview")); ax1.axis("off"); t(ax1, "A", f"{cl(primary)} overview")
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(img(companion, "overview")); ax2.axis("off"); t(ax2, "B", f"{cl(companion)} overview")
    ax3 = fig.add_subplot(gs[:, 2]); draw_component(ax3, p.iloc[0], "C", "Primary winner score composition")
    ax4 = fig.add_subplot(gs[1, 0]); ax4.imshow(img(primary, "design_window")); ax4.axis("off"); t(ax4, "D", f"{cl(primary)} design window")
    ax5 = fig.add_subplot(gs[1, 1]); ax5.imshow(img(companion, "design_window")); ax5.axis("off"); t(ax5, "E", f"{cl(companion)} design window")

    ax6 = fig.add_axes([0.08, 0.08, 0.55, 0.22])
    pair["label"] = pair["group"] + " | " + pair["source"].str.replace("_", " ", regex=False)
    draw_lollipop(ax6, pair[["source", "mutations", "ranking_score"]].assign(source=pair["label"]), "F", "Paired candidate strip")

    fig.text(0.01, 1.015, title_text, fontsize=17, fontweight="bold", ha="left", color=COL["ink"])
    fig.text(0.01, 0.988, subtitle, fontsize=10.5, ha="left", color=COL["muted"])
    fig.savefig(FIG / f"{fig_name}.svg")
    fig.savefig(FIG / f"{fig_name}.png", dpi=300)


def main():
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    single_case("1LBT")
    single_case("tem1_1btl")
    paired_case("petase_5xh3", "petase_5xfy")
    paired_case("CLD_3Q09_TOPIC", "CLD_3Q09_NOTOPIC")


if __name__ == "__main__":
    main()
