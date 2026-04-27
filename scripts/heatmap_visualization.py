"""
Heatmap Visualization Tools for MARS Analysis Results

Provides comprehensive heatmap visualizations including:
- Score heatmaps for ranking analysis
- Mutation effect heatmaps
- Structure alignment heatmaps
- Interactive visualizations with plotly

Author: MARS-FIELD Team
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "heatmap_visualization"
FIG_DIR = OUT_ROOT / "figures"
HTML_DIR = OUT_ROOT / "html"

# Color schemes
SCORE_COLORSCHEME = "RdYlGn"
MUTATION_COLORSCHEME = "RdBu_r"
ALIGNMENT_COLORSCHEME = "viridis"

# Amino acid list for mutation heatmaps
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# Color palette
COL = {
    "ink": "#111827",
    "muted": "#667085",
    "grid": "#e5e7eb",
    "axis": "#cbd5e1",
    "high": "#059669",
    "medium": "#f59e0b",
    "low": "#dc2626",
    "stable": "#059669",
    "unstable": "#dc2626",
    "neutral": "#6b7280",
    "pareto": "#2563eb",
    "mut_effect": "#f97316",
}


def style() -> None:
    """Configure matplotlib styling for consistent appearance."""
    mpl.rcParams.update({
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
    })


def ensure_dirs() -> None:
    """Create output directories."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if PLOTLY_AVAILABLE:
        HTML_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Score Heatmap Generation
# =============================================================================

def generate_score_heatmap(
    scores: pd.DataFrame,
    title: str = "Score Heatmap",
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 8),
    annot: bool = True,
    fmt: str = ".2f",
    cmap: str = SCORE_COLORSCHEME,
    center: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    interactive: bool = False,
) -> plt.Figure:
    """
    Generate a heatmap visualization for scoring analysis results.

    Args:
        scores: DataFrame with candidates as rows and metrics as columns.
        title: Title for the heatmap.
        output_path: Path to save the figure. If None, figure is not saved.
        figsize: Figure size as (width, height).
        annot: Whether to annotate cells with values.
        fmt: Format string for annotations.
        cmap: Colormap name.
        center: Center value for colormap normalization.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        interactive: Whether to generate interactive plotly version.

    Returns:
        matplotlib Figure object.
    """
    style()
    ensure_dirs()

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize scores per column for better visualization
    scores_normalized = scores.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10))

    sns.heatmap(
        scores_normalized,
        annot=scores if annot else None,
        fmt=fmt,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor=COL["grid"],
        cbar_kws={"label": "Normalized Score", "shrink": 0.8},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Metrics", fontsize=11, labelpad=10)
    ax.set_ylabel("Candidates", fontsize=11, labelpad=10)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Score heatmap saved to: {output_path}")

    if interactive and PLOTLY_AVAILABLE:
        interactive_path = HTML_DIR / f"{output_path.stem}_interactive.html" if output_path else None
        generate_interactive_score_heatmap(scores, title, interactive_path)

    return fig


def generate_score_comparison_heatmap(
    scores_dict: dict[str, pd.DataFrame],
    candidates: list[str],
    metrics: list[str],
    title: str = "Score Comparison Across Methods",
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Generate a heatmap comparing scores across multiple methods.

    Args:
        scores_dict: Dictionary mapping method names to score DataFrames.
        candidates: List of candidate names to include.
        metrics: List of metrics to compare.
        title: Title for the heatmap.
        output_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    style()
    ensure_dirs()

    n_methods = len(scores_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize, sharey=True)

    if n_methods == 1:
        axes = [axes]

    for ax, (method, scores) in zip(axes, scores_dict.items()):
        # Filter to specified candidates and metrics
        scores_filtered = scores.loc[candidates, metrics] if all(
            c in scores.index for c in candidates
        ) else scores[metrics].head(len(candidates))

        # Normalize
        scores_norm = scores_filtered.apply(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
        )

        sns.heatmap(
            scores_norm,
            cmap=SCORE_COLORSCHEME,
            annot=scores_filtered if len(metrics) <= 5 else None,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title(f"{method}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Metrics", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Candidates", fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Score comparison heatmap saved to: {output_path}")

    return fig


# =============================================================================
# Mutation Effect Heatmap
# =============================================================================

def generate_mutation_effect_heatmap(
    mutation_scores: pd.DataFrame,
    positions: list[int],
    title: str = "Mutation Effect Heatmap",
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 8),
    annot: bool = True,
    cmap: str = MUTATION_COLORSCHEME,
    center: float = 0.0,
    interactive: bool = False,
) -> plt.Figure:
    """
    Generate a heatmap showing the effect of mutations at each position.

    Args:
        mutation_scores: DataFrame with positions as index and amino acids as columns.
                        Values represent the effect score (e.g., delta thermostability).
        positions: List of positions to visualize.
        title: Title for the heatmap.
        output_path: Path to save the figure.
        figsize: Figure size.
        annot: Whether to annotate cells.
        cmap: Colormap.
        center: Center value for colormap (typically 0 for delta values).
        interactive: Whether to generate interactive version.

    Returns:
        matplotlib Figure object.
    """
    style()
    ensure_dirs()

    # Filter to specified positions
    mutation_filtered = mutation_scores.loc[
        mutation_scores.index.isin(positions), AA_LIST
    ]

    fig, ax = plt.subplots(figsize=figsize)

    im = sns.heatmap(
        mutation_filtered,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        center=center,
        linewidths=0.5,
        linecolor=COL["grid"],
        cbar_kws={"label": "Effect Score (Δ thermostability)", "shrink": 0.8},
        ax=ax,
        annot_kws={"size": 8},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Mutant Amino Acid", fontsize=11, labelpad=10)
    ax.set_ylabel("Position", fontsize=11, labelpad=10)

    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Mutation effect heatmap saved to: {output_path}")

    if interactive and PLOTLY_AVAILABLE:
        interactive_path = HTML_DIR / f"{output_path.stem}_interactive.html" if output_path else None
        generate_interactive_mutation_heatmap(mutation_scores, title, interactive_path)

    return fig


def generate_mutation_matrix_heatmap(
    wildtype_seq: str,
    mutation_data: dict[tuple[int, str], float],
    title: str = "Single Mutant Scan",
    output_path: Optional[Path] = None,
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """
    Generate a heatmap for single mutant scan results.

    Args:
        wildtype_seq: Wild-type amino acid sequence.
        mutation_data: Dictionary mapping (position, mutant_aa) to effect score.
        title: Title for the heatmap.
        output_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    style()
    ensure_dirs()

    n_positions = len(wildtype_seq)
    matrix = np.full((n_positions, 20), np.nan)

    for (pos, aa), score in mutation_data.items():
        if 0 <= pos < n_positions:
            aa_idx = AA_LIST.index(aa) if aa in AA_LIST else -1
            if aa_idx >= 0:
                matrix[pos, aa_idx] = score

    df_matrix = pd.DataFrame(
        matrix,
        index=[f"{i+1}\n{wt}" for i, wt in enumerate(wildtype_seq)],
        columns=AA_LIST,
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df_matrix,
        annot=True,
        fmt=".2f",
        cmap=MUTATION_COLORSCHEME,
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Effect Score", "shrink": 0.8},
        ax=ax,
        annot_kws={"size": 7},
        mask=df_matrix.isna(),
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Mutant Amino Acid", fontsize=11, labelpad=10)
    ax.set_ylabel("Position (WT)", fontsize=11, labelpad=10)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Mutation matrix heatmap saved to: {output_path}")

    return fig


# =============================================================================
# Structure Alignment Heatmap
# =============================================================================

def generate_alignment_heatmap(
    alignment_matrix: np.ndarray,
    labels: Optional[list[str]] = None,
    title: str = "Structure Alignment Heatmap",
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 10),
    annot: bool = True,
    cmap: str = ALIGNMENT_COLORSCHEME,
    interactive: bool = False,
) -> plt.Figure:
    """
    Generate a heatmap for structure alignment similarity matrix.

    Args:
        alignment_matrix: 2D numpy array of alignment scores (RMSD, TM-score, etc.).
        labels: Optional labels for rows and columns.
        title: Title for the heatmap.
        output_path: Path to save the figure.
        figsize: Figure size.
        annot: Whether to annotate cells.
        cmap: Colormap.
        interactive: Whether to generate interactive version.

    Returns:
        matplotlib Figure object.
    """
    style()
    ensure_dirs()

    if labels:
        df_alignment = pd.DataFrame(alignment_matrix, index=labels, columns=labels)
    else:
        df_alignment = pd.DataFrame(alignment_matrix)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df_alignment,
        annot=annot,
        fmt=".3f" if alignment_matrix.max() < 1 else ".2f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"label": "Similarity Score", "shrink": 0.8},
        ax=ax,
        annot_kws={"size": 9},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Alignment heatmap saved to: {output_path}")

    if interactive and PLOTLY_AVAILABLE:
        interactive_path = HTML_DIR / f"{output_path.stem}_interactive.html" if output_path else None
        generate_interactive_alignment_heatmap(alignment_matrix, labels, title, interactive_path)

    return fig


def generate_contact_map_heatmap(
    contact_map: np.ndarray,
    sequence: Optional[str] = None,
    title: str = "Contact Map",
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 10),
    threshold: float = 8.0,
) -> plt.Figure:
    """
    Generate a heatmap for protein contact map.

    Args:
        contact_map: 2D numpy array of distances between residues.
        sequence: Optional amino acid sequence for labeling.
        title: Title for the heatmap.
        output_path: Path to save the figure.
        figsize: Figure size.
        threshold: Distance threshold for contacts (Angstroms).

    Returns:
        matplotlib Figure object.
    """
    style()
    ensure_dirs()

    # Binarize contact map
    contacts = (contact_map <= threshold).astype(float)
    contacts[contact_map == 0] = np.nan  # Mask diagonal

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        contacts,
        cmap="Blues",
        linewidths=0,
        cbar_kws={"label": "Contact (1) / No Contact (0)", "shrink": 0.8},
        ax=ax,
        vmin=0,
        vmax=1,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Residue Position", fontsize=11, labelpad=10)
    ax.set_ylabel("Residue Position", fontsize=11, labelpad=10)

    if sequence:
        n = len(sequence)
        tick_positions = np.linspace(0, n-1, min(n, 20)).astype(int)
        tick_labels = [sequence[i] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Contact map heatmap saved to: {output_path}")

    return fig


# =============================================================================
# Interactive Visualizations with Plotly
# =============================================================================

def generate_interactive_score_heatmap(
    scores: pd.DataFrame,
    title: str = "Score Heatmap",
    output_path: Optional[Path] = None,
) -> go.Figure:
    """
    Generate interactive score heatmap using plotly.

    Args:
        scores: DataFrame with scores.
        title: Title for the plot.
        output_path: Path to save HTML file.

    Returns:
        plotly Figure object.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    ensure_dirs()

    # Normalize scores
    scores_norm = scores.apply(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    )

    fig = go.Figure(data=go.Heatmap(
        z=scores_norm.values,
        x=scores.columns.tolist(),
        y=scores.index.tolist(),
        colorscale="RdYlGn",
        text=scores.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 10},
        hovertemplate="Candidate: %{y}<br>Metric: %{x}<br>Score: %{text:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 16}},
        xaxis_title="Metrics",
        yaxis_title="Candidates",
        width=900,
        height=600,
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Interactive score heatmap saved to: {output_path}")

    return fig


def generate_interactive_mutation_heatmap(
    mutation_scores: pd.DataFrame,
    title: str = "Mutation Effect",
    output_path: Optional[Path] = None,
) -> go.Figure:
    """
    Generate interactive mutation effect heatmap using plotly.

    Args:
        mutation_scores: DataFrame with mutation effects.
        title: Title for the plot.
        output_path: Path to save HTML file.

    Returns:
        plotly Figure object.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    ensure_dirs()

    fig = go.Figure(data=go.Heatmap(
        z=mutation_scores.values,
        x=mutation_scores.columns.tolist(),
        y=[str(i) for i in mutation_scores.index],
        colorscale="RdBu_r",
        zmid=0,
        text=mutation_scores.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 8},
        hovertemplate="Position: %{y}<br>Mutant: %{x}<br>Effect: %{text:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 16}},
        xaxis_title="Mutant Amino Acid",
        yaxis_title="Position",
        width=1000,
        height=600,
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Interactive mutation heatmap saved to: {output_path}")

    return fig


def generate_interactive_alignment_heatmap(
    alignment_matrix: np.ndarray,
    labels: Optional[list[str]] = None,
    title: str = "Structure Alignment",
    output_path: Optional[Path] = None,
) -> go.Figure:
    """
    Generate interactive alignment heatmap using plotly.

    Args:
        alignment_matrix: 2D numpy array of alignment scores.
        labels: Optional labels for structures.
        title: Title for the plot.
        output_path: Path to save HTML file.

    Returns:
        plotly Figure object.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    ensure_dirs()

    x_labels = labels if labels else [f"Struct_{i}" for i in range(len(alignment_matrix))]
    y_labels = labels if labels else [f"Struct_{i}" for i in range(len(alignment_matrix))]

    fig = go.Figure(data=go.Heatmap(
        z=alignment_matrix,
        x=x_labels,
        y=y_labels,
        colorscale="Viridis",
        text=alignment_matrix,
        texttemplate="%{text:.3f}",
        textfont={"size": 9},
        hovertemplate="Structure X: %{x}<br>Structure Y: %{y}<br>Score: %{text:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 16}},
        xaxis_title="Structure",
        yaxis_title="Structure",
        width=800,
        height=800,
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Interactive alignment heatmap saved to: {output_path}")

    return fig


def generate_interactive_clustered_heatmap(
    data: pd.DataFrame,
    title: str = "Clustered Heatmap",
    output_path: Optional[Path] = None,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
) -> go.Figure:
    """
    Generate interactive clustered heatmap with dendrograms.

    Args:
        data: DataFrame to visualize.
        title: Title for the plot.
        output_path: Path to save HTML file.
        cluster_rows: Whether to cluster rows.
        cluster_cols: Whether to cluster columns.

    Returns:
        plotly Figure object.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    ensure_dirs()

    # Normalize data
    data_norm = data.apply(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    )

    fig = go.Figure(data=go.Heatmap(
        z=data_norm.values,
        x=data.columns.tolist(),
        y=[str(i) for i in data.index],
        colorscale="Plasma",
        text=data.values,
        texttemplate="%{text:.2f}",
        hovertemplate="%{y} | %{x}: %{text:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 16}},
        width=900,
        height=700,
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Interactive clustered heatmap saved to: {output_path}")

    return fig


# =============================================================================
# Combined Analysis Dashboard
# =============================================================================

def generate_analysis_dashboard(
    score_data: Optional[pd.DataFrame] = None,
    mutation_data: Optional[dict] = None,
    alignment_data: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """
    Generate a combined analysis dashboard with multiple heatmap panels.

    Args:
        score_data: Optional score DataFrame.
        mutation_data: Optional mutation data dictionary.
        alignment_data: Optional alignment matrix.
        output_dir: Directory to save dashboard components.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    style()
    ensure_dirs()

    n_panels = sum(x is not None for x in [score_data, mutation_data, alignment_data])
    if n_panels == 0:
        raise ValueError("At least one data source must be provided")

    if n_panels == 1:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    elif n_panels == 2:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

    panel_idx = 0

    if score_data is not None:
        scores_norm = score_data.apply(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
        )
        sns.heatmap(
            scores_norm,
            annot=len(score_data.columns) <= 5,
            fmt=".2f",
            cmap=SCORE_COLORSCHEME,
            linewidths=0.5,
            ax=axes[panel_idx],
            cbar_kws={"shrink": 0.8},
        )
        axes[panel_idx].set_title("Score Analysis", fontsize=12, fontweight="bold")
        axes[panel_idx].set_xlabel("Metrics")
        axes[panel_idx].set_ylabel("Candidates")
        panel_idx += 1

    if mutation_data is not None:
        # Convert mutation data to matrix format
        if isinstance(mutation_data, dict):
            positions = sorted(set(k[0] for k in mutation_data.keys()))
            matrix = np.array([
                [mutation_data.get((p, aa), 0) for aa in AA_LIST]
                for p in positions
            ])
            mutation_df = pd.DataFrame(matrix, index=positions, columns=AA_LIST)
        else:
            mutation_df = mutation_data

        sns.heatmap(
            mutation_df,
            cmap=MUTATION_COLORSCHEME,
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            ax=axes[panel_idx],
            cbar_kws={"shrink": 0.8},
        )
        axes[panel_idx].set_title("Mutation Effects", fontsize=12, fontweight="bold")
        axes[panel_idx].set_xlabel("Mutant AA")
        axes[panel_idx].set_ylabel("Position")
        panel_idx += 1

    if alignment_data is not None:
        sns.heatmap(
            alignment_data,
            cmap=ALIGNMENT_COLORSCHEME,
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            square=True,
            ax=axes[panel_idx],
            cbar_kws={"shrink": 0.8},
        )
        axes[panel_idx].set_title("Structure Alignment", fontsize=12, fontweight="bold")
        axes[panel_idx].set_xlabel("Structure")
        axes[panel_idx].set_ylabel("Structure")
        panel_idx += 1

    # Hide unused axes
    for i in range(panel_idx, len(axes)):
        axes[i].axis("off")

    fig.suptitle("MARS Analysis Dashboard", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_dir:
        output_path = output_dir / "analysis_dashboard.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Analysis dashboard saved to: {output_path}")

    return fig


# =============================================================================
# Demo / Example Usage
# =============================================================================

def generate_demo_data():
    """Generate synthetic demo data for testing visualizations."""
    np.random.seed(42)

    # Score data
    candidates = [f"Candidate_{i:03d}" for i in range(20)]
    metrics = ["Thermostability", "Expression", "Activity", "Selectivity", "Solubility"]
    scores = pd.DataFrame(
        np.random.uniform(0.3, 0.95, size=(20, 5)),
        index=candidates,
        columns=metrics,
    )

    # Mutation data
    positions = list(range(50, 80))
    mutation_scores = pd.DataFrame(
        np.random.normal(0, 0.5, size=(len(positions), 20)),
        index=positions,
        columns=AA_LIST,
    )

    # Alignment data
    n_structures = 8
    structures = [f"Struct_{i}" for i in range(n_structures)]
    alignment_matrix = np.random.uniform(0.5, 1.0, size=(n_structures, n_structures))
    alignment_matrix = (alignment_matrix + alignment_matrix.T) / 2
    np.fill_diagonal(alignment_matrix, 1.0)

    return scores, mutation_scores, alignment_matrix, structures


def main():
    """Demo main function to showcase visualization capabilities."""
    print("Generating heatmap visualizations...")

    # Generate demo data
    scores, mutation_scores, alignment_matrix, structures = generate_demo_data()

    # Ensure output directories
    ensure_dirs()

    # 1. Score heatmap
    print("\n1. Generating score heatmap...")
    score_fig = generate_score_heatmap(
        scores,
        title="Candidate Ranking Scores",
        output_path=FIG_DIR / "score_heatmap.png",
        interactive=True,
    )

    # 2. Mutation effect heatmap
    print("2. Generating mutation effect heatmap...")
    positions = list(range(50, 70))
    mut_fig = generate_mutation_effect_heatmap(
        mutation_scores,
        positions=positions,
        title="Mutation Effect on Thermostability",
        output_path=FIG_DIR / "mutation_effect_heatmap.png",
        interactive=True,
    )

    # 3. Structure alignment heatmap
    print("3. Generating structure alignment heatmap...")
    align_fig = generate_alignment_heatmap(
        alignment_matrix,
        labels=structures,
        title="Structural Similarity Matrix (TM-score)",
        output_path=FIG_DIR / "alignment_heatmap.png",
        interactive=True,
    )

    # 4. Combined dashboard
    print("4. Generating analysis dashboard...")
    mutation_dict = {
        (pos, aa): mutation_scores.loc[pos, aa]
        for pos in positions[:10]
        for aa in AA_LIST
    }
    dashboard_fig = generate_analysis_dashboard(
        score_data=scores.head(10),
        mutation_data=mutation_dict,
        alignment_data=alignment_matrix,
        output_dir=FIG_DIR,
    )

    print(f"\nVisualizations saved to: {FIG_DIR}")
    if PLOTLY_AVAILABLE:
        print(f"Interactive HTML files saved to: {HTML_DIR}")

    plt.show()


if __name__ == "__main__":
    main()
