"""
Sequence Logo Visualization Module

Generates WebLogo-style sequence conservation plots based on multiple sequence alignments.
Supports DNA, RNA, and protein sequences with per-position letter stacks showing
relative frequency (height) and conservation scores.

Usage:
    python generate_sequence_logo.py alignments.csv output_dir
    python generate_sequence_logo.py --fasta input.fasta --output logo.png
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# Color schemes following WebLogo convention
# ==============================================================================

DNA_COLORS = {
    "A": "#00DC00",  # Green
    "T": "#DC0000",  # Red
    "G": "#FFC000",  # Gold/Yellow
    "C": "#0000DC",  # Blue
    "U": "#DC0000",  # Red (RNA)
    "N": "#808080",  # Gray (any)
}

RNA_COLORS = {
    "A": "#00DC00",
    "U": "#DC0000",
    "G": "#FFC000",
    "C": "#0000DC",
    "N": "#808080",
}

PROTEIN_COLORS = {
    # Hydrophobic (green)
    "A": "#33FF33", "V": "#33FF33", "I": "#33FF33", "L": "#33FF33",
    "M": "#33CC33", "F": "#33CC33", "W": "#33CC33", "P": "#33CC33",
    # Polar (blue)
    "S": "#6666FF", "T": "#6666FF", "N": "#6666FF", "Q": "#6666FF",
    "Y": "#5555CC", "H": "#5555CC", "C": "#5555CC",
    # Charged positive (red)
    "K": "#FF3333", "R": "#FF3333",
    # Charged negative (magenta)
    "D": "#FF00FF", "E": "#FF00FF",
    # Special (gray)
    "G": "#CC3333",  # special backbone
    "-": "#808080",  # gap
    "X": "#808080",  # unknown
}


def get_color_scheme(sequence_type: str) -> dict:
    """Return color mapping for given sequence type."""
    if sequence_type.lower() in ("dna", "nt", "nucleotide"):
        return DNA_COLORS
    elif sequence_type.lower() in ("rna",):
        return RNA_COLORS
    elif sequence_type.lower() in ("protein", "aa", "amino", "aminoacid"):
        return PROTEIN_COLORS
    else:
        return PROTEIN_COLORS


# ==============================================================================
# Core logo computation
# ==============================================================================

def compute_position_frequencies(sequences: list[str]) -> list[dict[str, float]]:
    """
    Compute per-position amino acid/nucleotide frequencies from aligned sequences.

    Args:
        sequences: List of aligned sequences (all same length)

    Returns:
        List of dicts mapping character to frequency for each position
    """
    if not sequences:
        return []

    seq_len = len(sequences[0])
    counts: list[dict[str, int]] = [defaultdict(int) for _ in range(seq_len)]

    for seq in sequences:
        if len(seq) != seq_len:
            continue
        for i, char in enumerate(seq):
            char_upper = char.upper()
            counts[i][char_upper] += 1

    n = len(sequences)
    frequencies = []
    for pos_counts in counts:
        total = sum(pos_counts.values())
        if total > 0:
            freq = {aa: count / total for aa, count in pos_counts.items()}
        else:
            freq = {}
        frequencies.append(freq)

    return frequencies


def shannon_entropy(frequencies: dict[str, float], alphabet_size: int = 20) -> float:
    """
    Compute Shannon entropy for a position.

    Args:
        frequencies: Frequency dict for a position
        alphabet_size: Size of alphabet (4 for DNA, 20 for proteins)

    Returns:
        Entropy value in bits
    """
    h = 0.0
    for p in frequencies.values():
        if p > 0:
            h -= p * math.log2(p)
    # Normalize by maximum entropy
    h_max = math.log2(alphabet_size)
    return h / h_max if h_max > 0 else 0.0


def compute_conservation(frequencies: dict[str, float]) -> float:
    """
    Compute positional conservation score (1 - entropy_norm).

    Higher values indicate more conserved positions.
    """
    return 1.0 - shannon_entropy(frequencies)


def bits_to_height(p: float, h: float) -> float:
    """
    Convert probability and entropy to letter height in bits.

    Uses WebLogo convention: height = p * (H_max - H)
    """
    h_max = math.log2(len(PROTEIN_COLORS))  # ~4.3 bits for proteins
    return p * (h_max - h) if h_max > 0 else 0


# ==============================================================================
# Logo rendering
# ==============================================================================

def compute_logo_data(
    sequences: list[str],
    sequence_type: str = "protein",
) -> tuple[list[dict[str, float]], list[float]]:
    """
    Compute letter heights and conservation scores for logo generation.

    Args:
        sequences: List of aligned sequences
        sequence_type: One of 'dna', 'rna', 'protein'

    Returns:
        Tuple of (letter_heights, conservation_scores)
    """
    alphabet_size = {"dna": 4, "rna": 4, "protein": 20}.get(sequence_type.lower(), 20)
    frequencies = compute_position_frequencies(sequences)
    conservation = [shannon_entropy(f, alphabet_size) for f in frequencies]

    letter_heights = []
    for i, freq in enumerate(frequencies):
        h = conservation[i]
        heights = {aa: bits_to_height(p, h) for aa, p in freq.items()}
        letter_heights.append(heights)

    return letter_heights, conservation


def render_logo(
    sequences: list[str],
    output_path: Path,
    title: str = "Sequence Logo",
    ylabel: str = "Bits",
    sequence_type: str = "protein",
    width_per_position: float = 0.8,
    height: float = 4.0,
    show_conservation: bool = True,
    show_grid: bool = True,
    colormap: Optional[dict] = None,
    tick_spacing: int = 1,
    start_position: int = 1,
    show_fineprint: bool = True,
    color_scheme: Optional[str] = None,
) -> None:
    """
    Render a WebLogo-style sequence logo.

    Args:
        sequences: List of aligned sequences
        output_path: Path to save the figure
        title: Logo title
        ylabel: Y-axis label
        sequence_type: 'dna', 'rna', or 'protein'
        width_per_position: Width of each position column
        height: Total height of the logo
        show_conservation: Show conservation score track below logo
        show_grid: Show grid lines
        colormap: Custom color mapping
        tick_spacing: Spacing between position tick marks
        start_position: Starting position number (1 for 1-indexed)
        show_fineprint: Show WebLogo attribution text
        color_scheme: Alias for color scheme ('default', 'chemistry', 'mono')
    """
    if not sequences:
        raise ValueError("No sequences provided")

    seq_len = len(sequences[0])
    letter_heights, conservation = compute_logo_data(sequences, sequence_type)

    # Color scheme selection
    if color_scheme == "chemistry":
        colors = get_color_scheme(sequence_type)
    elif color_scheme == "mono":
        colors = {aa: "#333333" for aa in PROTEIN_COLORS}
    else:
        colors = colormap or get_color_scheme(sequence_type)

    # Set up figure
    conservation_height = 0.8 if show_conservation else 0.0
    fig_height = height + conservation_height
    fig_width = seq_len * width_per_position + 1.5

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax_logo = fig.add_axes([0.08, conservation_height / fig_height, 0.90, height / fig_height])

    # Color settings
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })

    # Sort letters by height for each position
    sorted_heights = []
    for pos_heights in letter_heights:
        sorted_heights.append(sorted(pos_heights.items(), key=lambda x: x[1], reverse=True))

    # Draw letter stacks
    h_max = math.log2(20)  # ~4.32 bits for proteins
    y_max = h_max
    y_offset = 0.0

    for pos_idx, letters in enumerate(sorted_heights):
        x_center = pos_idx + 0.5
        cumulative_height = 0.0

        for letter, letter_height in letters:
            if letter_height < 0.01:
                continue

            # Get color
            color = colors.get(letter, "#808080")

            # Draw the letter
            ax_logo.text(
                x_center,
                cumulative_height + letter_height / 2,
                letter,
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="center",
                color=color,
                fontfamily="DejaVu Sans",
            )

            # Draw the stack bar (optional - for debugging/emphasis)
            # ax_logo.bar(x_center, letter_height, bottom=cumulative_height,
            #             width=0.8, color=color, edgecolor='none', alpha=0.7)

            cumulative_height += letter_height

    # Configure axes
    ax_logo.set_xlim(0, seq_len)
    ax_logo.set_ylim(0, y_max)
    ax_logo.set_xticks([])
    ax_logo.set_yticks([0, 1, 2, 3, 4])

    if show_grid:
        ax_logo.grid(axis="y", color="#e2e8f0", linewidth=0.8, zorder=0)
    ax_logo.axhline(0, color="#cbd5e1", linewidth=0.8)
    ax_logo.set_ylabel(ylabel, fontsize=11)

    # Title
    ax_logo.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # X-axis position labels
    x_tick_positions = list(range(0, seq_len, tick_spacing))
    ax_logo.set_xticks(x_tick_positions)
    ax_logo.set_xticklabels([start_position + i for i in x_tick_positions], fontsize=8)

    # Spines
    ax_logo.spines["top"].set_visible(False)
    ax_logo.spines["right"].set_visible(False)
    ax_logo.spines["bottom"].set_visible(False)

    # Conservation track
    if show_conservation:
        ax_cons = fig.add_axes([0.08, 0.04, 0.90, conservation_height / fig_height])
        ax_cons.bar(
            np.arange(seq_len) + 0.5,
            conservation,
            width=0.8,
            color="#64748b",
            alpha=0.6,
        )
        ax_cons.set_xlim(0, seq_len)
        ax_cons.set_ylim(0, 1)
        ax_cons.set_yticks([0, 0.5, 1])
        ax_cons.set_yticklabels(["0", "0.5", "1"], fontsize=7)
        ax_cons.set_xticks(x_tick_positions)
        ax_cons.set_xticklabels([start_position + i for i in x_tick_positions], fontsize=8)
        ax_cons.set_ylabel("IC", fontsize=8)
        ax_cons.spines["top"].set_visible(False)
        ax_cons.spines["right"].set_visible(False)

    # Fineprint
    if show_fineprint:
        fig.text(0.99, 0.01, "Generated by MARS-FIELD Sequence Logo", fontsize=7, ha="right", color="#94a3b8")

    # Save
    fig.savefig(output_path)
    plt.close(fig)


def render_logo_stacked(
    sequences: list[str],
    output_path: Path,
    title: str = "Sequence Logo",
    sequence_type: str = "protein",
    width_per_position: float = 0.6,
    height: float = 5.0,
    show_conservation: bool = True,
    color_scheme: Optional[str] = None,
) -> None:
    """
    Alternative renderer using stacked bars for each position.

    This produces a more traditional WebLogo appearance with filled letter bars.

    Args:
        sequences: List of aligned sequences
        output_path: Path to save figure
        title: Logo title
        sequence_type: 'dna', 'rna', or 'protein'
        width_per_position: Width of each position
        height: Total figure height
        show_conservation: Show information content bar below
        color_scheme: Color scheme ('default', 'chemistry', 'mono')
    """
    if not sequences:
        raise ValueError("No sequences provided")

    seq_len = len(sequences[0])
    letter_heights, conservation = compute_logo_data(sequences, sequence_type)

    colors = get_color_scheme(sequence_type)
    if color_scheme == "mono":
        colors = {aa: "#444444" for aa in colors}

    fig_height = height + (1.0 if show_conservation else 0)
    fig_width = seq_len * width_per_position + 1.0

    fig = plt.figure(figsize=(fig_width, fig_height))

    logo_frac = height / fig_height
    cons_frac = 1.0 / fig_height if show_conservation else 0
    logo_bottom = cons_frac if show_conservation else 0

    ax = fig.add_axes([0.06, logo_bottom + 0.02, 0.92, logo_frac - 0.04])

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "savefig.bbox": "tight",
    })

    h_max = math.log2(20)
    letter_order = list("ACDEFGHIKLMNPQRSTVWY")  # Standard amino acid order

    for pos_idx, pos_heights in enumerate(letter_heights):
        x = pos_idx + 0.5
        y_cumulative = 0.0

        for aa in letter_order:
            if aa in pos_heights and pos_heights[aa] > 0.01:
                h = pos_heights[aa]
                color = colors.get(aa, "#808080")
                ax.bar(x, h, bottom=y_cumulative, width=0.7, color=color, edgecolor="none")

                # Add letter label
                if h > 0.15:
                    ax.text(
                        x,
                        y_cumulative + h / 2,
                        aa,
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="white" if h > 0.3 else "black",
                    )
                y_cumulative += h

    ax.set_xlim(0, seq_len)
    ax.set_ylim(0, h_max)
    ax.set_xticks(range(0, seq_len, 5))
    ax.set_xticklabels([str(i + 1) for i in range(0, seq_len, 5)], fontsize=8)
    ax.set_ylabel("Bits", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_conservation:
        ax_c = fig.add_axes([0.06, 0.02, 0.92, cons_frac - 0.02])
        ax_c.bar(
            np.arange(seq_len) + 0.5,
            conservation,
            width=0.7,
            color="#94a3b8",
            alpha=0.7,
        )
        ax_c.set_xlim(0, seq_len)
        ax_c.set_ylim(0, 1)
        ax_c.set_yticks([0, 0.5, 1])
        ax_c.set_yticklabels(["0", "0.5", "1"], fontsize=7)
        ax_c.set_xticks(range(0, seq_len, 5))
        ax_c.set_xticklabels([str(i + 1) for i in range(0, seq_len, 5)], fontsize=8)
        ax_c.set_ylabel("IC", fontsize=8)
        ax_c.spines["top"].set_visible(False)
        ax_c.spines["right"].set_visible(False)

    fig.savefig(output_path)
    plt.close(fig)


# ==============================================================================
# File I/O utilities
# ==============================================================================

def read_fasta(path: Path) -> list[tuple[str, str]]:
    """
    Read FASTA file and return list of (name, sequence) tuples.

    Args:
        path: Path to FASTA file

    Returns:
        List of (header, sequence) tuples
    """
    sequences = []
    current_name = ""
    current_seq = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name:
                    sequences.append((current_name, "".join(current_seq)))
                current_name = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line)

    if current_name:
        sequences.append((current_name, "".join(current_seq)))

    return sequences


def read_alignment_csv(path: Path, seq_column: str = "sequence", name_column: str = "name") -> list[str]:
    """
    Read multiple sequence alignment from CSV.

    Args:
        path: Path to CSV file
        seq_column: Column name containing sequences
        name_column: Column name for sequence names

    Returns:
        List of sequences
    """
    sequences = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row.get(seq_column, row.get("seq", ""))
            if seq:
                sequences.append(seq)
    return sequences


def detect_sequence_type(sequences: list[str]) -> str:
    """
    Detect sequence type from character composition.

    Returns:
        'dna', 'rna', or 'protein'
    """
    if not sequences:
        return "protein"

    dna_chars = set("ATCG")
    rna_chars = set("AUCG")
    protein_chars = set("ACDEFGHIKLMNPQRSTVWY-")

    sample = "".join(sequences[:10]).upper()

    if set(sample).issubset(dna_chars | {"N"}):
        return "dna"
    elif set(sample).issubset(rna_chars | {"N"}):
        return "rna"
    else:
        return "protein"


# ==============================================================================
# CLI interface
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate WebLogo-style sequence conservation plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --fasta alignment.fasta --output logo.png
  %(prog)s alignment.csv output_dir --format stacked
  %(prog)s --fasta protein_msa.fasta --type protein --title "PF00001 Logo"
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Input file (FASTA, CSV, or directory)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output file or directory path",
    )
    parser.add_argument(
        "--fasta",
        "-f",
        dest="fasta_file",
        help="Input FASTA file",
    )
    parser.add_argument(
        "--csv",
        help="Input CSV file with alignment",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_path",
        help="Output figure path",
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["dna", "rna", "protein", "auto"],
        default="auto",
        help="Sequence type (default: auto-detect)",
    )
    parser.add_argument(
        "--title",
        help="Logo title",
        default="Sequence Logo",
    )
    parser.add_argument(
        "--format",
        choices=["text", "stacked"],
        default="text",
        help="Rendering style",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=5.0,
        help="Logo height in inches",
    )
    parser.add_argument(
        "--width-per-pos",
        type=float,
        default=0.6,
        help="Width per position",
    )
    parser.add_argument(
        "--show-conservation",
        action="store_true",
        default=True,
        help="Show conservation track",
    )
    parser.add_argument(
        "--color-scheme",
        choices=["chemistry", "mono"],
        default="chemistry",
        help="Color scheme",
    )
    parser.add_argument(
        "--seq-column",
        default="sequence",
        help="Column name for sequences in CSV",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI",
    )

    args = parser.parse_args()

    # Determine input source
    sequences = []
    input_path = args.fasta_file or args.csv or args.input

    if not input_path:
        parser.error("No input file specified")

    input_path = Path(input_path)

    if args.fasta_file or input_path.suffix.lower() in (".fasta", ".fa", ".faa", ".fna"):
        fasta_data = read_fasta(input_path)
        sequences = [seq for _, seq in fasta_data]
    elif input_path.suffix.lower() == ".csv":
        sequences = read_alignment_csv(input_path, seq_column=args.seq_column)
    else:
        # Try to detect from content
        if input_path.exists():
            try:
                sequences = read_alignment_csv(input_path)
            except Exception:
                fasta_data = read_fasta(input_path)
                sequences = [seq for _, seq in fasta_data]

    if not sequences:
        parser.error("No sequences found in input")

    # Detect or use specified sequence type
    seq_type = args.type if args.type != "auto" else detect_sequence_type(sequences)

    # Determine output path
    if args.output_path:
        output_path = Path(args.output_path)
    elif args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_logo.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Render
    if args.format == "stacked":
        render_logo_stacked(
            sequences=sequences,
            output_path=output_path,
            title=args.title,
            sequence_type=seq_type,
            width_per_position=args.width_per_pos,
            height=args.height,
            show_conservation=args.show_conservation,
            color_scheme=args.color_scheme,
        )
    else:
        render_logo(
            sequences=sequences,
            output_path=output_path,
            title=args.title,
            sequence_type=seq_type,
            width_per_position=args.width_per_pos,
            height=args.height,
            show_conservation=args.show_conservation,
            color_scheme=args.color_scheme,
        )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
