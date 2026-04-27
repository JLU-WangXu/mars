"""Diversity metrics for protein sequence design evaluation and enhancement.

This module provides metrics for measuring and promoting diversity in beam search
decoding, supporting exploration-exploitation balance in protein sequence generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# Standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class DiversityMetrics:
    """Container for computed diversity metrics."""

    mean_hamming_distance: float = 0.0
    min_hamming_distance: float = 0.0
    shannon_entropy: float = 0.0
    positional_entropy: float = 0.0
    mutation_spectrum_diversity: float = 0.0
    structural_diversity_estimate: float = 0.0
    exploration_score: float = 0.0
    exploitation_score: float = 0.0
    balance_ratio: float = 0.5


@dataclass
class DiversityConfig:
    """Configuration for diversity-aware decoding."""

    diversity_penalty: float = 0.05
    hamming_weight: float = 0.4
    entropy_weight: float = 0.3
    spectrum_weight: float = 0.3
    min_diversity_threshold: float = 0.2
    max_diversity_threshold: float = 0.8
    exploration_bias: float = 0.1
    adaptive_penalty: bool = True


def compute_hamming_distances(sequences: list[str]) -> list[float]:
    """Compute pairwise Hamming distances between sequences.

    Args:
        sequences: List of protein sequences

    Returns:
        List of pairwise Hamming distances
    """
    if len(sequences) < 2:
        return []
    distances = []
    n = len(sequences)
    for i in range(n):
        for j in range(i + 1, n):
            dist = sum(
                1 for a, b in zip(sequences[i], sequences[j]) if a != b
            )
            distances.append(dist)
    return distances


def compute_mean_hamming_distance(sequences: list[str]) -> float:
    """Compute mean pairwise Hamming distance.

    Args:
        sequences: List of protein sequences

    Returns:
        Mean Hamming distance
    """
    distances = compute_hamming_distances(sequences)
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def compute_min_hamming_distance(sequences: list[str]) -> float:
    """Compute minimum pairwise Hamming distance.

    Args:
        sequences: List of protein sequences

    Returns:
        Minimum Hamming distance
    """
    distances = compute_hamming_distances(sequences)
    if not distances:
        return 0.0
    return min(distances)


def compute_shannon_entropy(sequences: list[str]) -> float:
    """Compute Shannon entropy of residue distribution across sequences.

    Args:
        sequences: List of protein sequences

    Returns:
        Shannon entropy in bits
    """
    if not sequences:
        return 0.0

    seq_len = len(sequences[0])
    total_residues = len(sequences) * seq_len

    entropy = 0.0
    for pos in range(seq_len):
        counts = {aa: 0 for aa in AMINO_ACIDS}
        for seq in sequences:
            if pos < len(seq):
                aa = seq[pos]
                if aa in counts:
                    counts[aa] += 1

        for count in counts.values():
            if count > 0:
                p = count / len(sequences)
                entropy -= (p * math.log2(p))

    return entropy / seq_len if seq_len > 0 else 0.0


def compute_positional_entropy(sequences: list[str]) -> float:
    """Compute positional entropy - measures variation at each position.

    Args:
        sequences: List of protein sequences

    Returns:
        Average positional entropy
    """
    if not sequences:
        return 0.0

    seq_len = len(sequences[0])
    total_entropy = 0.0

    for pos in range(seq_len):
        counts = {}
        for seq in sequences:
            if pos < len(seq):
                aa = seq[pos]
                counts[aa] = counts.get(aa, 0) + 1

        pos_entropy = 0.0
        n = len(sequences)
        for count in counts.values():
            if count > 0:
                p = count / n
                pos_entropy -= p * math.log2(p)

        # Normalize by max possible entropy
        max_entropy = math.log2(len(AMINO_ACIDS))
        total_entropy += pos_entropy / max_entropy if max_entropy > 0 else 0.0

    return total_entropy / seq_len if seq_len > 0 else 0.0


def compute_mutation_spectrum_diversity(
    mutations_list: list[list[str]], wt_seq: str
) -> float:
    """Compute diversity of mutation spectra.

    Measures how varied the mutations are across candidates.

    Args:
        mutations_list: List of mutation lists for each candidate
        wt_seq: Wild-type sequence

    Returns:
        Mutation spectrum diversity score (0-1)
    """
    if not mutations_list:
        return 0.0

    # Collect all mutations
    all_mutations: set[str] = set()
    for mutations in mutations_list:
        all_mutations.update(mutations)

    if not all_mutations:
        return 0.0

    # Count mutation types
    wt_to_mut: dict[str, int] = {}
    for mutations in mutations_list:
        for mut in mutations:
            if mut:
                wt_aa = mut[0]  # First character is wild-type
                wt_to_mut[wt_aa] = wt_to_mut.get(wt_aa, 0) + 1

    # Compute distribution entropy
    total = sum(wt_to_mut.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in wt_to_mut.values():
        p = count / total
        entropy -= p * math.log2(p)

    # Normalize
    max_entropy = math.log2(len(AMINO_ACIDS))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def estimate_structural_diversity(
    sequences: list[str], wt_seq: str
) -> float:
    """Estimate structural diversity based on sequence properties.

    Uses physicochemical properties to estimate structural variation.

    Args:
        sequences: List of protein sequences
        wt_seq: Wild-type sequence

    Returns:
        Estimated structural diversity score (0-1)
    """
    if not sequences or not wt_seq:
        return 0.0

    # Physicochemical properties: hydrophobicity, charge, size
    properties = {
        'A': (1.8, 0, 89), 'C': (2.5, 0, 121), 'D': (-3.5, -1, 133),
        'E': (-3.5, -1, 147), 'F': (2.8, 0, 165), 'G': (-0.4, 0, 75),
        'H': (-3.2, 0, 155), 'I': (4.5, 0, 131), 'K': (-3.9, 1, 146),
        'L': (3.8, 0, 131), 'M': (1.9, 0, 149), 'N': (-3.5, 0, 132),
        'P': (-1.6, 0, 115), 'Q': (-3.5, 0, 146), 'R': (-4.5, 1, 174),
        'S': (-0.8, 0, 105), 'T': (-0.7, 0, 119), 'V': (4.2, 0, 117),
        'W': (-0.9, 0, 204), 'Y': (-1.3, 0, 181)
    }

    def get_property_vector(seq: str) -> tuple[float, float, float]:
        hydro = 0.0
        charge = 0.0
        size = 0.0
        for aa in seq:
            if aa in properties:
                p = properties[aa]
                hydro += p[0]
                charge += p[1]
                size += p[2]
        n = len(seq) or 1
        return hydro / n, charge / n, size / n

    wt_vector = get_property_vector(wt_seq)
    vectors = [get_property_vector(seq) for seq in sequences]

    # Compute variance from wild-type
    variances = [0.0, 0.0, 0.0]
    for vec in vectors:
        for i in range(3):
            diff = vec[i] - wt_vector[i]
            variances[i] += diff * diff

    for i in range(3):
        variances[i] /= len(vectors) if vectors else 1

    # Normalize variance (rough estimate of structural impact)
    max_var = 10.0  # Approximate max variance for normalization
    total_variance = sum(min(v / max_var, 1.0) for v in variances) / 3.0

    return min(total_variance, 1.0)


def compute_exploration_score(
    diversity_metrics: DiversityMetrics, config: DiversityConfig
) -> float:
    """Compute exploration score - measures how much the search explores new space.

    Args:
        diversity_metrics: Computed diversity metrics
        config: Diversity configuration

    Returns:
        Exploration score (0-1, higher = more exploration)
    """
    # Exploration is high when:
    # - High Hamming distance (sequences are different)
    # - High entropy (varied residues)
    # - High mutation spectrum diversity

    seq_exploration = min(
        diversity_metrics.mean_hamming_distance / 20.0, 1.0
    )  # Assume 20 aa is max meaningful distance
    entropy_exploration = diversity_metrics.shannon_entropy
    spectrum_exploration = diversity_metrics.mutation_spectrum_diversity

    exploration = (
        config.hamming_weight * seq_exploration +
        config.entropy_weight * entropy_exploration +
        config.spectrum_weight * spectrum_exploration
    )

    return min(exploration, 1.0)


def compute_exploitation_score(
    diversity_metrics: DiversityMetrics, config: DiversityConfig
) -> float:
    """Compute exploitation score - measures focus on high-scoring regions.

    Args:
        diversity_metrics: Computed diversity metrics
        config: Diversity configuration

    Returns:
        Exploitation score (0-1, higher = more exploitation)
    """
    # Exploitation is high when:
    # - Low minimum Hamming distance (some similar sequences)
    # - Low entropy (converged on specific residues)

    # Inverse of exploration (complementary measure)
    exploration = compute_exploration_score(diversity_metrics, config)
    return 1.0 - exploration


def compute_balance_ratio(exploration: float, exploitation: float) -> float:
    """Compute exploration-exploitation balance ratio.

    Args:
        exploration: Exploration score
        exploitation: Exploitation score

    Returns:
        Balance ratio (0.5 = balanced, <0.5 = exploitation-biased, >0.5 = exploration-biased)
    """
    total = exploration + exploitation
    if total == 0:
        return 0.5
    return exploration / total


def compute_all_diversity_metrics(
    sequences: list[str],
    mutations_list: list[list[str]],
    wt_seq: str,
    config: DiversityConfig | None = None
) -> DiversityMetrics:
    """Compute all diversity metrics for a set of sequences.

    Args:
        sequences: List of designed sequences
        mutations_list: List of mutations for each sequence
        wt_seq: Wild-type sequence
        config: Diversity configuration

    Returns:
        DiversityMetrics with all computed values
    """
    if config is None:
        config = DiversityConfig()

    metrics = DiversityMetrics()

    # Sequence diversity metrics
    metrics.mean_hamming_distance = compute_mean_hamming_distance(sequences)
    metrics.min_hamming_distance = compute_min_hamming_distance(sequences)
    metrics.shannon_entropy = compute_shannon_entropy(sequences)
    metrics.positional_entropy = compute_positional_entropy(sequences)

    # Functional diversity
    metrics.mutation_spectrum_diversity = compute_mutation_spectrum_diversity(
        mutations_list, wt_seq
    )

    # Structural diversity estimate
    metrics.structural_diversity_estimate = estimate_structural_diversity(
        sequences, wt_seq
    )

    # Exploration-exploitation balance
    metrics.exploration_score = compute_exploration_score(metrics, config)
    metrics.exploitation_score = compute_exploitation_score(metrics, config)
    metrics.balance_ratio = compute_balance_ratio(
        metrics.exploration_score, metrics.exploitation_score
    )

    return metrics


def compute_diversity_penalty(
    new_sequence: str,
    existing_sequences: list[str],
    config: DiversityConfig
) -> float:
    """Compute diversity penalty for adding a new sequence to the beam.

    Used to promote diverse candidates during beam search.

    Args:
        new_sequence: Sequence being considered
        existing_sequences: Sequences already in beam
        config: Diversity configuration

    Returns:
        Penalty value to subtract from score
    """
    if not existing_sequences:
        return 0.0

    # Compute average Hamming distance to existing sequences
    total_distance = 0.0
    for existing in existing_sequences:
        distance = sum(
            1 for a, b in zip(new_sequence, existing) if a != b
        )
        total_distance += distance

    avg_distance = total_distance / len(existing_sequences)

    # Normalize: higher distance = more diverse = lower penalty
    # Low distance (very similar) = high penalty
    max_distance = len(new_sequence) if new_sequence else 1
    similarity_ratio = 1.0 - (avg_distance / max_distance)

    penalty = config.diversity_penalty * similarity_ratio

    # Adaptive penalty: increase when diversity is low
    if config.adaptive_penalty:
        if len(existing_sequences) >= 3:
            current_diversity = compute_mean_hamming_distance(existing_sequences)
            if current_diversity < config.min_diversity_threshold * max_distance:
                penalty *= 1.5
            elif current_diversity > config.max_diversity_threshold * max_distance:
                penalty *= 0.5

    return penalty


def rank_by_diversity_score(
    candidates: list[tuple[float, str, tuple[str, ...], frozenset[str]]],
    wt_seq: str,
    config: DiversityConfig | None = None
) -> list[tuple[float, str, tuple[str, ...], frozenset[str]]]:
    """Rank beam candidates by diversity-augmented score.

    Re-ranks candidates considering both score and diversity.

    Args:
        candidates: List of (score, sequence, mutations, support_sources)
        wt_seq: Wild-type sequence
        config: Diversity configuration

    Returns:
        Re-ranked list of candidates
    """
    if config is None:
        config = DiversityConfig()
    if len(candidates) <= 1:
        return candidates

    sequences = [c[1] for c in candidates]
    mutations_list = [list(c[2]) for c in candidates]

    # Compute metrics for full set
    metrics = compute_all_diversity_metrics(sequences, mutations_list, wt_seq, config)

    # Compute diversity-adjusted scores
    adjusted_candidates = []
    for score, seq, mutations, sources in candidates:
        penalty = compute_diversity_penalty(seq, sequences, config)
        adjusted_score = score - penalty
        adjusted_candidates.append((adjusted_score, seq, mutations, sources))

    # Re-sort by adjusted score
    adjusted_candidates.sort(key=lambda x: -x[0])

    return adjusted_candidates


@dataclass
class DiversityTracker:
    """Tracks diversity metrics across beam search iterations."""

    config: DiversityConfig = field(default_factory=DiversityConfig)
    history: list[DiversityMetrics] = field(default_factory=list)
    current_diversity: DiversityMetrics = field(default_factory=DiversityMetrics)

    def update(
        self,
        sequences: list[str],
        mutations_list: list[list[str]],
        wt_seq: str
    ) -> DiversityMetrics:
        """Update diversity metrics with new beam state.

        Args:
            sequences: Current beam sequences
            mutations_list: Current beam mutations
            wt_seq: Wild-type sequence

        Returns:
            Updated diversity metrics
        """
        metrics = compute_all_diversity_metrics(
            sequences, mutations_list, wt_seq, self.config
        )
        self.history.append(metrics)
        self.current_diversity = metrics
        return metrics

    def get_adaptive_penalty(self) -> float:
        """Get adaptive diversity penalty based on current state.

        Returns:
            Penalty value adjusted for current diversity level
        """
        if not self.history:
            return self.config.diversity_penalty

        recent = self.history[-1]
        base_penalty = self.config.diversity_penalty

        # Increase penalty if diversity is too low (exploration needed)
        if recent.balance_ratio < 0.4:
            return base_penalty * (1.0 + self.config.exploration_bias)

        # Decrease penalty if diversity is high (exploitation may be better)
        if recent.balance_ratio > 0.6:
            return base_penalty * (1.0 - self.config.exploration_bias * 0.5)

        return base_penalty

    def should_encourage_diversity(self) -> bool:
        """Determine if diversity should be encouraged.

        Returns:
            True if current diversity is below threshold
        """
        if not self.history:
            return True
        return self.history[-1].balance_ratio < 0.5
