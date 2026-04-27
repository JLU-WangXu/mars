"""
Active Learning Module for Efficient Design Exploration.

This module provides intelligent design selection strategies to reduce the number
of experiments required during protein design exploration. It uses uncertainty
sampling and ensemble methods to identify the most informative designs for evaluation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DesignCandidate:
    """A candidate design for active learning selection."""
    candidate_id: str
    sequence: str
    source: str = ""
    source_group: str = ""
    ensemble_predictions: dict[str, float] | None = None
    features: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_prediction_mean(self) -> float | None:
        """Get mean prediction across ensemble members."""
        if self.ensemble_predictions is None:
            return None
        return float(np.mean(list(self.ensemble_predictions.values())))

    def get_prediction_std(self) -> float | None:
        """Get prediction uncertainty (std across ensemble members)."""
        if self.ensemble_predictions is None:
            return None
        if len(self.ensemble_predictions) < 2:
            return 0.0
        return float(np.std(list(self.ensemble_predictions.values())))


@dataclass
class AcquisitionScore:
    """Result of acquisition function computation."""
    candidate_id: str
    score: float
    strategy: str
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning strategies."""
    # Query strategy type
    strategy: str = "entropy"  # Options: entropy, least_confidence, margin, committee, random

    # Batch selection
    batch_size: int = 32
    diversity_weight: float = 0.1  # Weight for diversity bonus in batch selection

    # Uncertainty thresholds
    uncertainty_threshold: float = 0.0  # Minimum uncertainty to include
    confidence_floor: float = 0.1  # Floor for confidence scores

    # Committee settings (for committee-based strategies)
    committee_size: int = 3
    disagreement_threshold: float = 0.1

    # Exploration vs exploitation
    exploration_weight: float = 0.5  # Balance between exploration and exploitation

    # Prioritization
    prioritize_novel: bool = True
    novelty_threshold: float = 0.3  # Minimum distance from evaluated designs

    # Convergence
    min_improvement: float = 0.01  # Minimum improvement to continue
    patience: int = 3  # Number of rounds without improvement before stopping

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ActiveLearningConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class UncertaintySampler:
    """
    Uncertainty sampling strategies for active learning.

    Implements multiple uncertainty quantification methods:
    - Entropy-based uncertainty
    - Least confidence sampling
    - Margin sampling
    - Standard deviation from ensemble predictions
    """

    def __init__(self, confidence_floor: float = 0.1) -> None:
        self.confidence_floor = confidence_floor

    def entropy(self, predictions: dict[str, float]) -> float:
        """
        Compute entropy-based uncertainty.

        Args:
            predictions: Dictionary mapping model IDs to predicted scores

        Returns:
            Entropy of the prediction distribution
        """
        if not predictions:
            return 0.0

        values = np.array(list(predictions.values()))
        # Normalize to probability distribution
        values_norm = values - values.min()
        if values_norm.max() > 0:
            probs = values_norm / values_norm.sum()
        else:
            probs = np.ones_like(values) / len(values)

        # Clip to avoid log(0)
        probs = np.clip(probs, self.confidence_floor, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)

    def least_confidence(self, predictions: dict[str, float]) -> float:
        """
        Compute least confidence score.

        Returns 1 - max_probability, so higher values indicate more uncertainty.

        Args:
            predictions: Dictionary mapping model IDs to predicted scores

        Returns:
            Least confidence score (0 = confident, 1 = uncertain)
        """
        if not predictions:
            return 1.0

        values = np.array(list(predictions.values()))
        max_prob = np.max(np.abs(values))
        # Normalize to [0, 1] range
        if values.max() != values.min():
            normalized = (values - values.min()) / (values.max() - values.min())
            max_prob = np.max(np.abs(normalized))

        return float(1.0 - max_prob)

    def margin(self, predictions: dict[str, float]) -> float:
        """
        Compute margin-based uncertainty.

        For binary classification: difference between top two predictions.
        For regression: normalized spread.

        Args:
            predictions: Dictionary mapping model IDs to predicted scores

        Returns:
            Margin score (higher = more uncertain)
        """
        if not predictions:
            return 1.0

        values = sorted(predictions.values(), reverse=True)
        if len(values) == 1:
            return 1.0

        # For regression: use normalized spread
        spread = values[0] - values[-1]
        if values[0] != values[-1]:
            return float(min(1.0, spread / (values[0] - values[-1] + 1e-8)))
        return 0.5

    def ensemble_std(self, predictions: dict[str, float]) -> float:
        """
        Compute standard deviation from ensemble predictions.

        Args:
            predictions: Dictionary mapping model IDs to predicted scores

        Returns:
            Standard deviation of predictions
        """
        if not predictions:
            return 0.0

        values = np.array(list(predictions.values()))
        if len(values) < 2:
            return 0.0

        return float(np.std(values))

    def compute_uncertainty(
        self,
        predictions: dict[str, float],
        method: str = "entropy"
    ) -> float:
        """
        Compute uncertainty using specified method.

        Args:
            predictions: Dictionary mapping model IDs to predicted scores
            method: Uncertainty computation method

        Returns:
            Uncertainty score
        """
        method_map = {
            "entropy": self.entropy,
            "least_confidence": self.least_confidence,
            "margin": self.margin,
            "std": self.ensemble_std,
        }

        if method not in method_map:
            logger.warning(f"Unknown uncertainty method '{method}', using entropy")
            method = "entropy"

        return method_map[method](predictions)


class CommitteeQueryStrategy:
    """
    Committee-based query strategy using Query by Committee.

    Maintains multiple model predictions and selects designs where
    committee members disagree most.
    """

    def __init__(
        self,
        committee_size: int = 3,
        disagreement_threshold: float = 0.1
    ) -> None:
        self.committee_size = committee_size
        self.disagreement_threshold = disagreement_threshold

    def vote_entropy(self, predictions: dict[str, float]) -> float:
        """
        Compute vote entropy for committee disagreement.

        Args:
            predictions: Dictionary mapping model IDs to predicted scores

        Returns:
            Vote entropy score
        """
        if len(predictions) < 2:
            return 0.0

        # Convert continuous predictions to discrete votes (above/below median)
        values = np.array(list(predictions.values()))
        median = np.median(values)
        votes = (values >= median).astype(int)

        # Compute entropy of vote distribution
        vote_probs = np.array([np.sum(votes) / len(votes), 1 - np.sum(votes) / len(votes)])
        vote_probs = np.clip(vote_probs, 1e-10, 1.0)
        entropy = -np.sum(vote_probs * np.log(vote_probs))
        max_entropy = np.log(2)
        return float(entropy / max_entropy if max_entropy > 0 else 0.0)

    def compute_disagreement(
        self,
        predictions: dict[str, float]
    ) -> float:
        """
        Compute overall committee disagreement.

        Args:
            predictions: Dictionary mapping model IDs to predicted scores

        Returns:
            Disagreement score
        """
        if len(predictions) < 2:
            return 0.0

        # Use standard deviation as disagreement measure
        std = float(np.std(list(predictions.values())))
        # Normalize by range
        values = list(predictions.values())
        value_range = max(values) - min(values) + 1e-8

        return min(1.0, std / value_range)


class DiversitySelector:
    """
    Diversity-aware batch selection.

    Ensures selected designs are diverse to maximize information gain.
    """

    def __init__(
        self,
        diversity_weight: float = 0.1,
        metric: str = "euclidean"
    ) -> None:
        self.diversity_weight = diversity_weight
        self.metric = metric

    def compute_diversity_bonus(
        self,
        candidate: DesignCandidate,
        selected: list[DesignCandidate],
        feature_key: str = "embedding"
    ) -> float:
        """
        Compute diversity bonus for a candidate given already selected items.

        Args:
            candidate: Candidate design
            selected: Already selected designs in batch
            feature_key: Key for features to use in distance computation

        Returns:
            Diversity bonus score (higher = more diverse)
        """
        if not selected:
            return 1.0

        if candidate.features is None or selected[0].features is None:
            return 0.5

        cand_feat = candidate.features.get(feature_key)
        if cand_feat is None:
            return 0.5

        distances = []
        for sel in selected:
            sel_feat = sel.features.get(feature_key)
            if sel_feat is not None:
                if self.metric == "euclidean":
                    dist = float(np.linalg.norm(np.array(cand_feat) - np.array(sel_feat)))
                else:
                    dist = 1.0
                distances.append(dist)

        if not distances:
            return 0.5

        return float(np.mean(distances))

    def greedy_batch_select(
        self,
        candidates: list[DesignCandidate],
        scores: list[float],
        batch_size: int,
        feature_key: str = "embedding"
    ) -> list[int]:
        """
        Select diverse batch using greedy approach.

        Args:
            candidates: List of candidate designs
            scores: Acquisition scores for candidates
            batch_size: Number of designs to select
            feature_key: Key for features to use in distance computation

        Returns:
            List of selected indices
        """
        if batch_size >= len(candidates):
            return list(range(len(candidates)))

        selected_indices = []
        remaining_indices = list(range(len(candidates)))

        # Sort by score descending
        sorted_indices = sorted(remaining_indices, key=lambda i: scores[i], reverse=True)

        # Select first (highest score) candidate
        selected_indices.append(sorted_indices[0])
        remaining_indices.remove(sorted_indices[0])

        # Greedily select remaining candidates
        for _ in range(batch_size - 1):
            if not remaining_indices:
                break

            best_score = float("-inf")
            best_idx = -1

            for idx in remaining_indices:
                # Combine acquisition score with diversity bonus
                base_score = scores[idx]
                diversity_bonus = self.compute_diversity_bonus(
                    candidates[idx],
                    [candidates[i] for i in selected_indices],
                    feature_key
                )
                combined_score = base_score + self.diversity_weight * diversity_bonus

                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx

            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        return selected_indices


class ActiveLearningOracle:
    """
    Oracle for providing ground truth labels.

    In practice, this would integrate with experimental evaluation systems.
    """

    def __init__(
        self,
        ground_truth_fn: Callable[[str], float] | None = None
    ) -> None:
        self.ground_truth_fn = ground_truth_fn
        self.evaluated: dict[str, float] = {}

    def evaluate(self, sequence: str, candidate_id: str) -> float:
        """
        Evaluate a design candidate.

        Args:
            sequence: Protein sequence
            candidate_id: Unique identifier for the candidate

        Returns:
            Ground truth score (e.g., experimental measurement)
        """
        if candidate_id in self.evaluated:
            return self.evaluated[candidate_id]

        if self.ground_truth_fn is not None:
            score = self.ground_truth_fn(sequence)
        else:
            # Default: use prediction mean
            score = 0.5

        self.evaluated[candidate_id] = score
        return score

    def add_evaluation(self, candidate_id: str, score: float) -> None:
        """Add a pre-computed evaluation result."""
        self.evaluated[candidate_id] = score

    def get_best_evaluated(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Get top-k evaluated designs."""
        sorted_evals = sorted(self.evaluated.items(), key=lambda x: x[1], reverse=True)
        return sorted_evals[:top_k]


class AcquisitionFunction:
    """
    Acquisition functions for active learning.

    Combines uncertainty with other factors like expected improvement.
    """

    @staticmethod
    def upper_confidence_bound(
        mean: float,
        std: float,
        beta: float = 1.96
    ) -> float:
        """
        Upper Confidence Bound (UCB) acquisition function.

        Args:
            mean: Predicted mean score
            std: Prediction uncertainty
            beta: Exploration parameter

        Returns:
            UCB score
        """
        return mean + beta * std

    @staticmethod
    def expected_improvement(
        mean: float,
        std: float,
        best_score: float,
        exploration_weight: float = 0.01
    ) -> float:
        """
        Expected Improvement (EI) acquisition function.

        Args:
            mean: Predicted mean score
            std: Prediction uncertainty
            best_score: Current best observed score
            exploration_weight: Minimum expected improvement

        Returns:
            Expected improvement score
        """
        if std < 1e-8:
            return 0.0

        z = (mean - best_score - exploration_weight) / std
        ei = (mean - best_score - exploration_weight) * 0.5 * (1 + np.math.erf(z / np.sqrt(2)))
        ei += std * np.exp(-0.5 * z * z) / np.sqrt(2 * np.math.pi)

        return float(max(0, ei))

    @staticmethod
    def probability_of_improvement(
        mean: float,
        std: float,
        best_score: float,
        threshold: float = 0.0
    ) -> float:
        """
        Probability of Improvement (PI) acquisition function.

        Args:
            mean: Predicted mean score
            std: Prediction uncertainty
            best_score: Current best observed score
            threshold: Minimum improvement threshold

        Returns:
            Probability of improvement
        """
        if std < 1e-8:
            return 1.0 if mean > best_score + threshold else 0.0

        z = (mean - best_score - threshold) / std
        return float(0.5 * (1 + np.math.erf(z / np.sqrt(2))))


class ActiveLearningLoop:
    """
    Main active learning loop for design exploration.

    Orchestrates the complete active learning cycle:
    1. Generate candidate designs
    2. Predict with ensemble
    3. Select informative designs using acquisition function
    4. Evaluate selected designs
    5. Update models
    6. Repeat until convergence
    """

    def __init__(
        self,
        config: ActiveLearningConfig | None = None,
        oracle: ActiveLearningOracle | None = None
    ) -> None:
        self.config = config or ActiveLearningConfig()
        self.oracle = oracle or ActiveLearningOracle()
        self.uncertainty_sampler = UncertaintySampler(
            confidence_floor=self.config.confidence_floor
        )
        self.committee_strategy = CommitteeQueryStrategy(
            committee_size=self.config.committee_size,
            disagreement_threshold=self.config.disagreement_threshold
        )
        self.diversity_selector = DiversitySelector(
            diversity_weight=self.config.diversity_weight
        )
        self.evaluated_history: list[dict[str, Any]] = []
        self.round_metrics: list[dict[str, float]] = []

    def select_batch(
        self,
        candidates: list[DesignCandidate],
        strategy: str | None = None
    ) -> list[DesignCandidate]:
        """
        Select a batch of designs using the configured strategy.

        Args:
            candidates: List of candidate designs
            strategy: Override query strategy (uses config default if None)

        Returns:
            Selected batch of candidates
        """
        strategy = strategy or self.config.strategy

        if strategy == "random":
            return self._random_select(candidates)

        if strategy == "entropy":
            return self._entropy_select(candidates)

        if strategy == "least_confidence":
            return self._least_confidence_select(candidates)

        if strategy == "committee":
            return self._committee_select(candidates)

        if strategy == "ucb":
            return self._ucb_select(candidates)

        if strategy == "ei":
            return self._expected_improvement_select(candidates)

        # Default to entropy-based selection
        return self._entropy_select(candidates)

    def _random_select(
        self,
        candidates: list[DesignCandidate]
    ) -> list[DesignCandidate]:
        """Select candidates uniformly at random."""
        n_select = min(self.config.batch_size, len(candidates))
        indices = np.random.choice(len(candidates), n_select, replace=False)
        return [candidates[i] for i in indices]

    def _entropy_select(
        self,
        candidates: list[DesignCandidate]
    ) -> list[DesignCandidate]:
        """Select candidates by entropy-based uncertainty."""
        acquisition_scores = []

        for cand in candidates:
            uncertainty = 0.0
            if cand.ensemble_predictions:
                uncertainty = self.uncertainty_sampler.compute_uncertainty(
                    cand.ensemble_predictions,
                    method="entropy"
                )
            acquisition_scores.append(uncertainty)

        selected_indices = self.diversity_selector.greedy_batch_select(
            candidates,
            acquisition_scores,
            self.config.batch_size
        )
        return [candidates[i] for i in selected_indices]

    def _least_confidence_select(
        self,
        candidates: list[DesignCandidate]
    ) -> list[DesignCandidate]:
        """Select candidates with lowest confidence (highest uncertainty)."""
        acquisition_scores = []

        for cand in candidates:
            uncertainty = 0.0
            if cand.ensemble_predictions:
                uncertainty = self.uncertainty_sampler.compute_uncertainty(
                    cand.ensemble_predictions,
                    method="least_confidence"
                )
            acquisition_scores.append(uncertainty)

        selected_indices = self.diversity_selector.greedy_batch_select(
            candidates,
            acquisition_scores,
            self.config.batch_size
        )
        return [candidates[i] for i in selected_indices]

    def _committee_select(
        self,
        candidates: list[DesignCandidate]
    ) -> list[DesignCandidate]:
        """Select candidates with highest committee disagreement."""
        acquisition_scores = []

        for cand in candidates:
            disagreement = 0.0
            if cand.ensemble_predictions:
                disagreement = self.committee_strategy.compute_disagreement(
                    cand.ensemble_predictions
                )
            acquisition_scores.append(disagreement)

        selected_indices = self.diversity_selector.greedy_batch_select(
            candidates,
            acquisition_scores,
            self.config.batch_size
        )
        return [candidates[i] for i in selected_indices]

    def _ucb_select(
        self,
        candidates: list[DesignCandidate]
    ) -> list[DesignCandidate]:
        """Select candidates using Upper Confidence Bound."""
        best_evaluated = self.oracle.get_best_evaluated(top_k=1)
        best_score = best_evaluated[0][1] if best_evaluated else 0.5

        acquisition_scores = []
        for cand in candidates:
            mean = cand.get_prediction_mean() or 0.5
            std = cand.get_prediction_std() or 0.0
            score = AcquisitionFunction.upper_confidence_bound(mean, std)
            # Boost if potentially better than current best
            if mean > best_score:
                score += self.config.exploration_weight
            acquisition_scores.append(score)

        selected_indices = self.diversity_selector.greedy_batch_select(
            candidates,
            acquisition_scores,
            self.config.batch_size
        )
        return [candidates[i] for i in selected_indices]

    def _expected_improvement_select(
        self,
        candidates: list[DesignCandidate]
    ) -> list[DesignCandidate]:
        """Select candidates using Expected Improvement."""
        best_evaluated = self.oracle.get_best_evaluated(top_k=1)
        best_score = best_evaluated[0][1] if best_evaluated else 0.5

        acquisition_scores = []
        for cand in candidates:
            mean = cand.get_prediction_mean() or 0.5
            std = cand.get_prediction_std() or 0.0
            score = AcquisitionFunction.expected_improvement(
                mean, std, best_score,
                exploration_weight=self.config.exploration_weight
            )
            acquisition_scores.append(score)

        selected_indices = self.diversity_selector.greedy_batch_select(
            candidates,
            acquisition_scores,
            self.config.batch_size
        )
        return [candidates[i] for i in selected_indices]

    def evaluate_batch(
        self,
        batch: list[DesignCandidate]
    ) -> list[tuple[DesignCandidate, float]]:
        """
        Evaluate a batch of candidates using the oracle.

        Args:
            batch: Batch of candidates to evaluate

        Returns:
            List of (candidate, score) tuples
        """
        results = []
        for cand in batch:
            score = self.oracle.evaluate(cand.sequence, cand.candidate_id)
            results.append((cand, score))
        return results

    def compute_round_metrics(
        self,
        batch_results: list[tuple[DesignCandidate, float]]
    ) -> dict[str, float]:
        """Compute metrics for the current round."""
        scores = [score for _, score in batch_results]
        best_new = max(scores) if scores else 0.0
        mean_new = np.mean(scores) if scores else 0.0

        metrics = {
            "batch_size": len(batch_results),
            "best_new_score": float(best_new),
            "mean_new_score": float(mean_new),
            "worst_new_score": float(min(scores)) if scores else 0.0,
        }

        # Compare to historical best
        if self.evaluated_history:
            all_best = max(
                entry["score"] for entry in self.evaluated_history
            )
            metrics["improvement"] = float(best_new - all_best)
            metrics["global_best"] = float(max(all_best, best_new))
        else:
            metrics["improvement"] = 0.0
            metrics["global_best"] = float(best_new)

        # Compute average uncertainty of selected batch
        uncertainties = []
        for cand, _ in batch_results:
            if cand.ensemble_predictions:
                unc = self.uncertainty_sampler.compute_uncertainty(
                    cand.ensemble_predictions,
                    method="entropy"
                )
                uncertainties.append(unc)
        metrics["mean_uncertainty"] = float(np.mean(uncertainties)) if uncertainties else 0.0

        return metrics

    def record_round(
        self,
        batch_results: list[tuple[DesignCandidate, float]],
        selected_candidates: list[DesignCandidate],
        metrics: dict[str, float]
    ) -> None:
        """Record round results for history tracking."""
        for cand, score in batch_results:
            self.evaluated_history.append({
                "candidate_id": cand.candidate_id,
                "sequence": cand.sequence,
                "score": score,
                "predicted_mean": cand.get_prediction_mean(),
                "predicted_std": cand.get_prediction_std(),
                "round": len(self.round_metrics),
            })

        self.round_metrics.append(metrics)

    def should_continue(
        self,
        min_improvement: float | None = None
    ) -> bool:
        """
        Check if active learning should continue.

        Args:
            min_imvement: Override minimum improvement threshold

        Returns:
            True if should continue, False otherwise
        """
        min_improvement = min_improvement or self.config.min_improvement

        if len(self.round_metrics) <= self.config.patience:
            return True

        # Check recent improvements
        recent_improvements = [
            m.get("improvement", 0.0)
            for m in self.round_metrics[-self.config.patience:]
        ]

        max_recent_improvement = max(recent_improvements)
        return max_recent_improvement > min_improvement

    def run(
        self,
        initial_candidates: list[DesignCandidate],
        max_rounds: int = 10,
        candidate_generator: Callable[[], list[DesignCandidate]] | None = None
    ) -> dict[str, Any]:
        """
        Run the complete active learning loop.

        Args:
            initial_candidates: Initial batch of candidates to evaluate
            max_rounds: Maximum number of rounds to run
            candidate_generator: Optional function to generate new candidates

        Returns:
            Dictionary containing final results and history
        """
        candidates = list(initial_candidates)
        round_num = 0

        # Evaluate initial batch
        if candidates:
            initial_results = self.evaluate_batch(candidates)
            metrics = self.compute_round_metrics(initial_results)
            self.record_round(initial_results, candidates, metrics)

        while round_num < max_rounds and self.should_continue():
            round_num += 1
            logger.info(f"Active learning round {round_num}/{max_rounds}")

            # Generate new candidates if generator provided
            if candidate_generator is not None:
                new_candidates = candidate_generator()
                candidates = new_candidates
            else:
                # Re-use candidates with updated predictions
                pass

            # Select batch
            selected = self.select_batch(candidates)

            # Evaluate
            results = self.evaluate_batch(selected)

            # Compute and record metrics
            metrics = self.compute_round_metrics(results)
            self.record_round(results, selected, metrics)

            logger.info(
                f"Round {round_num}: Best new={metrics['best_new_score']:.4f}, "
                f"Mean={metrics['mean_new_score']:.4f}, "
                f"Improvement={metrics['improvement']:.4f}"
            )

        # Return final results
        return self.get_results()

    def get_results(self) -> dict[str, Any]:
        """Get final results of active learning."""
        best_evaluated = self.oracle.get_best_evaluated(top_k=10)

        return {
            "total_evaluated": len(self.evaluated_history),
            "total_rounds": len(self.round_metrics),
            "best_designs": [
                {"candidate_id": cid, "score": score}
                for cid, score in best_evaluated
            ],
            "round_metrics": self.round_metrics,
            "history": self.evaluated_history,
        }

    def save_results(self, output_path: Path) -> None:
        """Save results to file."""
        results = self.get_results()

        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Process for JSON
        results_json = json.loads(
            json.dumps(results, default=convert)
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2)

        logger.info(f"Saved active learning results to {output_path}")

    @classmethod
    def load_results(cls, input_path: Path) -> dict[str, Any]:
        """Load results from file."""
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)


def convert_candidates_from_rows(
    rows: list[dict[str, Any]]
) -> list[DesignCandidate]:
    """
    Convert design rows from scoring pipeline to DesignCandidate objects.

    Args:
        rows: List of design row dictionaries

    Returns:
        List of DesignCandidate objects
    """
    candidates = []
    for row in rows:
        cand = DesignCandidate(
            candidate_id=str(row.get("candidate_id", "")),
            sequence=str(row.get("sequence", "")),
            source=str(row.get("source", "")),
            source_group=str(row.get("source_group", "")),
            metadata={
                "mars_score": row.get("mars_score"),
                "mutations": row.get("mutations", ""),
                "notes": row.get("notes", ""),
            }
        )
        candidates.append(cand)
    return candidates


def update_candidates_with_predictions(
    candidates: list[DesignCandidate],
    predictions: dict[str, dict[str, float]]
) -> list[DesignCandidate]:
    """
    Update candidates with ensemble predictions.

    Args:
        candidates: List of DesignCandidate objects
        predictions: Dictionary mapping candidate_id to prediction dict

    Returns:
        Updated list of DesignCandidate objects
    """
    pred_lookup = {str(k): v for k, v in predictions.items()}

    for cand in candidates:
        if cand.candidate_id in pred_lookup:
            cand.ensemble_predictions = pred_lookup[cand.candidate_id]
        else:
            cand.ensemble_predictions = {"default": 0.5}

    return candidates


def integrate_with_design_pipeline(
    candidates: list[dict[str, Any]],
    scoring_predictions: dict[str, dict[str, float]] | None = None,
    config: ActiveLearningConfig | None = None
) -> list[dict[str, Any]]:
    """
    Integrate active learning selection with design pipeline.

    This function takes scored design candidates and returns a selected
    subset for evaluation based on active learning criteria.

    Args:
        candidates: List of candidate design dictionaries from pipeline
        scoring_predictions: Optional ensemble predictions for candidates
        config: Active learning configuration

    Returns:
        Selected subset of candidates for evaluation
    """
    config = config or ActiveLearningConfig()

    # Convert to DesignCandidate objects
    design_candidates = convert_candidates_from_rows(candidates)

    # Update with predictions if provided
    if scoring_predictions:
        design_candidates = update_candidates_with_predictions(
            design_candidates, scoring_predictions
        )

    # Run active learning selection
    al_loop = ActiveLearningLoop(config=config)

    # If we have predictions, run a quick selection
    if scoring_predictions:
        selected = al_loop.select_batch(design_candidates)
    else:
        # Fall back to random selection for baseline
        selected = al_loop._random_select(design_candidates)

    # Convert back to original format
    selected_ids = {c.candidate_id for c in selected}
    return [c for c in candidates if str(c.get("candidate_id", "")) in selected_ids]


def estimate_reduction_ratio(
    total_candidates: int,
    batch_size: int,
    rounds: int
) -> float:
    """
    Estimate the evaluation reduction ratio.

    Args:
        total_candidates: Total number of candidate designs
        batch_size: Number selected per round
        rounds: Number of active learning rounds

    Returns:
        Estimated reduction ratio (evaluations / total)
    """
    total_evaluated = batch_size * rounds
    if total_candidates == 0:
        return 0.0
    return min(1.0, total_evaluated / total_candidates)


def estimate_sampling_efficiency(
    accuracy_improvement: float,
    reduction_ratio: float
) -> float:
    """
    Estimate sampling efficiency (accuracy improvement per evaluation).

    Args:
        accuracy_improvement: Improvement in top-1 accuracy
        reduction_ratio: Fraction of designs evaluated

    Returns:
        Efficiency score
    """
    if reduction_ratio == 0:
        return 0.0
    return accuracy_improvement / reduction_ratio
