"""
Multi-objective optimization framework for MARS-FIELD.

Provides Pareto front analysis, NSGA-II algorithm, weighted sum method,
and constraint satisfaction problem (CSP) solver for optimizing multiple
design objectives simultaneously.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Protocol


# =============================================================================
# Objective Functions and Data Structures
# =============================================================================


@dataclass
class ObjectiveVector:
    """Vector of objective values for a candidate solution."""

    stability: float = 0.0
    livability: float = 0.0
    expressivity: float = 0.0
    diversity: float = 0.0

    def to_list(self) -> list[float]:
        return [self.stability, self.livability, self.expressivity, self.diversity]

    @classmethod
    def from_list(cls, values: list[float]) -> "ObjectiveVector":
        if len(values) != 4:
            raise ValueError(f"Expected 4 values, got {len(values)}")
        return cls(stability=values[0], livability=values[1], expressivity=values[2], diversity=values[3])


@dataclass
class MultiObjectiveCandidate:
    """A candidate solution with its objective values and metadata."""

    sequence: str
    mutations: list[str]
    raw_scores: dict[str, float]
    objectives: ObjectiveVector
    dominated: bool = False
    rank: int = 0
    crowding_distance: float = 0.0


# =============================================================================
# Objective Protocols
# =============================================================================


class ObjectiveFunction(Protocol):
    """Protocol for objective functions."""

    def __call__(self, sequence: str, wt_seq: str, position_to_index: dict[int, int]) -> float:
        """Evaluate objective for a given sequence."""
        ...


# =============================================================================
# Pareto Front Analysis
# =============================================================================


def dominates(a: ObjectiveVector, b: ObjectiveVector) -> bool:
    """
    Check if objective vector a dominates b (Pareto dominance).

    a dominates b if a is better than or equal to b in all objectives
    and strictly better in at least one.
    """
    a_vals = a.to_list()
    b_vals = b.to_list()

    at_least_as_good = all(av >= bv for av, bv in zip(a_vals, b_vals))
    strictly_better = any(av > bv for av, bv in zip(a_vals, b_vals))

    return at_least_as_good and strictly_better


def is_pareto_optimal(
    candidates: list[MultiObjectiveCandidate],
    objectives: tuple[str, ...] = ("stability", "livability", "expressivity", "diversity"),
) -> set[int]:
    """
    Identify Pareto-optimal (non-dominated) candidates.

    Returns indices of non-dominated solutions in the input list.
    """
    n = len(candidates)
    is_pareto = set(range(n))

    for i in range(n):
        if i not in is_pareto:
            continue
        for j in range(n):
            if i == j or j not in is_pareto:
                continue

            cand_i = candidates[i]
            cand_j = candidates[j]

            obj_i = [getattr(cand_i.objectives, obj) for obj in objectives]
            obj_j = [getattr(cand_j.objectives, obj) for obj in objectives]

            if all(oi >= oj for oi, oj in zip(obj_i, obj_j)) and any(oi > oj for oi, oj in zip(obj_i, obj_j)):
                is_pareto.discard(i)
                break

    return is_pareto


def compute_pareto_front(
    candidates: list[MultiObjectiveCandidate],
) -> list[MultiObjectiveCandidate]:
    """Return list of non-dominated Pareto-optimal candidates."""
    pareto_indices = is_pareto_optimal(candidates)
    return [candidates[i] for i in sorted(pareto_indices)]


def crowding_distance(
    candidates: list[MultiObjectiveCandidate],
    objective_names: tuple[str, ...] = ("stability", "livability", "expressivity", "diversity"),
) -> None:
    """
    Compute crowding distance for each candidate.

    Modifies candidates in-place by setting their crowding_distance attribute.
    """
    n = len(candidates)
    if n <= 2:
        for c in candidates:
            c.crowding_distance = float("inf")
        return

    # Initialize distances
    for c in candidates:
        c.crowding_distance = 0.0

    # For each objective dimension
    for obj_name in objective_names:
        # Sort by objective value
        sorted_candidates = sorted(candidates, key=lambda c: getattr(c.objectives, obj_name))

        # Boundary solutions get infinite distance
        sorted_candidates[0].crowding_distance = float("inf")
        sorted_candidates[-1].crowding_distance = float("inf")

        # Get min and max values
        min_val = getattr(sorted_candidates[0].objectives, obj_name)
        max_val = getattr(sorted_candidates[-1].objectives, obj_name)
        range_val = max_val - min_val if max_val != min_val else 1.0

        # Compute distances for intermediate solutions
        for i in range(1, n - 1):
            next_val = getattr(sorted_candidates[i + 1].objectives, obj_name)
            prev_val = getattr(sorted_candidates[i - 1].objectives, obj_name)
            distance = (next_val - prev_val) / range_val
            sorted_candidates[i].crowding_distance += distance


# =============================================================================
# NSGA-II Algorithm (Simplified)
# =============================================================================


def fast_non_dominated_sort(
    candidates: list[MultiObjectiveCandidate],
) -> list[list[MultiObjectiveCandidate]]:
    """
    Fast non-dominated sorting (NSGA-II).

    Returns list of fronts, where each front is a list of non-dominated solutions.
    """
    n = len(candidates)
    domination_count: dict[int, int] = {i: 0 for i in range(n)}
    dominated_solutions: dict[int, list[int]] = {i: [] for i in range(n)}

    # For each candidate, find solutions it dominates and solutions that dominate it
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(candidates[i].objectives, candidates[j].objectives):
                dominated_solutions[i].append(j)
            elif dominates(candidates[j].objectives, candidates[i].objectives):
                domination_count[i] += 1

    # Find first front (non-dominated solutions)
    fronts: list[list[int]] = [[]]
    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)
            candidates[i].rank = 0

    # Find subsequent fronts
    current_front = 0
    while fronts[current_front]:
        next_front: list[int] = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
                    candidates[j].rank = current_front + 1
        current_front += 1
        if next_front:
            fronts.append(next_front)

    return [[candidates[i] for i in front] for front in fronts if front]


def nsga2_select(
    candidates: list[MultiObjectiveCandidate],
    population_size: int,
    objective_names: tuple[str, ...] = ("stability", "livability", "expressivity", "diversity"),
) -> list[MultiObjectiveCandidate]:
    """
    NSGA-II selection operator.

    Selects population_size candidates using non-dominated sorting and crowding distance.
    """
    # Compute ranks
    fronts = fast_non_dominated_sort(candidates)

    # Compute crowding distances for each front
    for front in fronts:
        crowding_distance(front, objective_names)

    # Select based on rank, then crowding distance
    selected: list[MultiObjectiveCandidate] = []
    for front in fronts:
        if len(selected) + len(front) <= population_size:
            selected.extend(front)
        else:
            # Sort by crowding distance (descending) and take remaining slots
            sorted_front = sorted(front, key=lambda c: -c.crowding_distance)
            remaining = population_size - len(selected)
            selected.extend(sorted_front[:remaining])
            break

    return selected


def nsga2_evolve(
    initial_candidates: list[MultiObjectiveCandidate],
    generations: int = 50,
    population_size: int = 100,
    mutation_rate: float = 0.1,
    objective_functions: dict[str, ObjectiveFunction] | None = None,
    random_seed: int | None = None,
) -> list[MultiObjectiveCandidate]:
    """
    Simplified NSGA-II optimization.

    Args:
        initial_candidates: Initial population of candidates
        generations: Number of generations to evolve
        population_size: Size of population
        mutation_rate: Probability of mutation
        objective_functions: Dict mapping objective names to evaluation functions
        random_seed: Random seed for reproducibility

    Returns:
        Final Pareto front
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Initialize population
    population = list(initial_candidates)

    # Ensure we have enough candidates
    while len(population) < population_size:
        population.append(random.choice(population))

    for _ in range(generations):
        # Evaluate objectives if functions provided
        if objective_functions:
            for candidate in population:
                obj_vec = ObjectiveVector()
                for obj_name, obj_func in objective_functions.items():
                    value = obj_func(candidate.sequence, "", {})
                    setattr(obj_vec, obj_name, value)
                candidate.objectives = obj_vec

        # Non-dominated sorting
        fronts = fast_non_dominated_sort(population)

        # Compute crowding distances
        for front in fronts:
            crowding_distance(front)

        # Sort by rank, then crowding distance
        population.sort(key=lambda c: (c.rank, -c.crowding_distance))

        # Select next generation
        population = population[:population_size]

    # Return final Pareto front
    return [c for c in population if c.rank == 0]


# =============================================================================
# Weighted Sum Method
# =============================================================================


@dataclass
class WeightsConfig:
    """Configuration for weighted sum optimization."""

    stability: float = 0.3
    livability: float = 0.3
    expressivity: float = 0.2
    diversity: float = 0.2

    def __post_init__(self) -> None:
        total = self.stability + self.livability + self.expressivity + self.diversity
        if abs(total - 1.0) > 1e-6:
            # Normalize weights
            self.stability /= total
            self.livability /= total
            self.expressivity /= total
            self.diversity /= total

    def to_list(self) -> list[float]:
        return [self.stability, self.livability, self.expressivity, self.diversity]


def weighted_sum_score(candidate: MultiObjectiveCandidate, weights: WeightsConfig) -> float:
    """
    Compute weighted sum of objectives.

    All objectives are normalized to [0, 1] range before weighting.
    """
    obj_vals = candidate.objectives.to_list()

    # Normalize using sigmoid-like scaling
    normalized = [1.0 / (1.0 + math.exp(-v)) for v in obj_vals]

    return sum(w * n for w, n in zip(weights.to_list(), normalized))


def optimize_weighted_sum(
    candidates: list[MultiObjectiveCandidate],
    weights: WeightsConfig | None = None,
    top_k: int = 10,
) -> list[MultiObjectiveCandidate]:
    """
    Optimize candidates using weighted sum method.

    Returns top_k candidates sorted by weighted sum score.
    """
    if weights is None:
        weights = WeightsConfig()

    scored = [(weighted_sum_score(c, weights), c) for c in candidates]
    scored.sort(key=lambda x: -x[0])

    return [c for _, c in scored[:top_k]]


# =============================================================================
# Constraint Satisfaction Problem (CSP) Solver
# =============================================================================


@dataclass
class Constraint:
    """A constraint for CSP solving."""

    name: str
    evaluate: Callable[[MultiObjectiveCandidate], bool]
    weight: float = 1.0  # Penalty weight when violated


@dataclass
class CSPConfig:
    """Configuration for CSP solving."""

    constraints: list[Constraint] = field(default_factory=list)
    hard_constraints: list[Constraint] = field(default_factory=list)
    max_penalty: float = 100.0

    def add_constraint(self, name: str, evaluate: Callable[[MultiObjectiveCandidate], bool],
                       weight: float = 1.0, hard: bool = False) -> None:
        """Add a constraint to the configuration."""
        constraint = Constraint(name=name, evaluate=evaluate, weight=weight)
        if hard:
            self.hard_constraints.append(constraint)
        else:
            self.constraints.append(constraint)


def satisfy_constraints(candidate: MultiObjectiveCandidate, config: CSPConfig) -> tuple[bool, float]:
    """
    Check if candidate satisfies CSP constraints.

    Returns:
        Tuple of (is_feasible, penalty_score)
    """
    penalty = 0.0

    # Check hard constraints first
    for constraint in config.hard_constraints:
        if not constraint.evaluate(candidate):
            return False, config.max_penalty

    # Evaluate soft constraints
    for constraint in config.constraints:
        if not constraint.evaluate(candidate):
            penalty += constraint.weight

    return True, penalty


def csp_solve(
    candidates: list[MultiObjectiveCandidate],
    config: CSPConfig,
    penalty_weight: float = 1.0,
) -> list[MultiObjectiveCandidate]:
    """
    Solve constraint satisfaction problem.

    Returns feasible candidates sorted by objective score with constraint penalties applied.
    """
    evaluated = []
    for candidate in candidates:
        is_feasible, penalty = satisfy_constraints(candidate, config)
        if is_feasible:
            # Adjust score by penalty
            adjusted_score = sum(candidate.objectives.to_list()) - penalty_weight * penalty
            candidate.raw_scores["csp_adjusted"] = adjusted_score
            evaluated.append(candidate)

    # Sort by adjusted score
    evaluated.sort(key=lambda c: -c.raw_scores.get("csp_adjusted", 0.0))

    return evaluated


# =============================================================================
# Multi-objective Score Integration
# =============================================================================


def compute_multi_objective_scores(
    sequence: str,
    wt_seq: str,
    position_to_index: dict[int, int],
    stability_score: float | None = None,
    livability_score: float | None = None,
    mars_score: float | None = None,
    diversity_baseline: float | None = None,
) -> ObjectiveVector:
    """
    Compute multi-objective vector for a sequence.

    Args:
        sequence: Designed sequence
        wt_seq: Wild-type sequence
        position_to_index: Position to index mapping
        stability_score: Pre-computed stability score (optional)
        livability_score: Pre-computed livability score (optional)
        mars_score: MARS score
        diversity_baseline: Baseline diversity metric

    Returns:
        ObjectiveVector with computed objective values
    """
    # Stability: based on MARS score (higher is better)
    stability = stability_score if stability_score is not None else (mars_score or 0.0)

    # Livability: measure of active site preservation
    livability = livability_score if livability_score is not None else 0.5

    # Expressivity: mutation count and diversity
    mutation_count = sum(1 for i, aa in enumerate(sequence) if i < len(wt_seq) and aa != wt_seq[i])
    expressivity = min(1.0, mutation_count / max(1, len(wt_seq) * 0.1))

    # Diversity: distance from baseline
    if diversity_baseline is not None:
        diversity = diversity_baseline
    else:
        diversity = expressivity * 0.5

    return ObjectiveVector(
        stability=stability,
        livability=livability,
        expressivity=expressivity,
        diversity=diversity,
    )


def rank_candidates_multi_objective(
    candidates: list[MultiObjectiveCandidate],
    method: str = "pareto",
    weights: WeightsConfig | None = None,
    csp_config: CSPConfig | None = None,
    top_k: int = 10,
) -> list[MultiObjectiveCandidate]:
    """
    Rank candidates using multi-objective optimization.

    Args:
        candidates: List of candidates to rank
        method: Ranking method ("pareto", "weighted_sum", "nsga2", "csp")
        weights: Weights for weighted sum method
        csp_config: CSP configuration for constraint solving
        top_k: Number of top candidates to return

    Returns:
        Ranked list of candidates
    """
    if not candidates:
        return []

    if method == "pareto":
        pareto_front = compute_pareto_front(candidates)
        # Sort by crowding distance within Pareto front
        crowding_distance(pareto_front)
        pareto_front.sort(key=lambda c: -c.crowding_distance)
        return pareto_front[:top_k]

    elif method == "weighted_sum":
        return optimize_weighted_sum(candidates, weights, top_k)

    elif method == "nsga2":
        fronts = fast_non_dominated_sort(candidates)
        for front in fronts:
            crowding_distance(front)
        # Combine all fronts, sorted by rank then crowding distance
        all_sorted = []
        for front in fronts:
            all_sorted.extend(sorted(front, key=lambda c: -c.crowding_distance))
        return all_sorted[:top_k]

    elif method == "csp" and csp_config is not None:
        return csp_solve(candidates, csp_config)[:top_k]

    else:
        # Default: simple sum of objectives
        scored = [(sum(c.objectives.to_list()), c) for c in candidates]
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:top_k]]


# =============================================================================
# Default Constraint Templates
# =============================================================================


def min_stability_constraint(threshold: float = 0.0) -> Constraint:
    """Create minimum stability constraint."""
    return Constraint(
        name="min_stability",
        evaluate=lambda c: c.objectives.stability >= threshold,
        weight=10.0,
    )


def max_mutations_constraint(max_mut: int = 10) -> Constraint:
    """Create maximum mutations constraint."""
    return Constraint(
        name="max_mutations",
        evaluate=lambda c: len(c.mutations) <= max_mut,
        weight=5.0,
    )


def min_diversity_constraint(threshold: float = 0.1) -> Constraint:
    """Create minimum diversity constraint."""
    return Constraint(
        name="min_diversity",
        evaluate=lambda c: c.objectives.diversity >= threshold,
        weight=3.0,
    )


def create_default_csp_config() -> CSPConfig:
    """Create default CSP configuration."""
    config = CSPConfig()
    config.add_constraint("min_stability", lambda c: c.objectives.stability >= 0.0, hard=True)
    config.add_constraint("max_mutations", lambda c: len(c.mutations) <= 20, weight=5.0)
    config.add_constraint("min_livability", lambda c: c.objectives.livability >= 0.3, weight=3.0)
    return config
