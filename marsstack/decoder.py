from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .cache_manager import MARSCacheManager
from .physics_constraints import (
    compute_constraint_penalty,
    get_constraint_summary,
    validate_sequence_constraints,
    CHARGED_POSITIVE,
    CHARGED_NEGATIVE,
    CLASH_DISTANCE,
)
from .disulfide_design import (
    DisulfideConstraint,
    DisulfideDesignResult,
    compute_disulfide_penalty,
    design_disulfide_bonds,
    get_design_summary,
    get_disulfide_summary,
    filter_candidate_pairs,
    predict_disulfide_formation,
    suggest_cysteine_mutations,
    MIN_SS_DISTANCE,
    MAX_SS_DISTANCE,
)
from .diversity_metrics import (
    DiversityConfig,
    DiversityTracker,
    DiversityMetrics,
    compute_diversity_penalty,
    compute_all_diversity_metrics,
)
from .mutation_predictor import (
    MutationEffect,
    ThermostabilityScore,
    compute_beam_search_mutation_bonus,
    predict_ddg,
    score_sequence_thermostability,
)

if TYPE_CHECKING:
    from .online_learner import OnlineLearner


@dataclass
class ResidueOption:
    residue: str
    score: float
    supporting_sources: list[str] | None = None
    support_strength: float = 0.0
    evidence_breakdown: dict[str, float] | None = None


@dataclass
class PositionField:
    position: int
    wt_residue: str
    options: list[ResidueOption]


@dataclass
class DecodedCandidate:
    sequence: str
    mutations: list[str]
    decoder_score: float
    mutation_count: int
    supporting_sources: list[str]
    thermostability_score: ThermostabilityScore | None = None
    mutation_effects: list[MutationEffect] | None = None


class ConstrainedBeamDecoder:
    def __init__(
        self,
        beam_size: int = 32,
        max_candidates: int = 64,
        mutation_penalty: float = 0.15,
        require_change: bool = True,
        enable_physics_constraints: bool = True,
        constraint_weight: float = 1.0,
        max_constraint_violations: int = 3,
        learner: "OnlineLearner | None" = None,
        learning_reward_scale: float = 1.0,
        diversity_config: DiversityConfig | None = None,
        use_diversity_penalty: bool = False,
        use_diversity_selection: bool = False,
    ) -> None:
        self.beam_size = int(beam_size)
        self.max_candidates = int(max_candidates)
        self.mutation_penalty = float(mutation_penalty)
        self.require_change = bool(require_change)
        self.enable_physics_constraints = bool(enable_physics_constraints)
        self.constraint_weight = float(constraint_weight)
        self.max_constraint_violations = int(max_constraint_violations)
        self.learner = learner
        self.learning_reward_scale = float(learning_reward_scale)
        self.diversity_config = diversity_config or DiversityConfig()
        self.use_diversity_penalty = use_diversity_penalty
        self.use_diversity_selection = use_diversity_selection
        self._diversity_tracker: DiversityTracker | None = None

    def _init_diversity_tracker(self, wt_seq: str) -> DiversityTracker:
        """Initialize diversity tracker for this decode run."""
        self._diversity_tracker = DiversityTracker(
            config=self.diversity_config,
            history=[],
            current_diversity=DiversityMetrics(),
        )
        return self._diversity_tracker

    def _apply_diversity_penalty(
        self,
        new_sequence: str,
        existing_sequences: list[str],
    ) -> float:
        """Compute diversity penalty for a new candidate."""
        if not self.use_diversity_penalty or not existing_sequences:
            return 0.0
        return compute_diversity_penalty(
            new_sequence, existing_sequences, self.diversity_config
        )

    def _get_adaptive_diversity_penalty(self) -> float:
        """Get adaptive penalty from tracker if available."""
        if self._diversity_tracker:
            return self._diversity_tracker.get_adaptive_penalty()
        return self.diversity_config.diversity_penalty

    def decode(
        self,
        wt_seq: str,
        position_to_index: dict[int, int],
        fields: list[PositionField],
        pairwise_energies: dict[tuple[int, int], dict[tuple[str, str], float]] | None = None,
        pair_distances: dict[tuple[int, int], float] | None = None,
        sasa_map: dict[int, float] | None = None,
        existing_disulfides: set[tuple[int, int]] | None = None,
        core_positions: set[int] | None = None,
    ) -> list[DecodedCandidate]:
        # State: (score, sequence_tuple, mutation_tuple, support_frozenset)
        beam: list[tuple[float, tuple[str, ...], tuple[str, ...], frozenset[str]]] = [
            (0.0, tuple(wt_seq), (), frozenset())
        ]
        ordered_fields = sorted(fields, key=lambda item: item.position)
        pairwise_energies = pairwise_energies or {}
        pair_distances = pair_distances or {}
        sasa_map = sasa_map or {}
        existing_disulfides = existing_disulfides or set()
        core_positions = core_positions or set()

        # Pre-compute previous indices for each field position
        field_positions = [f.position for f in ordered_fields]

        # Cache constraint data for validation
        constraint_cache: dict[tuple[str, ...], float] = {}

        for field_idx, field in enumerate(ordered_fields):
            seq_idx = position_to_index[field.position]
            # Get indices of previously processed fields (fields before current field)

            next_beam: list[tuple[float, tuple[str, ...], tuple[str, ...], frozenset[str]]] = []

            for current_score, seq_chars, mutations, support_sources in beam:
                seq_list = list(seq_chars)  # Convert to list only once per beam state

                for option in field.options:
                    # Modify list in place, then convert back to tuple
                    seq_list[seq_idx] = option.residue
                    new_chars = tuple(seq_list)
                    seq_list[seq_idx] = seq_chars[seq_idx]  # Restore for next option

                    updated_score = current_score + float(option.score)

                    # Compute pairwise energies for all previous positions at once
                    option_residue = option.residue
                    for prev_idx in range(field_idx):
                        prev_field_pos = field_positions[prev_idx]
                        prev_residue = new_chars[position_to_index[prev_field_pos]]
                        pair_key = (prev_field_pos, field.position)
                        pair_bucket = pairwise_energies.get(pair_key)
                        if pair_bucket is not None:
                            updated_score += float(pair_bucket.get((prev_residue, option_residue), 0.0))
                        else:
                            reverse_bucket = pairwise_energies.get((field.position, prev_field_pos), {})
                            updated_score += float(reverse_bucket.get((option_residue, prev_residue), 0.0))

                    # Track mutations using tuple (immutable, hashable)
                    if option_residue != field.wt_residue:
                        updated_score -= self.mutation_penalty
                        mutation = f"{field.wt_residue}{field.position}{option_residue}"
                        # Check set for O(1) lookup instead of list O(n)
                        if mutation not in mutations:
                            new_mutations = mutations + (mutation,)
                        else:
                            new_mutations = mutations
                    else:
                        new_mutations = mutations

                    # Update support sources
                    if option.supporting_sources:
                        new_support_sources = support_sources | frozenset(option.supporting_sources)
                    else:
                        new_support_sources = support_sources

                    # Apply physics constraints if enabled
                    constraint_penalty = 0.0
                    if self.enable_physics_constraints and pair_distances:
                        # Use cached constraint penalty if available
                        if new_chars in constraint_cache:
                            constraint_penalty = constraint_cache[new_chars]
                        else:
                            constraint_penalty = compute_constraint_penalty(
                                seq=new_chars,
                                position_to_index=position_to_index,
                                field_positions=field_positions,
                                pair_distances=pair_distances,
                                sasa_map=sasa_map,
                                existing_disulfides=existing_disulfides,
                                core_positions=core_positions,
                            )
                            # Cache the constraint penalty for reuse
                            constraint_cache[new_chars] = constraint_penalty

                        updated_score += constraint_penalty * self.constraint_weight

                    # Apply diversity penalty if enabled
                    if self.use_diversity_penalty and next_beam:
                        existing_seqs = [seq for _, seq, _, _ in next_beam]
                        diversity_penalty = self._apply_diversity_penalty(
                            "".join(new_chars), existing_seqs
                        )
                        updated_score -= diversity_penalty

                    next_beam.append((updated_score, new_chars, new_mutations, new_support_sources))

            # Apply diversity-aware selection if enabled
            if self.use_diversity_selection and next_beam:
                if self._diversity_tracker is None:
                    self._init_diversity_tracker(wt_seq)
                seq_chars_list = [c[1] for c in next_beam]
                mutations_list = [list(c[2]) for c in next_beam]
                self._diversity_tracker.update(seq_chars_list, mutations_list, wt_seq)

            # Sort and prune beam - use mutation_count directly (pre-computed)
            next_beam.sort(
                key=lambda item: (
                    -item[0],
                    -len(item[3]),
                    len(item[2]),
                    item[2]  # Tuples are comparable, no string join needed
                )
            )
            beam = next_beam[: self.beam_size]

        # Post-filter candidates by constraint violations
        candidates: list[DecodedCandidate] = []
        seen_sequences: set[str] = set()

        for score, seq_chars, mutations, support_sources in beam:
            if self.require_change and not mutations:
                continue
            sequence = "".join(seq_chars)
            if sequence in seen_sequences:
                continue

            # Validate physics constraints for final candidates
            if self.enable_physics_constraints and pair_distances and sasa_map:
                result = validate_sequence_constraints(
                    sequence=sequence,
                    residue_numbers=field_positions,
                    pair_distances=pair_distances,
                    sasa_map=sasa_map,
                    existing_disulfides=existing_disulfides,
                    core_positions=core_positions,
                )
                if len(result.violations) > self.max_constraint_violations:
                    continue

            seen_sequences.add(sequence)
            candidates.append(
                DecodedCandidate(
                    sequence=sequence,
                    mutations=mutations,
                    decoder_score=round(float(score), 6),
                    mutation_count=len(mutations),
                    supporting_sources=sorted(support_sources),
                )
            )
            if len(candidates) >= self.max_candidates:
                break
        return candidates

    def get_diversity_metrics(
        self,
        sequences: list[str],
        mutations_list: list[list[str]],
        wt_seq: str,
    ) -> DiversityMetrics:
        """Get diversity metrics for a set of sequences.

        Args:
            sequences: List of designed sequences
            mutations_list: List of mutations for each sequence
            wt_seq: Wild-type sequence

        Returns:
            Computed diversity metrics
        """
        return compute_all_diversity_metrics(
            sequences, mutations_list, wt_seq, self.diversity_config
        )
