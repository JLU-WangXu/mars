from __future__ import annotations

from dataclasses import dataclass


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


class ConstrainedBeamDecoder:
    def __init__(
        self,
        beam_size: int = 32,
        max_candidates: int = 64,
        mutation_penalty: float = 0.15,
        require_change: bool = True,
    ) -> None:
        self.beam_size = int(beam_size)
        self.max_candidates = int(max_candidates)
        self.mutation_penalty = float(mutation_penalty)
        self.require_change = bool(require_change)

    def decode(
        self,
        wt_seq: str,
        position_to_index: dict[int, int],
        fields: list[PositionField],
        pairwise_energies: dict[tuple[int, int], dict[tuple[str, str], float]] | None = None,
    ) -> list[DecodedCandidate]:
        # State: (score, sequence_tuple, mutation_tuple, support_frozenset)
        beam: list[tuple[float, tuple[str, ...], tuple[str, ...], frozenset[str]]] = [
            (0.0, tuple(wt_seq), (), frozenset())
        ]
        ordered_fields = sorted(fields, key=lambda item: item.position)
        pairwise_energies = pairwise_energies or {}

        # Pre-compute previous indices for each field position
        field_positions = [f.position for f in ordered_fields]

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

                    next_beam.append((updated_score, new_chars, new_mutations, new_support_sources))

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

        candidates: list[DecodedCandidate] = []
        seen_sequences: set[str] = set()
        for score, seq_chars, mutations, support_sources in beam:
            if self.require_change and not mutations:
                continue
            sequence = "".join(seq_chars)
            if sequence in seen_sequences:
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
