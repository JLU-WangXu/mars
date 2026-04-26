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
        beam: list[tuple[float, list[str], list[str], set[str]]] = [(0.0, list(wt_seq), [], set())]
        ordered_fields = sorted(fields, key=lambda item: item.position)
        pairwise_energies = pairwise_energies or {}

        for field in ordered_fields:
            seq_idx = position_to_index[field.position]
            next_beam: list[tuple[float, list[str], list[str], set[str]]] = []
            for current_score, seq_chars, mutations, support_sources in beam:
                for option in field.options:
                    new_chars = list(seq_chars)
                    new_mutations = list(mutations)
                    new_support_sources = set(support_sources)
                    new_chars[seq_idx] = option.residue
                    updated_score = current_score + float(option.score)
                    for prev_field in ordered_fields:
                        if prev_field.position == field.position:
                            break
                        prev_idx = position_to_index[prev_field.position]
                        prev_residue = new_chars[prev_idx]
                        pair_key = (prev_field.position, field.position)
                        reverse_pair_key = (field.position, prev_field.position)
                        pair_bucket = pairwise_energies.get(pair_key)
                        if pair_bucket is not None:
                            updated_score += float(pair_bucket.get((prev_residue, option.residue), 0.0))
                        else:
                            reverse_bucket = pairwise_energies.get(reverse_pair_key, {})
                            updated_score += float(reverse_bucket.get((option.residue, prev_residue), 0.0))
                    if option.residue != field.wt_residue:
                        updated_score -= self.mutation_penalty
                        mutation = f"{field.wt_residue}{field.position}{option.residue}"
                        if mutation not in new_mutations:
                            new_mutations.append(mutation)
                    if option.supporting_sources:
                        new_support_sources.update(option.supporting_sources)
                    next_beam.append((updated_score, new_chars, new_mutations, new_support_sources))
            next_beam.sort(key=lambda item: (-item[0], -len(item[3]), len(item[2]), "".join(item[2])))
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
