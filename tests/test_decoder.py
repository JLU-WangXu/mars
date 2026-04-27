"""Constrained beam decoder + pairwise energy plumbing."""

from __future__ import annotations

import pytest

from marsstack.decoder import ConstrainedBeamDecoder, PositionField, ResidueOption


def make_field(position: int, wt: str, options: list[tuple[str, float]]) -> PositionField:
    return PositionField(
        position=position,
        wt_residue=wt,
        options=[
            ResidueOption(residue=aa, score=score, supporting_sources=["test"], support_strength=score)
            for aa, score in options
        ],
    )


def test_decoder_returns_only_mutated_when_require_change():
    wt_seq = "AAAAA"
    fields = [
        make_field(1, "A", [("A", 0.0), ("C", 1.0)]),
        make_field(3, "A", [("A", 0.0), ("D", 0.5)]),
    ]
    decoder = ConstrainedBeamDecoder(beam_size=8, max_candidates=8, mutation_penalty=0.0, require_change=True)
    candidates = decoder.decode(
        wt_seq=wt_seq,
        position_to_index={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
        fields=fields,
    )
    assert candidates
    assert all(item.mutation_count >= 1 for item in candidates)


def test_decoder_pairwise_energy_can_flip_top_choice():
    wt_seq = "AAAA"
    fields = [
        make_field(1, "A", [("C", 1.0), ("D", 0.0)]),
        make_field(2, "A", [("C", 1.0), ("D", 0.0)]),
    ]
    pairwise = {(1, 2): {("D", "D"): 5.0}}
    decoder = ConstrainedBeamDecoder(beam_size=4, max_candidates=4, mutation_penalty=0.0)
    candidates = decoder.decode(
        wt_seq=wt_seq,
        position_to_index={1: 0, 2: 1, 3: 2, 4: 3},
        fields=fields,
        pairwise_energies=pairwise,
    )
    top = candidates[0]
    # Pair-bonus on D/D outweighs the higher per-position C/C score.
    assert top.sequence == "DDAA"


def test_decoder_caps_max_candidates():
    wt_seq = "A" * 6
    fields = [make_field(pos, "A", [("A", 0.0), ("C", 0.5), ("D", 0.4)]) for pos in range(1, 7)]
    decoder = ConstrainedBeamDecoder(beam_size=12, max_candidates=3, mutation_penalty=0.0)
    candidates = decoder.decode(
        wt_seq=wt_seq,
        position_to_index={i: i - 1 for i in range(1, 7)},
        fields=fields,
    )
    assert len(candidates) <= 3
