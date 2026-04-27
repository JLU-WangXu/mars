"""Pipeline-helper sanity tests."""

from __future__ import annotations

import pytest

from marsstack.pipeline import (
    ALPHABET,
    build_parsed_index_maps,
    collapse_mpnn_sequence,
    merge_recommendation_maps,
    project_to_design_positions,
    restore_template_mismatches,
    summarize_aligned_entries,
)


def test_alphabet_has_x_at_end():
    assert ALPHABET.endswith("X")
    assert "M" in ALPHABET
    assert len(ALPHABET) == 21


def test_summarize_aligned_entries_excludes_reference():
    wt = "ACDE"
    entries = [
        ("wt_reference", wt),
        ("h1", "ACDE"),
        ("h2", "AC-E"),
    ]
    accepted, mean_coverage = summarize_aligned_entries(entries, wt)
    assert accepted == 2
    # Position 3 has 1/2 coverage; others 2/2. Mean across positions = (1+1+0.5+1)/4 = 0.875
    assert mean_coverage == pytest.approx(0.875, rel=1e-3)


def test_summarize_aligned_entries_empty_input():
    assert summarize_aligned_entries([], "A") == (0, 0.0)


def test_merge_recommendation_maps_takes_max_per_aa():
    map_a = {1: {"A": 0.4, "C": 0.2}}
    map_b = {1: {"A": 0.7, "D": 0.3}, 2: {"E": 0.5}}
    merged = merge_recommendation_maps(map_a, map_b, top_k=4)
    assert merged[1]["A"] == pytest.approx(0.7)
    assert merged[1]["C"] == pytest.approx(0.2)
    assert merged[1]["D"] == pytest.approx(0.3)
    assert merged[2] == {"E": 0.5}


def test_merge_recommendation_maps_top_k_truncates():
    bias = {1: {aa: idx * 0.1 for idx, aa in enumerate("ACDEFG")}}
    merged = merge_recommendation_maps(bias, top_k=2)
    assert len(merged[1]) == 2
    # Top-k by score, descending
    assert set(merged[1]) == {"G", "F"}


def test_build_parsed_index_maps_skips_dashes():
    parsed_chain_seq = "A-CDE-FG"
    residue_numbers = [1, 2, 3, 4, 5, 6]
    pos_map, keep = build_parsed_index_maps(parsed_chain_seq, residue_numbers)
    assert keep == [0, 2, 3, 4, 6, 7]
    assert pos_map == {1: 0, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7}


def test_build_parsed_index_maps_raises_on_undercoverage():
    with pytest.raises(ValueError):
        build_parsed_index_maps("AB", [1, 2, 3])


def test_collapse_and_restore_round_trip():
    parsed_chain_seq = "AB-CD-EF"
    keep = [0, 1, 3, 4, 6, 7]
    collapsed = collapse_mpnn_sequence(parsed_chain_seq, keep)
    assert collapsed == "ABCDEF"


def test_restore_template_mismatches_only_touches_mismatch_positions():
    wt = "ACDEFG"
    seq = "AAAAAA"
    out = restore_template_mismatches(
        seq=seq,
        wt_seq=wt,
        mismatch_positions=[2, 4],
        position_to_index={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
    )
    assert out == "ACAEAA"


def test_project_to_design_positions_preserves_wt_outside_design_set():
    wt = "AAAAA"
    seq = "CDEFG"
    out = project_to_design_positions(
        seq=seq,
        wt_seq=wt,
        design_positions=[2, 4],
        position_to_index={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
    )
    assert out == "ADAFA"
