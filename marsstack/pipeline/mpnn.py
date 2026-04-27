from __future__ import annotations

import json
from pathlib import Path


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def build_parsed_index_maps(
    parsed_chain_seq: str,
    residue_numbers: list[int],
) -> tuple[dict[int, int], list[int]]:
    """Map structure residue numbers to ProteinMPNN parsed-chain indices.

    Returns ``(position_to_parsed_index, parsed_keep_indices)``.
    """

    position_to_parsed_index: dict[int, int] = {}
    parsed_keep_indices: list[int] = []
    residue_iter = iter(residue_numbers)
    current_residue = next(residue_iter, None)
    for idx, aa in enumerate(parsed_chain_seq):
        if aa == "-":
            continue
        if current_residue is None:
            break
        position_to_parsed_index[current_residue] = idx
        parsed_keep_indices.append(idx)
        current_residue = next(residue_iter, None)
    if current_residue is not None:
        raise ValueError("Parsed chain sequence does not cover all structure residue numbers.")
    if len(parsed_keep_indices) != len(residue_numbers):
        raise ValueError("Parsed chain sequence keep-indices do not match structure residue count.")
    return position_to_parsed_index, parsed_keep_indices


def collapse_mpnn_sequence(seq: str, parsed_keep_indices: list[int]) -> str:
    """Project a ProteinMPNN-emitted sequence back onto the structure residue order."""
    if parsed_keep_indices and max(parsed_keep_indices) >= len(seq):
        raise ValueError("ProteinMPNN sequence is shorter than expected parsed template length.")
    return "".join(seq[idx] for idx in parsed_keep_indices)


def restore_template_mismatches(
    seq: str,
    wt_seq: str,
    mismatch_positions: list[int],
    position_to_index: dict[int, int],
) -> str:
    """Restore WT residues at positions where the template differs from the WT."""
    if not mismatch_positions:
        return seq
    chars = list(seq)
    for pos in mismatch_positions:
        idx = position_to_index[pos]
        chars[idx] = wt_seq[idx]
    return "".join(chars)


def project_to_design_positions(
    seq: str,
    wt_seq: str,
    design_positions: list[int],
    position_to_index: dict[int, int],
) -> str:
    """Return ``wt_seq`` with only the design-position residues taken from ``seq``."""
    chars = list(wt_seq)
    for pos in design_positions:
        idx = position_to_index[pos]
        chars[idx] = seq[idx]
    return "".join(chars)


def build_bias_and_omit(
    protein_name: str,
    chain: str,
    seq_len: int,
    manual_bias: dict[int, dict[str, float]],
    oxidation_hotspots: list[int],
    wt_seq: str,
    position_to_index: dict[int, int],
    position_to_parsed_index: dict[int, int],
    bias_out: Path,
    omit_out: Path,
) -> None:
    """Write ProteinMPNN's ``bias_by_res`` and ``omit_aa`` JSONL files."""
    bias = [[[0.0 for _ in ALPHABET] for _ in range(seq_len)]]
    bias_rows = bias[0]
    for pos, aa_bias in manual_bias.items():
        idx = position_to_parsed_index[pos]
        for aa, val in aa_bias.items():
            bias_rows[idx][ALPHABET.index(aa)] = float(val)

    omit_items = []
    for pos in oxidation_hotspots:
        idx = position_to_parsed_index[pos]
        wt = wt_seq[position_to_index[pos]]
        if wt == "M":
            allowed = {"L", "I", "V"}
            forbidden = "".join([aa for aa in ALPHABET[:-1] if aa not in allowed])
            omit_items.append([[idx + 1], forbidden])

    bias_obj = {protein_name: {chain: bias_rows}}
    omit_obj = {protein_name: {chain: omit_items}}
    bias_out.write_text(json.dumps(bias_obj) + "\n", encoding="utf-8")
    omit_out.write_text(json.dumps(omit_obj) + "\n", encoding="utf-8")
