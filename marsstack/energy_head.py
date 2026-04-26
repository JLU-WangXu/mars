from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gemmi

from .decoder import PositionField


def compute_design_pair_distances(
    pdb_path: Path,
    chain_id: str,
    positions: list[int],
) -> dict[tuple[int, int], float]:
    st = gemmi.read_structure(str(pdb_path))
    chain = st[0][chain_id]

    ca_positions: dict[int, tuple[float, float, float]] = {}
    for residue in chain:
        if residue.seqid.num not in positions:
            continue
        atom = residue.find_atom("CA", "\0")
        if atom:
            ca_positions[residue.seqid.num] = (atom.pos.x, atom.pos.y, atom.pos.z)

    distances: dict[tuple[int, int], float] = {}
    for i, pos_i in enumerate(positions):
        for pos_j in positions[i + 1 :]:
            if pos_i not in ca_positions or pos_j not in ca_positions:
                continue
            xi, yi, zi = ca_positions[pos_i]
            xj, yj, zj = ca_positions[pos_j]
            d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
            distances[(pos_i, pos_j)] = round(d, 6)
    return distances


def build_pairwise_energy_tensor(
    rows: list[dict[str, Any]],
    fields: list[PositionField],
    position_to_index: dict[int, int],
    pair_distances: dict[tuple[int, int], float],
    top_rows: int = 40,
) -> dict[tuple[int, int], dict[tuple[str, str], float]]:
    field_positions = [field.position for field in fields]
    top_ranked = sorted(
        rows,
        key=lambda item: (-float(item.get("ranking_score", item.get("mars_score", 0.0))), -float(item.get("mars_score", 0.0))),
    )[: int(top_rows)]

    pairwise: dict[tuple[int, int], dict[tuple[str, str], float]] = {}
    for i, pos_i in enumerate(field_positions):
        for pos_j in field_positions[i + 1 :]:
            pair = (pos_i, pos_j)
            reverse_pair = (pos_j, pos_i)
            distance = pair_distances.get(pair, pair_distances.get(reverse_pair, 12.0))
            if distance > 18.0:
                continue
            distance_weight = 1.0 / max(1.0, distance / 4.0)
            bucket: dict[tuple[str, str], float] = {}
            for rank_idx, row in enumerate(top_ranked, start=1):
                sequence = str(row["sequence"])
                aa_i = sequence[position_to_index[pos_i]]
                aa_j = sequence[position_to_index[pos_j]]
                rank_weight = 1.0 / math.sqrt(rank_idx)
                score_weight = 0.5 + 0.25 * math.tanh(float(row.get("ranking_score", row.get("mars_score", 0.0))) / 4.0)
                bucket[(aa_i, aa_j)] = bucket.get((aa_i, aa_j), 0.0) + distance_weight * rank_weight * score_weight
            if bucket:
                pairwise[pair] = {key: round(value, 6) for key, value in bucket.items()}
    return pairwise


def serialize_pairwise_energy_tensor(
    pairwise: dict[tuple[int, int], dict[tuple[str, str], float]]
) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for (pos_i, pos_j), bucket in pairwise.items():
        pair_key = f"{pos_i}-{pos_j}"
        payload[pair_key] = {
            f"{aa_i}:{aa_j}": round(score, 6)
            for (aa_i, aa_j), score in sorted(bucket.items(), key=lambda item: (-item[1], item[0]))[:32]
        }
    return payload
