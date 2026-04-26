from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from ..mars_score import SAFE_OXIDATION_MAP
from ..topic_score import build_topic_local_recommendations


SOURCE_PRIORITY = {
    "manual": 1,
    "baseline_mpnn": 2,
    "local_proposal": 3,
    "mars_mpnn": 4,
    "esm_if": 5,
    "fusion_decoder": 6,
    "neural_decoder": 7,
}


@dataclass
class CandidateEntry:
    candidate_id: str
    source: str
    sequence: str
    header: str = ""
    source_group: str = ""
    supporting_sources: list[str] | None = None


def parse_mpnn_fasta(path: Path) -> list[dict[str, object]]:
    entries = []
    header = None
    seq: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header is not None:
                entries.append({"header": header, "sequence": "".join(seq)})
            header = line[1:]
            seq = []
        else:
            seq.append(line.strip())
    if header is not None:
        entries.append({"header": header, "sequence": "".join(seq)})
    return [entry for entry in entries if "sample=" in str(entry["header"])]


def parse_sample_fasta(path: Path) -> list[dict[str, object]]:
    entries = []
    header = None
    seq: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header is not None:
                entries.append({"header": header, "sequence": "".join(seq)})
            header = line[1:]
            seq = []
        else:
            seq.append(line.strip())
    if header is not None:
        entries.append({"header": header, "sequence": "".join(seq)})
    return [entry for entry in entries if "sample=" in str(entry["header"])]


def write_shortlist_fasta(rows: list[dict[str, object]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(f">{row['candidate_id']} {row['mutations']} {row['source']}\n")
            sequence = str(row["sequence"])
            for idx in range(0, len(sequence), 80):
                fh.write(sequence[idx : idx + 80] + "\n")


def classify_source_group(source: str) -> str:
    if source == "manual":
        return "manual_control"
    if source == "local_proposal":
        return "heuristic_local"
    return "learned"


def register_candidate(
    candidates: "OrderedDict[str, dict[str, object]]",
    entry: dict[str, object],
) -> None:
    sequence = str(entry["sequence"])
    source = str(entry["source"])
    source_group = classify_source_group(source)
    if sequence not in candidates:
        new_entry = dict(entry)
        new_entry["source_group"] = source_group
        new_entry["supporting_sources"] = [source]
        candidates[sequence] = new_entry
        return

    current = candidates[sequence]
    current_sources = set(current.get("supporting_sources", []))
    current_sources.add(source)
    current["supporting_sources"] = sorted(current_sources)
    if SOURCE_PRIORITY.get(source, 0) > SOURCE_PRIORITY.get(str(current["source"]), 0):
        current["candidate_id"] = entry["candidate_id"]
        current["source"] = source
        current["source_group"] = source_group
        if "header" in entry:
            current["header"] = entry.get("header", "")


def build_manual_candidates(
    wt_seq: str,
    manual_bias: dict[int, dict[str, float]],
    position_to_index: dict[int, int],
) -> list[dict[str, object]]:
    candidates = [{"candidate_id": "manual_wt", "source": "manual", "sequence": wt_seq}]
    top_by_pos: dict[int, str] = {}
    for pos, aa_bias in manual_bias.items():
        idx = position_to_index[pos]
        ranked = sorted(
            [(aa, val) for aa, val in aa_bias.items() if aa != wt_seq[idx] and val > 0],
            key=lambda item: (-item[1], item[0]),
        )
        if not ranked:
            continue
        top_aa = ranked[0][0]
        top_by_pos[pos] = top_aa
        seq_chars = list(wt_seq)
        seq_chars[idx] = top_aa
        candidates.append(
            {
                "candidate_id": f"manual_{wt_seq[idx]}{pos}{top_aa}",
                "source": "manual",
                "sequence": "".join(seq_chars),
            }
        )

    if top_by_pos:
        seq_chars = list(wt_seq)
        name_parts = []
        for pos in sorted(top_by_pos):
            idx = position_to_index[pos]
            seq_chars[idx] = top_by_pos[pos]
            name_parts.append(f"{wt_seq[idx]}{pos}{top_by_pos[pos]}")
        candidates.append(
            {
                "candidate_id": f"manual_combo_{'_'.join(name_parts)}",
                "source": "manual",
                "sequence": "".join(seq_chars),
            }
        )
    return candidates


def build_local_proposal_candidates(
    wt_seq: str,
    design_positions: list[int],
    position_to_index: dict[int, int],
    features: list[object],
    manual_bias: dict[int, dict[str, float]],
    oxidation_hotspots: list[int],
    flexible_positions: list[int],
    profile: list[dict[str, float]] | None,
    family_recommendations: dict[int, dict[str, float]] | None = None,
    asr_recommendations: dict[int, dict[str, float]] | None = None,
    topic_name: str | None = None,
    topic_cfg: dict[str, object] | None = None,
    max_variants_per_position: int = 5,
    max_candidates: int = 256,
) -> list[dict[str, object]]:
    oxidation_set = set(oxidation_hotspots)
    flexible_set = set(flexible_positions)
    hydrating = {"Q": 0.7, "N": 0.6, "E": 0.45, "D": 0.35, "S": 0.2, "T": 0.2}
    topic_recommendations = build_topic_local_recommendations(
        topic_name=topic_name,
        wt_seq=wt_seq,
        features=features,
        design_positions=design_positions,
        position_to_index=position_to_index,
        topic_cfg=topic_cfg,
    )

    per_position_choices: list[tuple[int, str, list[tuple[str, float]]]] = []
    for pos in design_positions:
        idx = position_to_index[pos]
        wt = wt_seq[idx]
        aa_scores: dict[str, float] = {wt: 0.0}

        for aa, val in manual_bias.get(pos, {}).items():
            aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), float(val))

        if pos in oxidation_set:
            for aa, val in SAFE_OXIDATION_MAP.get(wt, {}).items():
                aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), 0.6 + 0.4 * float(val))

        if pos in flexible_set and pos not in oxidation_set:
            for aa, val in hydrating.items():
                if aa != wt:
                    aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), float(val))

        if profile is not None:
            for aa, prob in sorted(profile[idx].items(), key=lambda item: (-item[1], item[0]))[:4]:
                if aa == "-" or aa == wt:
                    continue
                if pos in oxidation_set:
                    allowed_hotspot = set(SAFE_OXIDATION_MAP.get(wt, {})) | set(manual_bias.get(pos, {}))
                    if aa not in allowed_hotspot:
                        continue
                if prob >= 0.05:
                    aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), 0.4 + float(prob))

        if family_recommendations is not None:
            for aa, delta in family_recommendations.get(pos, {}).items():
                if aa == wt:
                    continue
                if pos in oxidation_set:
                    allowed_hotspot = set(SAFE_OXIDATION_MAP.get(wt, {})) | set(manual_bias.get(pos, {}))
                    if aa not in allowed_hotspot:
                        continue
                aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), 0.45 + 1.2 * float(delta))

        if asr_recommendations is not None:
            for aa, prob in asr_recommendations.get(pos, {}).items():
                if aa == wt:
                    continue
                if pos in oxidation_set:
                    allowed_hotspot = set(SAFE_OXIDATION_MAP.get(wt, {})) | set(manual_bias.get(pos, {}))
                    if aa not in allowed_hotspot:
                        continue
                aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), 0.35 + 1.1 * float(prob))

        for aa, bias in topic_recommendations.get(pos, {}).items():
            if aa == wt and aa not in aa_scores:
                aa_scores[aa] = 0.0
            aa_scores[aa] = max(aa_scores.get(aa, float("-inf")), float(bias))

        ranked = sorted(aa_scores.items(), key=lambda item: (-item[1], item[0]))
        trimmed = ranked[:max_variants_per_position]
        if wt not in {aa for aa, _ in trimmed}:
            trimmed = [(wt, 0.0)] + trimmed[: max_variants_per_position - 1]
        per_position_choices.append((pos, wt, trimmed))

    seq_entries: list[tuple[float, str, str]] = []
    choice_products = [choices for _, _, choices in per_position_choices]
    total_states = 1
    for choices in choice_products:
        total_states *= max(1, len(choices))
    if total_states > 20000:
        raise ValueError(f"Local proposal branch would enumerate too many states: {total_states}")

    for combo in product(*choice_products):
        seq_chars = list(wt_seq)
        name_parts: list[str] = []
        local_priority = 0.0
        mutation_count = 0
        for (pos, wt, _), (aa, aa_score) in zip(per_position_choices, combo):
            idx = position_to_index[pos]
            seq_chars[idx] = aa
            if aa != wt:
                mutation_count += 1
                name_parts.append(f"{wt}{pos}{aa}")
                local_priority += aa_score
        if mutation_count == 0:
            continue
        local_priority -= 0.15 * max(0, mutation_count - 1)
        seq_entries.append((local_priority, "".join(name_parts), "".join(seq_chars)))

    seq_entries.sort(key=lambda item: (-item[0], item[1]))
    shortlisted = seq_entries[:max_candidates]
    return [
        {
            "candidate_id": f"local_{idx:03d}",
            "source": "local_proposal",
            "sequence": seq,
        }
        for idx, (_, _, seq) in enumerate(shortlisted, start=1)
    ]
