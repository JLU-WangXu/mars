from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import OrderedDict
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
HYBRID_SELECTION_TOLERANCE = 0.10
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import marsstack.field_network.proposals as proposal_ops

from marsstack.evolution import (
    build_family_pair_profiles,
    build_structure_position_weights,
    build_profile,
    build_profile_from_homologs,
    differential_family_recommendations,
    load_aligned_fasta,
    load_fasta,
    load_yaml,
    profile_recommendations,
    write_fasta,
)
from marsstack.ancestral_field import build_ancestral_posterior_field
from marsstack.decoder import ConstrainedBeamDecoder
from marsstack.energy_head import serialize_pairwise_energy_tensor
from marsstack.evidence_field import serialize_evidence_fields
from marsstack.field_network import (
    EvidencePaths,
    MarsFieldSystem,
    ProteinDesignContext,
    build_neural_residue_field,
    build_runtime_neural_target_batch,
    ScoringInputs,
    score_candidate_rows,
    train_holdout_neural_model,
)
from marsstack.fusion_ranker import apply_learned_fusion_ranking, rank_rows_with_model
from marsstack.mars_score import SAFE_OXIDATION_MAP, mutation_list, score_candidate
from marsstack.topic_score import build_topic_local_recommendations
from marsstack.structure_features import (
    analyze_structure,
    detect_flexible_surface_positions,
    detect_oxidation_hotspots,
)


VENDOR = ROOT / "vendors" / "ProteinMPNN"
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
SOURCE_PRIORITY = {
    "manual": 1,
    "baseline_mpnn": 2,
    "local_proposal": 3,
    "mars_mpnn": 4,
    "esm_if": 5,
    "fusion_decoder": 6,
    "neural_decoder": 7,
}


def run(cmd: list[str], cwd: Path) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    dataset_candidate = (ROOT / "datasets" / path).resolve()
    if dataset_candidate.exists():
        return dataset_candidate
    return (ROOT / path).resolve()


def preprocess_pdb(
    src_path: Path,
    dst_path: Path,
    residue_renames: list[dict[str, object]] | None = None,
) -> Path:
    if not residue_renames:
        dst_path.write_bytes(src_path.read_bytes())
        return dst_path

    rename_map: dict[tuple[str, int, str | None], str] = {}
    for item in residue_renames:
        chain = str(item["chain"])
        residue_number = int(item["residue_number"])
        from_name = str(item["from_name"]).upper() if item.get("from_name") else None
        to_name = str(item["to_name"]).upper()
        rename_map[(chain, residue_number, from_name)] = to_name

    out_lines: list[str] = []
    for line in src_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            chain = line[21].strip()
            try:
                residue_number = int(line[22:26].strip())
            except ValueError:
                residue_number = None
            resname = line[17:20].strip().upper()
            replacement = None
            if residue_number is not None:
                replacement = rename_map.get((chain, residue_number, resname))
                if replacement is None:
                    replacement = rename_map.get((chain, residue_number, None))
            if replacement is not None:
                line = "ATOM  " + line[6:]
                line = line[:17] + f"{replacement:>3}" + line[20:]
        out_lines.append(line)

    dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return dst_path


def normalize_parsed_names(parsed_jsonl: Path) -> None:
    lines = []
    with parsed_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            obj["name"] = Path(obj["name"]).stem
            lines.append(obj)
    with parsed_jsonl.open("w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")


def load_parsed_chain_sequence(parsed_jsonl: Path, chain: str) -> str:
    with parsed_jsonl.open("r", encoding="utf-8") as fh:
        first = json.loads(next(fh))
    return first[f"seq_chain_{chain}"]


def build_parsed_index_maps(parsed_chain_seq: str, residue_numbers: list[int]) -> tuple[dict[int, int], list[int]]:
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
    if parsed_keep_indices and max(parsed_keep_indices) >= len(seq):
        raise ValueError("ProteinMPNN sequence is shorter than expected parsed template length.")
    return "".join(seq[idx] for idx in parsed_keep_indices)


def restore_template_mismatches(
    seq: str,
    wt_seq: str,
    mismatch_positions: list[int],
    position_to_index: dict[int, int],
) -> str:
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


def parse_mpnn_fasta(path: Path) -> list[dict[str, object]]:
    entries = []
    header = None
    seq = []
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
    return [e for e in entries if "sample=" in e["header"]]


def parse_sample_fasta(path: Path) -> list[dict[str, object]]:
    entries = []
    header = None
    seq = []
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
    return [e for e in entries if "sample=" in e["header"]]


def write_shortlist_fasta(rows: list[dict[str, object]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(f">{row['candidate_id']} {row['mutations']} {row['source']}\n")
            seq = row["sequence"]
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")


def classify_source_group(source: str) -> str:
    if source == "manual":
        return "manual_control"
    if source == "local_proposal":
        return "heuristic_local"
    return "learned"


def summarize_aligned_entries(entries: list[tuple[str, str]], wt_seq: str) -> tuple[int, float]:
    if not entries:
        return 0, 0.0
    accepted = max(0, len(entries) - 1)
    if accepted <= 0:
        return accepted, 0.0
    coverage = [
        sum(1 for _, seq in entries[1:] if seq[i] != "-")
        for i in range(len(wt_seq))
    ]
    mean_coverage = round(sum(coverage) / max(1, len(coverage) * accepted), 3)
    return accepted, mean_coverage


def merge_recommendation_maps(
    *maps: dict[int, dict[str, float]] | None,
    top_k: int = 4,
) -> dict[int, dict[str, float]]:
    merged: dict[int, dict[str, float]] = {}
    for rec_map in maps:
        if not rec_map:
            continue
        for pos, aa_scores in rec_map.items():
            bucket = merged.setdefault(int(pos), {})
            for aa, score in aa_scores.items():
                bucket[aa] = round(max(float(score), float(bucket.get(aa, 0.0))), 4)

    final: dict[int, dict[str, float]] = {}
    for pos, aa_scores in merged.items():
        ranked = sorted(aa_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        if ranked:
            final[pos] = {aa: score for aa, score in ranked}
    return final


def register_candidate(
    candidates: "OrderedDict[str, dict[str, object]]",
    entry: dict[str, object],
) -> None:
    seq = str(entry["sequence"])
    source = str(entry["source"])
    source_group = classify_source_group(source)
    if seq not in candidates:
        new_entry = dict(entry)
        new_entry["source_group"] = source_group
        new_entry["supporting_sources"] = [source]
        candidates[seq] = new_entry
        return

    current = candidates[seq]
    current_sources = set(current.get("supporting_sources", []))
    current_sources.add(source)
    current["supporting_sources"] = sorted(current_sources)
    if SOURCE_PRIORITY.get(source, 0) > SOURCE_PRIORITY.get(str(current["source"]), 0):
        current["candidate_id"] = entry["candidate_id"]
        current["source"] = source
        current["source_group"] = source_group
        if "header" in entry:
            current["header"] = entry.get("header", "")


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
            key=lambda x: (-x[1], x[0]),
        )
        if not ranked:
            continue
        top_aa = ranked[0][0]
        top_by_pos[pos] = top_aa
        seq = list(wt_seq)
        seq[idx] = top_aa
        candidates.append(
            {
                "candidate_id": f"manual_{wt_seq[idx]}{pos}{top_aa}",
                "source": "manual",
                "sequence": "".join(seq),
            }
        )

    if top_by_pos:
        seq = list(wt_seq)
        name_parts = []
        for pos in sorted(top_by_pos):
            idx = position_to_index[pos]
            seq[idx] = top_by_pos[pos]
            name_parts.append(f"{wt_seq[idx]}{pos}{top_by_pos[pos]}")
        candidates.append(
            {
                "candidate_id": f"manual_combo_{'_'.join(name_parts)}",
                "source": "manual",
                "sequence": "".join(seq),
            }
        )
    return candidates


def materialize_decoded_candidate_rows(
    *,
    decoded_candidates: list[object],
    source_name: str,
    wt_seq: str,
    features: list[object],
    oxidation_hotspots: list[int],
    flexible_positions: list[int],
    profile: list[dict[str, float]] | None,
    asr_profile: list[dict[str, float]] | None,
    family_positive_profile: list[dict[str, float]] | None,
    family_negative_profile: list[dict[str, float]] | None,
    manual_bias: dict[int, dict[str, float]],
    design_positions: list[int],
    score_weights: dict[str, float],
    position_to_index: dict[int, int],
    evolution_position_weights: dict[int, float],
    residue_numbers: list[int],
    evo_cfg: dict[str, object],
    topic_name: str,
    topic_cfg: dict[str, object],
    existing_sequences: set[str],
    min_mars_score: float,
    min_support_count: int,
    max_mars_gap_vs_best: float,
    max_mars_gap_vs_best_learned: float,
    best_existing_mars_score: float,
    best_existing_learned_mars_score: float,
    skip_bad_hotspots: bool = True,
) -> tuple[list[dict[str, object]], int, dict[str, int]]:
    generated_rows: list[dict[str, object]] = []
    rejected_count = 0
    rejection_reasons = {
        "bad_hotspot": 0,
        "low_mars_score": 0,
        "low_support": 0,
        "mars_gap_vs_best": 0,
        "mars_gap_vs_best_learned": 0,
    }
    for idx, item in enumerate(decoded_candidates, start=1):
        if item.sequence in existing_sequences:
            continue
        res = score_candidate(
            wt_seq=wt_seq,
            seq=item.sequence,
            features=features,
            oxidation_hotspots=oxidation_hotspots,
            flexible_positions=flexible_positions,
            profile=profile,
            asr_profile=asr_profile,
            family_positive_profile=family_positive_profile,
            family_negative_profile=family_negative_profile,
            manual_preferred=manual_bias,
            evolution_positions=design_positions,
            mutable_positions=design_positions,
            term_weights=score_weights,
            position_to_index=position_to_index,
            evolution_position_weights=evolution_position_weights,
            residue_numbers=residue_numbers,
            profile_prior_scale=float(evo_cfg.get("profile_prior_scale", 0.35)),
            asr_prior_scale=float(evo_cfg.get("asr_prior_scale", 0.45)),
            family_prior_scale=float(evo_cfg.get("family_prior_scale", 0.60)),
            topic_name=topic_name,
            topic_cfg=topic_cfg,
        )
        note_items = list(res.notes)
        if skip_bad_hotspots and any(note.startswith("bad_hotspot_choice_") for note in note_items):
            rejected_count += 1
            rejection_reasons["bad_hotspot"] += 1
            continue
        if res.total < min_mars_score:
            rejected_count += 1
            rejection_reasons["low_mars_score"] += 1
            continue
        if len(item.supporting_sources) < min_support_count:
            rejected_count += 1
            rejection_reasons["low_support"] += 1
            continue
        if (best_existing_mars_score - res.total) > max_mars_gap_vs_best:
            rejected_count += 1
            rejection_reasons["mars_gap_vs_best"] += 1
            continue
        if (best_existing_learned_mars_score - res.total) > max_mars_gap_vs_best_learned:
            rejected_count += 1
            rejection_reasons["mars_gap_vs_best_learned"] += 1
            continue
        header_prefix = "decoder" if source_name == "fusion_decoder" else source_name
        generated_rows.append(
            {
                "candidate_id": f"{source_name}_{idx:03d}",
                "source": source_name,
                "source_group": "learned",
                "supporting_sources": ";".join([source_name] + list(item.supporting_sources)),
                "mutations": ";".join(mutation_list(wt_seq, item.sequence, residue_numbers=residue_numbers)) or "WT",
                "mars_score": res.total,
                "notes": ";".join(note_items),
                "sequence": item.sequence,
                "header": f"{header_prefix}_score={item.decoder_score};mutation_count={item.mutation_count};support_count={len(item.supporting_sources)}",
                "score_oxidation": res.components["oxidation"],
                "score_surface": res.components["surface"],
                "score_manual": res.components["manual"],
                "score_evolution": res.components["evolution"],
                "score_burden": res.components["burden"],
                "score_topic_sequence": res.components["topic_sequence"],
                "score_topic_structure": res.components["topic_structure"],
                "score_topic_evolution": res.components["topic_evolution"],
            }
        )
        existing_sequences.add(item.sequence)
    return generated_rows, rejected_count, rejection_reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "calb_1lbt.yaml")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--decoder-enabled", choices=["true", "false"], default=None)
    parser.add_argument("--neural-rerank", choices=["true", "false"], default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    protein = cfg["protein"]
    generation = cfg["generation"]
    method = cfg["method"]
    evo_cfg = cfg.get("evolution", {})
    topic_cfg = cfg.get("topic", {})
    preprocess_cfg = protein.get("preprocess", {})
    local_cfg = method.get("local_proposals", {})
    learned_fusion_cfg = dict(method.get("learned_fusion", {}) or {})
    neural_rerank_cfg = dict(method.get("neural_rerank", {}) or {})
    esm_if_cfg = generation.get("esm_if", {})
    mpnn_cfg = generation.get("protein_mpnn", {})
    mpnn_enabled = bool(mpnn_cfg.get("enabled", True))
    decoder_enabled = bool(learned_fusion_cfg.get("decoder_enabled", True))
    if args.decoder_enabled is not None:
        decoder_enabled = args.decoder_enabled.lower() == "true"
        learned_fusion_cfg["decoder_enabled"] = decoder_enabled
    neural_rerank_enabled = bool(neural_rerank_cfg.get("enabled", False))
    if args.neural_rerank is not None:
        neural_rerank_enabled = args.neural_rerank.lower() == "true"
        neural_rerank_cfg["enabled"] = neural_rerank_enabled
    neural_rerank_epochs = int(neural_rerank_cfg.get("epochs", 1))
    neural_rerank_lr = float(neural_rerank_cfg.get("lr", 1e-3))
    neural_selection_policy = str(neural_rerank_cfg.get("selection_policy", "current"))
    neural_decoder_enabled = bool(neural_rerank_cfg.get("decoder_enabled", neural_rerank_enabled))
    neural_decoder_top_k_per_position = int(neural_rerank_cfg.get("decoder_top_k_per_position", 4))
    neural_decoder_pair_top_k = int(neural_rerank_cfg.get("decoder_pair_top_k", 32))
    neural_decoder_beam_size = int(neural_rerank_cfg.get("decoder_beam_size", learned_fusion_cfg.get("decoder_beam_size", 32)))
    neural_decoder_max_candidates = int(neural_rerank_cfg.get("decoder_max_candidates", learned_fusion_cfg.get("decoder_max_candidates", 32)))
    neural_decoder_mutation_penalty = float(neural_rerank_cfg.get("decoder_mutation_penalty", learned_fusion_cfg.get("decoder_mutation_penalty", 0.15)))

    name = protein["name"]
    pdb_path = Path(protein["pdb_path"])
    if not pdb_path.is_absolute():
        pdb_path = (ROOT / pdb_path).resolve()
    chain = protein["chain"]
    wt_seq = protein["wt_sequence"]
    protected_positions = set(protein["protected_positions"])
    design_positions = [int(x) for x in generation["design_positions"]]
    manual_bias = {int(k): v for k, v in method["manual_bias"].items()}
    score_weights = {str(k): float(v) for k, v in method.get("score_weights", {}).items()}
    topic_enabled = bool(topic_cfg.get("enabled", False))
    topic_name = str(topic_cfg.get("name", "")).strip().lower() if topic_enabled else ""
    pdb_stem = pdb_path.stem

    out_root = ROOT / "outputs" / f"{name.lower()}_pipeline"
    input_dir = out_root / "input_pdbs"
    baseline_dir = out_root / "baseline_mpnn"
    mars_dir = out_root / "mars_mpnn"
    input_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    mars_dir.mkdir(parents=True, exist_ok=True)
    esm_if_dir = out_root / "esm_if"
    if esm_if_cfg.get("enabled", False):
        esm_if_dir.mkdir(parents=True, exist_ok=True)

    copied_pdb = input_dir / pdb_path.name
    working_pdb = preprocess_pdb(
        src_path=pdb_path,
        dst_path=copied_pdb,
        residue_renames=preprocess_cfg.get("residue_renames"),
    )

    parsed_jsonl = out_root / "parsed_pdbs.jsonl"
    assigned_jsonl = out_root / "assigned_pdbs.jsonl"
    fixed_jsonl = out_root / "fixed_pdbs.jsonl"
    bias_jsonl = out_root / "bias_by_res.jsonl"
    omit_jsonl = out_root / "omit_aa.jsonl"

    features = analyze_structure(
        pdb_path=working_pdb,
        chain_id=chain,
        protected_positions=protected_positions,
    )
    residue_numbers = [f.num for f in features]
    position_to_index = {num: idx for idx, num in enumerate(residue_numbers)}
    feature_rows = [vars(f) for f in features]
    pd.DataFrame(feature_rows).to_csv(out_root / "structure_features.csv", index=False)

    oxidation_hotspots = detect_oxidation_hotspots(
        features,
        min_sasa=float(method["oxidation_min_sasa"]),
        min_dist_protected=float(method["oxidation_min_dist_protected"]),
    )
    flexible_positions = detect_flexible_surface_positions(
        features,
        min_sasa=float(method["flexible_surface_min_sasa"]),
    )
    flexible_positions = sorted(set(flexible_positions) | set(design_positions))

    summary = {
        "protein": name,
        "chain": chain,
        "oxidation_hotspots": oxidation_hotspots,
        "flexible_surface_positions": flexible_positions,
        "design_positions": design_positions,
    }
    (out_root / "feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mars_context = ProteinDesignContext(
        target=name,
        pdb_path=working_pdb,
        chain_id=chain,
        wt_sequence=wt_seq,
        design_positions=design_positions,
        protected_positions=protected_positions,
        position_to_index=position_to_index,
        score_weights=score_weights,
        metadata={"manual_bias": manual_bias},
    )
    mars_system = MarsFieldSystem(outputs_root=ROOT / "outputs")

    template_mismatch_positions: list[int] = []
    parsed_keep_indices: list[int] = []
    if mpnn_enabled:
        run(
            [
                sys.executable,
                "helper_scripts/parse_multiple_chains.py",
                "--input_path",
                str(input_dir),
                "--output_path",
                str(parsed_jsonl),
            ],
            cwd=VENDOR,
        )
        normalize_parsed_names(parsed_jsonl)
        parsed_chain_seq = load_parsed_chain_sequence(parsed_jsonl, chain)
        position_to_parsed_index, parsed_keep_indices = build_parsed_index_maps(parsed_chain_seq, residue_numbers)
        parsed_design_positions = [position_to_parsed_index[pos] + 1 for pos in design_positions]
        parsed_actual_seq = collapse_mpnn_sequence(parsed_chain_seq, parsed_keep_indices)
        if len(parsed_actual_seq) != len(wt_seq):
            raise ValueError("Collapsed parsed template sequence does not match WT sequence length.")
        template_mismatch_positions = [
            residue_numbers[i]
            for i, (wt_aa, pdb_aa) in enumerate(zip(wt_seq, parsed_actual_seq))
            if wt_aa != pdb_aa and residue_numbers[i] not in design_positions
        ]

        run(
            [
                sys.executable,
                "helper_scripts/assign_fixed_chains.py",
                "--input_path",
                str(parsed_jsonl),
                "--output_path",
                str(assigned_jsonl),
                "--chain_list",
                chain,
            ],
            cwd=VENDOR,
        )

        run(
            [
                sys.executable,
                "helper_scripts/make_fixed_positions_dict.py",
                "--input_path",
                str(parsed_jsonl),
                "--output_path",
                str(fixed_jsonl),
                "--chain_list",
                chain,
                "--position_list",
                " ".join(str(x) for x in parsed_design_positions),
                "--specify_non_fixed",
            ],
            cwd=VENDOR,
        )

        build_bias_and_omit(
            protein_name=pdb_stem,
            chain=chain,
            seq_len=len(parsed_chain_seq),
            manual_bias=manual_bias,
            oxidation_hotspots=oxidation_hotspots,
            wt_seq=wt_seq,
            position_to_index=position_to_index,
            position_to_parsed_index=position_to_parsed_index,
            bias_out=bias_jsonl,
            omit_out=omit_jsonl,
        )

        base_cmd = [
            sys.executable,
            "protein_mpnn_run.py",
            "--path_to_model_weights",
            str(VENDOR / ("soluble_model_weights" if generation["use_soluble_model"] else "vanilla_model_weights")),
            "--jsonl_path",
            str(parsed_jsonl),
            "--chain_id_jsonl",
            str(assigned_jsonl),
            "--fixed_positions_jsonl",
            str(fixed_jsonl),
            "--num_seq_per_target",
            str(generation["num_seq_per_target"]),
            "--sampling_temp",
            generation["sampling_temp"],
            "--seed",
            str(generation["seed"]),
            "--batch_size",
            str(generation["batch_size"]),
            "--suppress_print",
            "0",
        ]
        if generation["use_soluble_model"]:
            base_cmd.append("--use_soluble_model")

        run(base_cmd + ["--out_folder", str(baseline_dir)], cwd=VENDOR)
        run(
            base_cmd
            + [
                "--out_folder",
                str(mars_dir),
                "--bias_by_res_jsonl",
                str(bias_jsonl),
                "--omit_AA_jsonl",
                str(omit_jsonl),
            ],
            cwd=VENDOR,
        )

    esm_if_fasta = esm_if_dir / f"{pdb_stem}_esm_if.fa"
    if esm_if_cfg.get("enabled", False):
        esm_if_python = Path(str(esm_if_cfg.get("python_executable") or sys.executable))
        esm_if_cmd = [
            str(esm_if_python),
            str(ROOT / "scripts" / "run_esm_if_generator.py"),
            "--pdbfile",
            str(working_pdb),
            "--chain",
            chain,
            "--outpath",
            str(esm_if_fasta),
            "--num-samples",
            str(int(esm_if_cfg.get("num_samples", generation["num_seq_per_target"]))),
            "--temperature",
            str(float(esm_if_cfg.get("temperature", 1e-6))),
        ]
        if esm_if_cfg.get("multichain_backbone", False):
            esm_if_cmd.append("--multichain-backbone")
        if esm_if_cfg.get("nogpu", False):
            esm_if_cmd.append("--nogpu")
        if esm_if_cfg.get("esm_root"):
            esm_if_cmd.extend(["--esm-root", str((ROOT / esm_if_cfg["esm_root"]).resolve())])
        run(esm_if_cmd, cwd=ROOT)

    template_weight_cfg = dict(evo_cfg.get("template_weighting", {}) or {})
    evidence_paths = EvidencePaths(
        homologs_fasta=Path(str(evo_cfg["homologs_fasta"])) if evo_cfg.get("homologs_fasta") else None,
        aligned_fasta=Path(str(evo_cfg["aligned_fasta"])) if evo_cfg.get("aligned_fasta") else None,
        asr_fasta=Path(str(evo_cfg["asr_fasta"])) if evo_cfg.get("asr_fasta") else None,
        asr_aligned_fasta=Path(str(evo_cfg["asr_aligned_fasta"])) if evo_cfg.get("asr_aligned_fasta") else None,
        family_manifest=Path(str(evo_cfg["family_manifest"])) if evo_cfg.get("family_manifest") else None,
        template_structure_path=Path(str(evo_cfg["template_structure_path"])) if evo_cfg.get("template_structure_path") else None,
    )
    evidence_bundle = mars_system.build_evidence(
        context=mars_context,
        paths=evidence_paths,
        oxidation_min_sasa=float(method["oxidation_min_sasa"]),
        oxidation_min_dist_protected=float(method["oxidation_min_dist_protected"]),
        flexible_surface_min_sasa=float(method["flexible_surface_min_sasa"]),
        min_identity=float(evo_cfg.get("min_identity", 0.20)),
        family_min_identity=float(evo_cfg.get("family_min_identity", evo_cfg.get("min_identity", 0.20))),
        family_top_k=int(evo_cfg.get("family_top_k", 3)),
        family_min_delta=float(evo_cfg.get("family_min_delta", 0.05)),
        asr_top_k=int(evo_cfg.get("asr_top_k", 2)),
        asr_min_prob=float(evo_cfg.get("asr_min_prob", 0.20)),
        template_weight_cfg=template_weight_cfg,
    )
    profile = evidence_bundle.evolution.homolog_profile
    family_positive_profile = evidence_bundle.evolution.family_positive_profile
    family_negative_profile = evidence_bundle.evolution.family_negative_profile
    family_recommendations = evidence_bundle.evolution.family_recommendations
    evolution_position_weights = evidence_bundle.evolution.position_weights
    aligned_entries = evidence_bundle.evolution.aligned_entries
    asr_profile = evidence_bundle.ancestral.asr_profile
    asr_recommendations = evidence_bundle.ancestral.recommendations
    asr_aligned_entries = evidence_bundle.ancestral.aligned_entries
    ancestral_field = evidence_bundle.ancestral.ancestral_field
    profile_summary = dict(evidence_bundle.evolution.profile_summary)
    profile_summary.update(
        {
            "asr_prior_enabled": bool(asr_profile is not None),
            "input_asr": int(evidence_bundle.ancestral.diagnostics.get("input_asr", 0)),
            "accepted_asr": int(evidence_bundle.ancestral.diagnostics.get("accepted_asr", 0)),
            "mean_asr_coverage": float(evidence_bundle.ancestral.diagnostics.get("mean_asr_coverage", 0.0)),
        }
    )
    combined_prior_recommendations = merge_recommendation_maps(
        family_recommendations,
        asr_recommendations,
        top_k=max(
            int(evo_cfg.get("family_top_k", 3)),
            int(evo_cfg.get("asr_top_k", 2)),
        ),
    )
    if aligned_entries:
        write_fasta(aligned_entries, out_root / "aligned_homologs.fasta")
    if asr_aligned_entries:
        write_fasta(asr_aligned_entries, out_root / "aligned_asr.fasta")
    (out_root / "profile_summary.json").write_text(json.dumps(profile_summary, indent=2), encoding="utf-8")
    (out_root / "prior_recommendations.json").write_text(
        json.dumps(
            {
                "family_recommendations": family_recommendations,
                "asr_recommendations": asr_recommendations,
                "combined_recommendations": combined_prior_recommendations,
                "ancestral_field": ancestral_field,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_root / "ancestral_field.json").write_text(json.dumps(ancestral_field, indent=2), encoding="utf-8")

    seed_proposals: list[dict[str, object]] = []

    if mpnn_enabled:
        for source, fa_path in [
            ("baseline_mpnn", baseline_dir / "seqs" / f"{pdb_stem}.fa"),
            ("mars_mpnn", mars_dir / "seqs" / f"{pdb_stem}.fa"),
        ]:
            if not fa_path.exists():
                continue
            for idx, entry in enumerate(proposal_ops.parse_mpnn_fasta(fa_path), start=1):
                collapsed_seq = collapse_mpnn_sequence(entry["sequence"], parsed_keep_indices)
                restored_seq = restore_template_mismatches(
                    collapsed_seq,
                    wt_seq,
                    template_mismatch_positions,
                    position_to_index,
                )
                seed_proposals.append(
                    {
                        "candidate_id": f"{source}_{idx:03d}",
                        "source": source,
                        "sequence": restored_seq,
                        "header": entry["header"],
                    }
                )

    if esm_if_cfg.get("enabled", False) and esm_if_fasta.exists():
        for idx, entry in enumerate(proposal_ops.parse_sample_fasta(esm_if_fasta), start=1):
            if len(entry["sequence"]) != len(wt_seq):
                continue
            restored_seq = restore_template_mismatches(
                entry["sequence"],
                wt_seq,
                template_mismatch_positions,
                position_to_index,
            )
            projected_seq = project_to_design_positions(
                restored_seq,
                wt_seq,
                design_positions,
                position_to_index,
            )
            seed_proposals.append(
                {
                    "candidate_id": f"esm_if_{idx:03d}",
                    "source": "esm_if",
                    "sequence": projected_seq,
                    "header": entry["header"],
                }
            )

    candidates = mars_system.generate_candidates(
        context=mars_context,
        manual_bias=manual_bias,
        geometric_features=features,
        proposal_rows=seed_proposals,
        profile=profile,
        family_recommendations=family_recommendations,
        asr_recommendations=asr_recommendations,
        topic_name=topic_name,
        topic_cfg=topic_cfg,
        local_enabled=bool(local_cfg.get("enabled", True)),
        local_max_variants_per_position=int(local_cfg.get("max_variants_per_position", 5)),
        local_max_candidates=int(local_cfg.get("max_candidates", 256)),
    )

    scoring_inputs = ScoringInputs(
        wt_seq=wt_seq,
        features=features,
        oxidation_hotspots=oxidation_hotspots,
        flexible_positions=flexible_positions,
        profile=profile,
        asr_profile=asr_profile,
        family_positive_profile=family_positive_profile,
        family_negative_profile=family_negative_profile,
        manual_preferred=manual_bias,
        design_positions=design_positions,
        term_weights=score_weights,
        position_to_index=position_to_index,
        evolution_position_weights=evolution_position_weights,
        residue_numbers=residue_numbers,
        profile_prior_scale=float(evo_cfg.get("profile_prior_scale", 0.35)),
        asr_prior_scale=float(evo_cfg.get("asr_prior_scale", 0.45)),
        family_prior_scale=float(evo_cfg.get("family_prior_scale", 0.60)),
        topic_name=topic_name,
        topic_cfg=topic_cfg,
    )
    rows = mars_system.score_candidates(
        candidates=list(candidates.values()),
        scoring=scoring_inputs,
    )

    rows, fusion_summary, fusion_model = apply_learned_fusion_ranking(
        rows=rows,
        protein_name=name,
        feature_summary=summary,
        profile_summary=profile_summary,
        outputs_root=ROOT / "outputs",
        cfg_raw=learned_fusion_cfg,
    )
    for row in rows:
        row["selection_score"] = float(row.get("ranking_score", row.get("mars_score", 0.0)))
        row["selection_score_name"] = "ranking_score" if "ranking_score" in row else "mars_score"
        row["engineering_score"] = float(row.get("mars_score", 0.0))
    field_build = mars_system.construct_field(
        bundle=evidence_bundle,
        proposal_rows=rows,
    )
    position_fields = field_build.field.position_fields
    pairwise_tensor = field_build.field.pairwise_tensor
    decoded_candidates = []
    neural_position_fields = []
    neural_pairwise_tensor = {}
    neural_decoded_candidates = []
    neural_field_runtime_summary: dict[str, object] = {
        "enabled": False,
    }
    if decoder_enabled:
        decoded_candidates = mars_system.decode(
            context=mars_context,
            field=field_build.field,
            beam_size=int(learned_fusion_cfg.get("decoder_beam_size", 32)),
            max_candidates=int(learned_fusion_cfg.get("decoder_max_candidates", 32)),
            mutation_penalty=float(learned_fusion_cfg.get("decoder_mutation_penalty", 0.15)),
        )
    if neural_decoder_enabled:
        runtime_profile_summary = dict(profile_summary)
        runtime_profile_summary["oxidation_hotspots"] = oxidation_hotspots
        runtime_profile_summary["flexible_surface_positions"] = flexible_positions
        runtime_batch = build_runtime_neural_target_batch(
            target=name,
            pipeline_dir=out_root,
            feature_rows=feature_rows,
            ranked_rows=rows,
            retrieval_payload={
                "recommendations": field_build.bundle.retrieval.recommendations,
                "neighbors": field_build.bundle.retrieval.diagnostics,
                "atlas": field_build.bundle.retrieval.atlas,
            },
            ancestral_payload=field_build.bundle.ancestral.ancestral_field,
            pair_payload=serialize_pairwise_energy_tensor(pairwise_tensor),
            position_fields=position_fields,
            profile_summary=runtime_profile_summary,
        )
        if runtime_batch is not None:
            neural_model, neural_history, neural_training_targets = train_holdout_neural_model(
                outputs_root=ROOT / "outputs",
                holdout_batch=runtime_batch,
                epochs=neural_rerank_epochs,
                lr=neural_rerank_lr,
            )
            neural_position_fields, neural_pairwise_tensor, neural_field_diagnostics = build_neural_residue_field(
                model=neural_model,
                batch=runtime_batch,
                top_k_per_position=neural_decoder_top_k_per_position,
                pair_top_k=neural_decoder_pair_top_k,
                prior_position_fields=position_fields,
                prior_pairwise=pairwise_tensor,
            )
            neural_decoder = ConstrainedBeamDecoder(
                beam_size=neural_decoder_beam_size,
                max_candidates=neural_decoder_max_candidates,
                mutation_penalty=neural_decoder_mutation_penalty,
            )
            neural_decoded_candidates = neural_decoder.decode(
                wt_seq=wt_seq,
                position_to_index=position_to_index,
                fields=neural_position_fields,
                pairwise_energies=neural_pairwise_tensor,
            )
            neural_field_runtime_summary = {
                "enabled": True,
                "training_targets": neural_training_targets,
                "training_target_count": len(neural_training_targets),
                "epochs": neural_rerank_epochs,
                "lr": neural_rerank_lr,
                "decoder_beam_size": neural_decoder_beam_size,
                "decoder_max_candidates": neural_decoder_max_candidates,
                "generated_count": len(neural_decoded_candidates),
                "field_diagnostics": neural_field_diagnostics,
                "history": neural_history,
            }
    (out_root / "position_fields.json").write_text(
        json.dumps(serialize_evidence_fields(position_fields), indent=2),
        encoding="utf-8",
    )
    (out_root / "pairwise_energy_tensor.json").write_text(
        json.dumps(serialize_pairwise_energy_tensor(pairwise_tensor), indent=2),
        encoding="utf-8",
    )
    (out_root / "retrieval_memory_hits.json").write_text(
        json.dumps(
            {
                "recommendations": field_build.bundle.retrieval.recommendations,
                "neighbors": field_build.bundle.retrieval.diagnostics,
                "atlas": field_build.bundle.retrieval.atlas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_root / "ancestral_field.json").write_text(
        json.dumps(field_build.bundle.ancestral.ancestral_field, indent=2),
        encoding="utf-8",
    )
    (out_root / "decoder_preview.json").write_text(
        json.dumps(
            [
                {
                    "sequence": item.sequence,
                    "mutations": item.mutations,
                    "decoder_score": item.decoder_score,
                    "mutation_count": item.mutation_count,
                    "supporting_sources": item.supporting_sources,
                }
                for item in decoded_candidates
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_root / "neural_position_fields.json").write_text(
        json.dumps(serialize_evidence_fields(neural_position_fields), indent=2),
        encoding="utf-8",
    )
    (out_root / "neural_pairwise_energy_tensor.json").write_text(
        json.dumps(serialize_pairwise_energy_tensor(neural_pairwise_tensor), indent=2),
        encoding="utf-8",
    )
    (out_root / "neural_decoder_preview.json").write_text(
        json.dumps(
            [
                {
                    "sequence": item.sequence,
                    "mutations": item.mutations,
                    "decoder_score": item.decoder_score,
                    "mutation_count": item.mutation_count,
                    "supporting_sources": item.supporting_sources,
                }
                for item in neural_decoded_candidates
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_root / "neural_field_runtime_summary.json").write_text(
        json.dumps(neural_field_runtime_summary, indent=2),
        encoding="utf-8",
    )
    decoder_generated_rows = []
    neural_decoder_generated_rows = []
    existing_sequences = {str(row["sequence"]) for row in rows}
    decoder_skip_bad_hotspots = bool(learned_fusion_cfg.get("decoder_skip_bad_hotspots", True))
    decoder_min_mars_score = float(learned_fusion_cfg.get("decoder_min_mars_score", 0.5))
    decoder_min_support_count = int(learned_fusion_cfg.get("decoder_min_support_count", 3))
    decoder_max_mars_gap_vs_best = float(learned_fusion_cfg.get("decoder_max_mars_gap_vs_best", 2.0))
    decoder_max_mars_gap_vs_best_learned = float(learned_fusion_cfg.get("decoder_max_mars_gap_vs_best_learned", 1.0))
    neural_decoder_min_mars_score = float(neural_rerank_cfg.get("decoder_min_mars_score", decoder_min_mars_score))
    neural_decoder_min_support_count = int(neural_rerank_cfg.get("decoder_min_support_count", 1))
    neural_decoder_max_mars_gap_vs_best = float(neural_rerank_cfg.get("decoder_max_mars_gap_vs_best", decoder_max_mars_gap_vs_best))
    neural_decoder_max_mars_gap_vs_best_learned = float(neural_rerank_cfg.get("decoder_max_mars_gap_vs_best_learned", decoder_max_mars_gap_vs_best_learned))
    decoder_rejected_count = 0
    decoder_rejection_reasons = {
        "bad_hotspot": 0,
        "low_mars_score": 0,
        "low_support": 0,
        "mars_gap_vs_best": 0,
        "mars_gap_vs_best_learned": 0,
    }
    neural_decoder_rejected_count = 0
    neural_decoder_rejection_reasons = {
        "bad_hotspot": 0,
        "low_mars_score": 0,
        "low_support": 0,
        "mars_gap_vs_best": 0,
        "mars_gap_vs_best_learned": 0,
    }
    best_existing_mars_score = max((float(row["mars_score"]) for row in rows), default=0.0)
    best_existing_learned_mars_score = max(
        (float(row["mars_score"]) for row in rows if row["source_group"] == "learned"),
        default=best_existing_mars_score,
    )
    decoder_generated_rows, decoder_rejected_count, decoder_rejection_reasons = materialize_decoded_candidate_rows(
        decoded_candidates=decoded_candidates,
        source_name="fusion_decoder",
        wt_seq=wt_seq,
        features=features,
        oxidation_hotspots=oxidation_hotspots,
        flexible_positions=flexible_positions,
        profile=profile,
        asr_profile=asr_profile,
        family_positive_profile=family_positive_profile,
        family_negative_profile=family_negative_profile,
        manual_bias=manual_bias,
        design_positions=design_positions,
        score_weights=score_weights,
        position_to_index=position_to_index,
        evolution_position_weights=evolution_position_weights,
        residue_numbers=residue_numbers,
        evo_cfg=evo_cfg,
        topic_name=topic_name,
        topic_cfg=topic_cfg,
        existing_sequences=existing_sequences,
        min_mars_score=decoder_min_mars_score,
        min_support_count=decoder_min_support_count,
        max_mars_gap_vs_best=decoder_max_mars_gap_vs_best,
        max_mars_gap_vs_best_learned=decoder_max_mars_gap_vs_best_learned,
        best_existing_mars_score=best_existing_mars_score,
        best_existing_learned_mars_score=best_existing_learned_mars_score,
        skip_bad_hotspots=decoder_skip_bad_hotspots,
    )
    if neural_decoded_candidates:
        neural_decoder_generated_rows, neural_decoder_rejected_count, neural_decoder_rejection_reasons = materialize_decoded_candidate_rows(
            decoded_candidates=neural_decoded_candidates,
            source_name="neural_decoder",
            wt_seq=wt_seq,
            features=features,
            oxidation_hotspots=oxidation_hotspots,
            flexible_positions=flexible_positions,
            profile=profile,
            asr_profile=asr_profile,
            family_positive_profile=family_positive_profile,
            family_negative_profile=family_negative_profile,
            manual_bias=manual_bias,
            design_positions=design_positions,
            score_weights=score_weights,
            position_to_index=position_to_index,
            evolution_position_weights=evolution_position_weights,
            residue_numbers=residue_numbers,
            evo_cfg=evo_cfg,
            topic_name=topic_name,
            topic_cfg=topic_cfg,
            existing_sequences=existing_sequences,
            min_mars_score=neural_decoder_min_mars_score,
            min_support_count=neural_decoder_min_support_count,
            max_mars_gap_vs_best=neural_decoder_max_mars_gap_vs_best,
            max_mars_gap_vs_best_learned=neural_decoder_max_mars_gap_vs_best_learned,
            best_existing_mars_score=best_existing_mars_score,
            best_existing_learned_mars_score=best_existing_learned_mars_score,
            skip_bad_hotspots=decoder_skip_bad_hotspots,
        )

    if decoder_generated_rows or neural_decoder_generated_rows:
        augmented_rows = rows + decoder_generated_rows + neural_decoder_generated_rows
        if fusion_model is not None:
            rows = rank_rows_with_model(
                rows=augmented_rows,
                protein_name=name,
                feature_summary=summary,
                profile_summary=profile_summary,
                model_payload=fusion_model,
            )
        else:
            rows = augmented_rows
            for row in rows:
                row["ranking_score"] = float(row.get("mars_score", 0.0))
                row["ranking_model"] = "mars_score_v0"
                row["fusion_score"] = float(row.get("mars_score", 0.0))
                row.setdefault("fusion_linear_generator", 0.0)
                row.setdefault("fusion_linear_structure", 0.0)
                row.setdefault("fusion_linear_evolution", 0.0)
                row.setdefault("fusion_linear_consensus", 0.0)
                row.setdefault("fusion_linear_topic", 0.0)
                row.setdefault("fusion_linear_context", 0.0)
                row.setdefault("fusion_linear_misc", 0.0)
                row.setdefault("fusion_interaction", 0.0)
                row.setdefault("fusion_raw_feature_norm", 0.0)
            rows.sort(
                key=lambda item: (
                    -float(item["ranking_score"]),
                    -float(item.get("mars_score", 0.0)),
                    str(item.get("mutations", "")),
                    str(item.get("source", "")),
                )
            )

    fusion_summary["decoder_generated_count"] = len(decoded_candidates)
    fusion_summary["decoder_novel_count"] = len(decoder_generated_rows)
    fusion_summary["decoder_rejected_count"] = decoder_rejected_count
    fusion_summary["decoder_rejection_reasons"] = decoder_rejection_reasons
    fusion_summary["decoder_injected"] = bool(decoder_generated_rows)
    fusion_summary["neural_decoder_enabled"] = neural_decoder_enabled
    fusion_summary["neural_decoder_generated_count"] = len(neural_decoded_candidates)
    fusion_summary["neural_decoder_novel_count"] = len(neural_decoder_generated_rows)
    fusion_summary["neural_decoder_rejected_count"] = neural_decoder_rejected_count
    fusion_summary["neural_decoder_rejection_reasons"] = neural_decoder_rejection_reasons
    fusion_summary["neural_decoder_injected"] = bool(neural_decoder_generated_rows)
    fusion_summary["decoder_enabled"] = decoder_enabled
    best_decoder = next((row for row in rows if row["source"] == "fusion_decoder"), None)
    best_neural_decoder = next((row for row in rows if row["source"] == "neural_decoder"), None)
    fusion_summary["best_decoder_candidate"] = best_decoder["mutations"] if best_decoder else ""
    fusion_summary["best_decoder_ranking_score"] = float(best_decoder["ranking_score"]) if best_decoder else 0.0
    fusion_summary["best_neural_decoder_candidate"] = best_neural_decoder["mutations"] if best_neural_decoder else ""
    fusion_summary["best_neural_decoder_ranking_score"] = float(best_neural_decoder["ranking_score"]) if best_neural_decoder else 0.0

    for row in rows:
        selection_score = float(row.get("ranking_score", row.get("mars_score", 0.0)))
        engineering_score = float(row.get("mars_score", 0.0))
        row["selection_score"] = selection_score
        row["selection_score_name"] = "ranking_score" if "ranking_score" in row else "mars_score"
        row["engineering_score"] = engineering_score
        row.setdefault("neural_score", "")
        row.setdefault("neural_score_z", "")
        row.setdefault("neural_rank", "")
        row.setdefault("neural_selection_pred", "")
        row.setdefault("neural_selection_z", "")
        row.setdefault("neural_engineering_pred", "")
        row.setdefault("neural_engineering_z", "")
        row.setdefault("neural_policy_pred", "")
        row.setdefault("neural_policy_z", "")
        row.setdefault("neural_policy_score", "")

    if decoder_generated_rows:
        with (out_root / "decoder_generated_candidates.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "candidate_id",
                    "source",
                    "source_group",
                    "supporting_sources",
                    "mutations",
                    "selection_score",
                    "selection_score_name",
                    "engineering_score",
                    "neural_score",
                    "neural_score_z",
                    "neural_rank",
                    "neural_selection_pred",
                    "neural_selection_z",
                    "neural_engineering_pred",
                    "neural_engineering_z",
                    "neural_policy_pred",
                    "neural_policy_z",
                    "neural_policy_score",
                    "ranking_score",
                    "ranking_model",
                    "fusion_score",
                    "ranking_score_raw",
                    "ranking_score_z",
                    "ranking_score_bounded",
                    "ranking_score_mars_calibrated",
                    "ranking_penalty",
                    "ranking_penalty_reasons",
                    "mars_score",
                    "score_oxidation",
                    "score_surface",
                    "score_manual",
                    "score_evolution",
                    "score_burden",
                    "score_topic_sequence",
                    "score_topic_structure",
                    "score_topic_evolution",
                    "fusion_linear_generator",
                    "fusion_linear_structure",
                    "fusion_linear_evolution",
                    "fusion_linear_consensus",
                    "fusion_linear_topic",
                    "fusion_linear_context",
                    "fusion_linear_misc",
                    "fusion_interaction",
                    "fusion_raw_feature_norm",
                    "notes",
                    "sequence",
                    "header",
                ],
            )
            writer.writeheader()
            writer.writerows([row for row in rows if row["source"] == "fusion_decoder"])
    if neural_decoder_generated_rows:
        with (out_root / "neural_decoder_generated_candidates.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "candidate_id",
                    "source",
                    "source_group",
                    "supporting_sources",
                    "mutations",
                    "selection_score",
                    "selection_score_name",
                    "engineering_score",
                    "neural_score",
                    "neural_score_z",
                    "neural_rank",
                    "neural_selection_pred",
                    "neural_selection_z",
                    "neural_engineering_pred",
                    "neural_engineering_z",
                    "neural_policy_pred",
                    "neural_policy_z",
                    "neural_policy_score",
                    "ranking_score",
                    "ranking_model",
                    "fusion_score",
                    "ranking_score_raw",
                    "ranking_score_z",
                    "ranking_score_bounded",
                    "ranking_score_mars_calibrated",
                    "ranking_penalty",
                    "ranking_penalty_reasons",
                    "mars_score",
                    "score_oxidation",
                    "score_surface",
                    "score_manual",
                    "score_evolution",
                    "score_burden",
                    "score_topic_sequence",
                    "score_topic_structure",
                    "score_topic_evolution",
                    "fusion_linear_generator",
                    "fusion_linear_structure",
                    "fusion_linear_evolution",
                    "fusion_linear_consensus",
                    "fusion_linear_topic",
                    "fusion_linear_context",
                    "fusion_linear_misc",
                    "fusion_interaction",
                    "fusion_raw_feature_norm",
                    "notes",
                    "sequence",
                    "header",
                ],
            )
            writer.writeheader()
            writer.writerows([row for row in rows if row["source"] == "neural_decoder"])
    if fusion_model is not None:
        (out_root / "learned_fusion_model.json").write_text(
            json.dumps(fusion_model, indent=2),
            encoding="utf-8",
        )
    (out_root / "learned_fusion_summary.json").write_text(
        json.dumps(fusion_summary, indent=2),
        encoding="utf-8",
    )

    with (out_root / "combined_ranked_candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "candidate_id",
                "source",
                "source_group",
                "supporting_sources",
                "mutations",
                "selection_score",
                "selection_score_name",
                "engineering_score",
                "neural_score",
                "neural_score_z",
                "neural_rank",
                "neural_selection_pred",
                "neural_selection_z",
                "neural_engineering_pred",
                "neural_engineering_z",
                "neural_policy_pred",
                "neural_policy_z",
                "neural_policy_score",
                "ranking_score",
                "ranking_model",
                "fusion_score",
                "ranking_score_raw",
                "ranking_score_z",
                "ranking_score_bounded",
                "ranking_score_mars_calibrated",
                "ranking_penalty",
                "ranking_penalty_reasons",
                "mars_score",
                "score_oxidation",
                "score_surface",
                "score_manual",
                "score_evolution",
                "score_burden",
                "score_topic_sequence",
                "score_topic_structure",
                "score_topic_evolution",
                "fusion_linear_generator",
                "fusion_linear_structure",
                "fusion_linear_evolution",
                "fusion_linear_consensus",
                "fusion_linear_topic",
                "fusion_linear_context",
                "fusion_linear_misc",
                "fusion_interaction",
                "fusion_raw_feature_norm",
                "notes",
                "sequence",
                "header",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    neural_summary: dict[str, object] = {}
    neural_reranked_path = out_root / "neural_field_rerank" / "neural_reranked_candidates.csv"
    neural_summary_path = out_root / "neural_field_rerank" / "neural_rerank_summary.json"
    if neural_rerank_enabled:
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_mars_field_neural_reranker.py"),
                "--target",
                str(name),
                "--epochs",
                str(neural_rerank_epochs),
                "--lr",
                str(neural_rerank_lr),
            ],
            cwd=ROOT,
        )
        if neural_summary_path.exists():
            neural_summary = json.loads(neural_summary_path.read_text(encoding="utf-8"))
        if neural_reranked_path.exists():
            neural_df = pd.read_csv(neural_reranked_path)
            neural_by_id = {
                str(row["candidate_id"]): {
                    "neural_score": float(row.get("neural_energy", 0.0)),
                    "neural_score_z": float(row.get("neural_energy_z", 0.0)),
                    "neural_rank": int(row.get("neural_rank", 0)),
                    "neural_selection_pred": float(row.get("neural_selection_pred", 0.0)),
                    "neural_selection_z": float(row.get("neural_selection_z", 0.0)),
                    "neural_engineering_pred": float(row.get("neural_engineering_pred", 0.0)),
                    "neural_engineering_z": float(row.get("neural_engineering_z", 0.0)),
                    "neural_policy_pred": float(row.get("neural_policy_pred", 0.0)),
                    "neural_policy_z": float(row.get("neural_policy_z", 0.0)),
                    "neural_policy_score": float(row.get("neural_policy_score", 0.0)),
                }
                for _, row in neural_df.iterrows()
            }
            for row in rows:
                match = neural_by_id.get(str(row["candidate_id"]))
                if not match:
                    continue
                row["neural_score"] = match["neural_score"]
                row["neural_score_z"] = match["neural_score_z"]
                row["neural_rank"] = match["neural_rank"]
                row["neural_selection_pred"] = match["neural_selection_pred"]
                row["neural_selection_z"] = match["neural_selection_z"]
                row["neural_engineering_pred"] = match["neural_engineering_pred"]
                row["neural_engineering_z"] = match["neural_engineering_z"]
                row["neural_policy_pred"] = match["neural_policy_pred"]
                row["neural_policy_z"] = match["neural_policy_z"]
                row["neural_policy_score"] = match["neural_policy_score"]
            with (out_root / "combined_ranked_candidates.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "candidate_id",
                        "source",
                        "source_group",
                        "supporting_sources",
                        "mutations",
                        "selection_score",
                        "selection_score_name",
                        "engineering_score",
                        "neural_score",
                        "neural_score_z",
                        "neural_rank",
                        "neural_selection_pred",
                        "neural_selection_z",
                        "neural_engineering_pred",
                        "neural_engineering_z",
                        "neural_policy_pred",
                        "neural_policy_z",
                        "neural_policy_score",
                        "ranking_score",
                        "ranking_model",
                        "fusion_score",
                        "ranking_score_raw",
                        "ranking_score_z",
                        "ranking_score_bounded",
                        "ranking_score_mars_calibrated",
                        "ranking_penalty",
                        "ranking_penalty_reasons",
                        "mars_score",
                        "score_oxidation",
                        "score_surface",
                        "score_manual",
                        "score_evolution",
                        "score_burden",
                        "score_topic_sequence",
                        "score_topic_structure",
                        "score_topic_evolution",
                        "fusion_linear_generator",
                        "fusion_linear_structure",
                        "fusion_linear_evolution",
                        "fusion_linear_consensus",
                        "fusion_linear_topic",
                        "fusion_linear_context",
                        "fusion_linear_misc",
                        "fusion_interaction",
                        "fusion_raw_feature_norm",
                        "notes",
                        "sequence",
                        "header",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

    shortlist = rows[: args.top_k]
    proposal_ops.write_shortlist_fasta(shortlist, out_root / "shortlist_top.fasta")

    best_overall = rows[0] if rows else None
    learned_rows = [row for row in rows if row["source_group"] == "learned"]
    best_learned = learned_rows[0] if learned_rows else None
    best_neural = None
    if neural_rerank_enabled and neural_summary:
        best_neural = next((row for row in rows if str(row.get("mutations", "")) == str(neural_summary.get("top_neural_mutations", ""))), None)
    policy_row = best_overall
    policy_resolution = "current"
    if neural_rerank_enabled and best_neural is not None:
        if neural_selection_policy == "neural":
            policy_row = best_neural
            policy_resolution = "neural"
        elif neural_selection_policy == "hybrid":
            selection_delta = float(best_neural.get("selection_score", 0.0)) - float(best_overall.get("selection_score", 0.0))
            if (
                float(best_neural.get("engineering_score", 0.0)) >= float(best_overall.get("engineering_score", 0.0))
                and selection_delta >= -HYBRID_SELECTION_TOLERANCE
            ):
                policy_row = best_neural
                policy_resolution = "hybrid_neural"
            else:
                policy_resolution = "hybrid_current"
    neural_policy_summary = {
        "enabled": neural_rerank_enabled,
        "selection_policy": neural_selection_policy,
        "policy_resolution": policy_resolution,
        "policy_candidate_id": str(policy_row["candidate_id"]) if policy_row else "",
        "policy_source": str(policy_row["source"]) if policy_row else "",
        "policy_mutations": str(policy_row["mutations"]) if policy_row else "",
        "policy_selection_score": float(policy_row.get("selection_score", 0.0)) if policy_row else 0.0,
        "policy_engineering_score": float(policy_row.get("engineering_score", 0.0)) if policy_row else 0.0,
        "neural_summary_path": str(neural_summary_path) if neural_rerank_enabled else "",
    }
    (out_root / "neural_policy_summary.json").write_text(
        json.dumps(neural_policy_summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"# {name} Mars pipeline summary",
        "",
        f"- oxidation hotspots: {oxidation_hotspots}",
        f"- flexible positions: {flexible_positions}",
        f"- design positions: {design_positions}",
        f"- MSA prior loaded: {'yes' if profile is not None else 'no'}",
        f"- aligned homolog count: {profile_summary['accepted_homologs']}",
        f"- ASR prior loaded: {'yes' if profile_summary['asr_prior_enabled'] else 'no'}",
        f"- ASR accepted sequences: {profile_summary['accepted_asr']}",
        f"- family prior loaded: {'yes' if profile_summary['family_prior_enabled'] else 'no'}",
        f"- family prior dataset: {profile_summary['family_dataset_id'] or 'NA'}",
        f"- family prior accepted positive/negative: {profile_summary['accepted_positive']}/{profile_summary['accepted_negative']}",
        f"- topic scoring enabled: {'yes' if topic_enabled else 'no'}",
        f"- topic scorer: {topic_name or 'NA'}",
        f"- template-aware weighting: {'yes' if profile_summary['template_weighting_enabled'] else 'no'}",
        f"- template context reference: {profile_summary['template_context_reference'] or 'NA'}",
        f"- evolution position weights: {profile_summary['evolution_position_weights'] or {}}",
        f"- ranking model: {fusion_summary.get('ranking_model', 'mars_score_v0')}",
        f"- selection score contract: selection_score={rows[0].get('selection_score_name', 'mars_score') if rows else 'NA'}, engineering_score=mars_score",
        f"- neural rerank enabled: {'yes' if neural_rerank_enabled else 'no'}",
        f"- neural selection policy: {neural_selection_policy}",
        f"- learned fusion reason: {fusion_summary.get('reason', 'trained')}",
        f"- learned fusion training targets: {fusion_summary.get('training_target_count', 0)}",
        f"- learned fusion training examples: {fusion_summary.get('training_example_count', 0)}",
        f"- retrieval memory hits: {sum(len(v) for v in field_build.bundle.retrieval.recommendations.values())}",
        f"- retrieval motif prototypes: {len(field_build.bundle.retrieval.atlas)}",
        f"- ancestral field active: {'yes' if field_build.bundle.ancestral.ancestral_field else 'no'}",
        f"- pairwise design edges: {len(pairwise_tensor)}",
        f"- decoder preview candidates: {len(decoded_candidates)}",
        f"- decoder novel candidates injected: {fusion_summary.get('decoder_novel_count', 0)}",
        f"- decoder candidates rejected by safety filter: {fusion_summary.get('decoder_rejected_count', 0)}",
        f"- best decoder candidate: {fusion_summary.get('best_decoder_candidate', 'NA') or 'NA'}",
        f"- neural field decoder enabled: {'yes' if neural_decoder_enabled else 'no'}",
        f"- neural field decoder preview candidates: {len(neural_decoded_candidates)}",
        f"- neural field decoder novel candidates injected: {fusion_summary.get('neural_decoder_novel_count', 0)}",
        f"- neural field decoder candidates rejected by safety filter: {fusion_summary.get('neural_decoder_rejected_count', 0)}",
        f"- best neural field decoder candidate: {fusion_summary.get('best_neural_decoder_candidate', 'NA') or 'NA'}",
        f"- neural top candidate: {neural_summary.get('top_neural_mutations', 'NA') or 'NA'}" if neural_summary else "- neural top candidate: NA",
        f"- neural top source: {neural_summary.get('top_neural_source', 'NA') or 'NA'}" if neural_summary else "- neural top source: NA",
        f"- neural top energy: {neural_summary.get('top_neural_energy', 'NA')}" if neural_summary else "- neural top energy: NA",
        f"- neural top selection prediction: {neural_summary.get('top_neural_selection_pred', 'NA')}" if neural_summary else "- neural top selection prediction: NA",
        f"- neural top engineering prediction: {neural_summary.get('top_neural_engineering_pred', 'NA')}" if neural_summary else "- neural top engineering prediction: NA",
        f"- neural top policy prediction: {neural_summary.get('top_neural_policy_pred', 'NA')}" if neural_summary else "- neural top policy prediction: NA",
        f"- neural top policy z: {neural_summary.get('top_neural_policy_z', 'NA')}" if neural_summary else "- neural top policy z: NA",
        f"- neural top policy score: {neural_summary.get('top_neural_policy_score', 'NA')}" if neural_summary else "- neural top policy score: NA",
        f"- policy winner: `{policy_row['mutations']}` from {policy_row['source']} ({policy_resolution})" if policy_row else "- policy winner: none",
        f"- protein_mpnn enabled: {'yes' if mpnn_enabled else 'no'}",
        f"- template mismatch positions restored to WT for scoring: {template_mismatch_positions}",
        f"- best overall winner: `{best_overall['mutations']}` from {best_overall['source']} (ranking_score={best_overall['ranking_score']})" if best_overall else "- best overall winner: none",
        f"- best learned winner: `{best_learned['mutations']}` from {best_learned['source']} (ranking_score={best_learned['ranking_score']})" if best_learned else "- best learned winner: none",
        "",
        "## Top candidates",
        "",
    ]
    for row in shortlist[:8]:
        lines.append(
            f"- {row['candidate_id']}: `{row['mutations']}` selection_score={row['selection_score']} engineering_score={row['engineering_score']}"
        )
    lines.extend(["", "## Decoder Preview", ""])
    for item in decoded_candidates[:6]:
        lines.append(f"- `{';'.join(item.mutations) or 'WT'}` decoder_score={item.decoder_score}")
    (out_root / "pipeline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote ranked candidates to {out_root / 'combined_ranked_candidates.csv'}")
    print(f"Wrote shortlist FASTA to {out_root / 'shortlist_top.fasta'}")


if __name__ == "__main__":
    main()
