from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from .structure_features import ResidueFeature


def load_fasta(path: Path | None) -> list[tuple[str, str]]:
    if not path or not path.exists():
        return []
    entries: list[tuple[str, str]] = []
    header: str | None = None
    seq: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(seq)))
            header = line[1:]
            seq = []
        else:
            seq.append(line)
    if header is not None:
        entries.append((header, "".join(seq)))
    return entries


def load_aligned_fasta(path: Path | None) -> list[str]:
    return [seq for _, seq in load_fasta(path)]


def load_yaml(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_fasta(entries: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for header, seq in entries:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")


def _needleman_wunsch(ref: str, query: str, match_score: int = 2, mismatch_score: int = -1, gap_score: int = -2) -> tuple[str, str]:
    n = len(ref)
    m = len(query)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        score[i][0] = i * gap_score
        trace[i][0] = "U"
    for j in range(1, m + 1):
        score[0][j] = j * gap_score
        trace[0][j] = "L"

    for i in range(1, n + 1):
        ref_aa = ref[i - 1]
        for j in range(1, m + 1):
            query_aa = query[j - 1]
            diag = score[i - 1][j - 1] + (match_score if ref_aa == query_aa else mismatch_score)
            up = score[i - 1][j] + gap_score
            left = score[i][j - 1] + gap_score
            best = max(diag, up, left)
            score[i][j] = best
            trace[i][j] = "D" if best == diag else ("U" if best == up else "L")

    aligned_ref: list[str] = []
    aligned_query: list[str] = []
    i = n
    j = m
    while i > 0 or j > 0:
        move = trace[i][j] if i >= 0 and j >= 0 else ""
        if i > 0 and j > 0 and move == "D":
            aligned_ref.append(ref[i - 1])
            aligned_query.append(query[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or move == "U"):
            aligned_ref.append(ref[i - 1])
            aligned_query.append("-")
            i -= 1
        else:
            aligned_ref.append("-")
            aligned_query.append(query[j - 1])
            j -= 1

    return "".join(reversed(aligned_ref)), "".join(reversed(aligned_query))


def align_to_reference(ref: str, query: str) -> str:
    aligned_ref, aligned_query = _needleman_wunsch(ref, query)
    projected: list[str] = []
    for ref_aa, query_aa in zip(aligned_ref, aligned_query):
        if ref_aa == "-":
            continue
        projected.append(query_aa if query_aa != "-" else "-")
    return "".join(projected)


def anchored_align_entries(
    wt_seq: str,
    entries: list[tuple[str, str]],
    min_identity: float = 0.20,
) -> list[tuple[str, str]]:
    aligned: list[tuple[str, str]] = [("wt_reference", wt_seq)]
    seen: set[str] = {wt_seq}
    for header, seq in entries:
        clean = seq.strip().upper()
        if not clean or clean in seen:
            continue
        projected = align_to_reference(wt_seq, clean)
        matches = sum(1 for a, b in zip(wt_seq, projected) if b != "-" and a == b)
        covered = sum(1 for aa in projected if aa != "-")
        denom = max(1, min(len(wt_seq), covered))
        identity = matches / denom
        if identity < min_identity:
            continue
        aligned.append((header, projected))
        seen.add(clean)
    return aligned


def build_profile(seqs: list[str], wt_seq: str) -> list[dict[str, float]] | None:
    if not seqs:
        return None
    clean = [s for s in seqs if len(s) == len(wt_seq)]
    if not clean:
        return None
    profile = []
    for i in range(len(wt_seq)):
        counts: dict[str, int] = {}
        total = 0
        for seq in clean:
            aa = seq[i]
            if aa == "-":
                continue
            counts[aa] = counts.get(aa, 0) + 1
            total += 1
        profile.append({aa: c / total for aa, c in counts.items()} if total else {})
    return profile


def build_profile_from_homologs(
    wt_seq: str,
    homolog_entries: list[tuple[str, str]],
    aligned_out: Path | None = None,
    min_identity: float = 0.20,
) -> tuple[list[tuple[str, str]], list[dict[str, float]] | None]:
    aligned_entries = anchored_align_entries(wt_seq, homolog_entries, min_identity=min_identity)
    if aligned_out is not None:
        write_fasta(aligned_entries, aligned_out)
    profile = build_profile([seq for _, seq in aligned_entries], wt_seq)
    return aligned_entries, profile


def build_family_pair_profiles(
    wt_seq: str,
    positive_entries: list[tuple[str, str]],
    negative_entries: list[tuple[str, str]],
    positive_aligned_out: Path | None = None,
    negative_aligned_out: Path | None = None,
    min_identity: float = 0.20,
) -> tuple[dict[str, Any], list[dict[str, float]] | None, list[dict[str, float]] | None]:
    positive_aligned = anchored_align_entries(wt_seq, positive_entries, min_identity=min_identity)
    negative_aligned = anchored_align_entries(wt_seq, negative_entries, min_identity=min_identity)
    if positive_aligned_out is not None:
        write_fasta(positive_aligned, positive_aligned_out)
    if negative_aligned_out is not None:
        write_fasta(negative_aligned, negative_aligned_out)

    positive_profile = build_profile([seq for _, seq in positive_aligned], wt_seq)
    negative_profile = build_profile([seq for _, seq in negative_aligned], wt_seq)

    summary = {
        "input_positive": len(positive_entries),
        "input_negative": len(negative_entries),
        "accepted_positive": max(0, len(positive_aligned) - 1),
        "accepted_negative": max(0, len(negative_aligned) - 1),
        "mean_positive_coverage": 0.0,
        "mean_negative_coverage": 0.0,
    }
    for key, aligned in [("positive", positive_aligned), ("negative", negative_aligned)]:
        accepted = max(0, len(aligned) - 1)
        if accepted <= 0:
            continue
        coverage = [sum(1 for _, seq in aligned[1:] if seq[i] != "-") for i in range(len(wt_seq))]
        summary[f"mean_{key}_coverage"] = round(sum(coverage) / max(1, len(wt_seq) * accepted), 3)

    return summary, positive_profile, negative_profile


def profile_log_score(
    seq: str,
    profile: list[dict[str, float]] | None,
    positions: list[int],
    position_to_index: dict[int, int] | None = None,
    position_weights: dict[int, float] | None = None,
) -> float:
    if profile is None:
        return 0.0
    terms = []
    weights = []
    for pos in positions:
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        aa = seq[idx]
        p = profile[idx].get(aa, 1e-6)
        terms.append(math.log(p + 1e-6))
        weights.append(float(position_weights.get(pos, 1.0)) if position_weights is not None else 1.0)
    if not terms:
        return 0.0
    denom = sum(weights)
    if denom <= 0:
        return 0.0
    return sum(term * weight for term, weight in zip(terms, weights)) / denom


def differential_profile_score(
    seq: str,
    positive_profile: list[dict[str, float]] | None,
    negative_profile: list[dict[str, float]] | None,
    positions: list[int],
    position_to_index: dict[int, int] | None = None,
    position_weights: dict[int, float] | None = None,
    eps: float = 1e-6,
) -> float:
    if positive_profile is None or negative_profile is None:
        return 0.0
    terms = []
    weights = []
    for pos in positions:
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        aa = seq[idx]
        p_pos = positive_profile[idx].get(aa, eps)
        p_neg = negative_profile[idx].get(aa, eps)
        terms.append(math.log(p_pos + eps) - math.log(p_neg + eps))
        weights.append(float(position_weights.get(pos, 1.0)) if position_weights is not None else 1.0)
    if not terms:
        return 0.0
    denom = sum(weights)
    if denom <= 0:
        return 0.0
    return sum(term * weight for term, weight in zip(terms, weights)) / denom


def differential_family_recommendations(
    positive_profile: list[dict[str, float]] | None,
    negative_profile: list[dict[str, float]] | None,
    positions: list[int],
    position_to_index: dict[int, int] | None = None,
    top_k: int = 3,
    min_delta: float = 0.05,
) -> dict[int, dict[str, float]]:
    if positive_profile is None or negative_profile is None:
        return {}

    recommendations: dict[int, dict[str, float]] = {}
    for pos in positions:
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        aa_scores: dict[str, float] = {}
        all_aas = set(positive_profile[idx]) | set(negative_profile[idx])
        for aa in all_aas:
            if aa == "-":
                continue
            delta = float(positive_profile[idx].get(aa, 0.0) - negative_profile[idx].get(aa, 0.0))
            if delta >= min_delta:
                aa_scores[aa] = round(delta, 4)
        if aa_scores:
            ranked = sorted(aa_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            recommendations[pos] = {aa: score for aa, score in ranked}
    return recommendations


def profile_recommendations(
    profile: list[dict[str, float]] | None,
    wt_seq: str,
    positions: list[int],
    position_to_index: dict[int, int] | None = None,
    top_k: int = 2,
    min_prob: float = 0.20,
) -> dict[int, dict[str, float]]:
    if profile is None:
        return {}

    recommendations: dict[int, dict[str, float]] = {}
    for pos in positions:
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        wt = wt_seq[idx]
        aa_scores = {
            aa: round(float(prob), 4)
            for aa, prob in profile[idx].items()
            if aa not in {"-", wt} and float(prob) >= min_prob
        }
        if aa_scores:
            ranked = sorted(aa_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            recommendations[pos] = {aa: score for aa, score in ranked}
    return recommendations


def build_structure_position_weights(
    features: list[ResidueFeature],
    positions: list[int],
    oxidation_hotspots: list[int] | None = None,
    flexible_positions: list[int] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[int, float]:
    settings = {
        "base_weight": 1.0,
        "oxidation_bonus": 0.8,
        "flexible_bonus": 0.4,
        "surface_bonus_scale": 0.012,
        "surface_bonus_cap": 0.8,
        "surface_reference_sasa": 20.0,
        "buried_penalty_threshold": 10.0,
        "buried_penalty_factor": 0.75,
        "disulfide_penalty_factor": 0.85,
        "glyco_penalty_factor": 0.9,
        "max_weight": 3.0,
        "min_weight": 0.2,
    }
    if cfg:
        settings.update(cfg)

    feat_map = {feat.num: feat for feat in features}
    oxidation_set = set(oxidation_hotspots or [])
    flexible_set = set(flexible_positions or [])
    weights: dict[int, float] = {}

    for pos in positions:
        feat = feat_map.get(pos)
        if feat is None:
            continue

        weight = float(settings["base_weight"])
        if pos in oxidation_set:
            weight += float(settings["oxidation_bonus"])
        if pos in flexible_set:
            weight += float(settings["flexible_bonus"])

        excess_sasa = max(0.0, float(feat.sasa) - float(settings["surface_reference_sasa"]))
        weight += min(
            float(settings["surface_bonus_cap"]),
            excess_sasa * float(settings["surface_bonus_scale"]),
        )

        if float(feat.sasa) < float(settings["buried_penalty_threshold"]):
            weight *= float(settings["buried_penalty_factor"])
        if feat.in_disulfide:
            weight *= float(settings["disulfide_penalty_factor"])
        if feat.glyco_motif:
            weight *= float(settings["glyco_penalty_factor"])

        weight = min(float(settings["max_weight"]), max(float(settings["min_weight"]), weight))
        weights[pos] = round(weight, 3)

    return weights
