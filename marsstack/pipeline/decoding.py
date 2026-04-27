from __future__ import annotations

from typing import Any

from ..mars_score import mutation_list, score_candidate


def materialize_decoded_candidate_rows(
    *,
    decoded_candidates: list[Any],
    source_name: str,
    wt_seq: str,
    features: list[Any],
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
    """Score decoded candidates and emit pipeline-ready rows.

    Filters: WT-identity duplicates, sequences already present in
    ``existing_sequences``, "bad hotspot" notes (when ``skip_bad_hotspots``),
    insufficient MARS score, low support count, and MARS gap vs the best
    existing / best learned candidate.

    Returns ``(generated_rows, rejected_count, rejection_reasons)``.
    """
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
