from __future__ import annotations

from dataclasses import dataclass

from .evolution import differential_profile_score, profile_log_score
from .structure_features import ResidueFeature
from .topic_score import score_topic_candidate


SAFE_OXIDATION_MAP = {
    "M": {"L": 3.0, "I": 2.6, "V": 1.8},
    "C": {"S": 2.0, "A": 1.4},
    "W": {"F": 1.6, "Y": 0.8},
    "Y": {"F": 1.2},
    "H": {"Q": 1.1, "N": 0.8},
}


@dataclass
class ScoreResult:
    total: float
    notes: list[str]
    components: dict[str, float]


def mutation_list(wt_seq: str, seq: str, residue_numbers: list[int] | None = None) -> list[str]:
    return mutation_list_with_numbering(wt_seq, seq, residue_numbers=residue_numbers)


def mutation_list_with_numbering(wt_seq: str, seq: str, residue_numbers: list[int] | None = None) -> list[str]:
    muts = []
    labels = residue_numbers if residue_numbers is not None else list(range(1, len(wt_seq) + 1))
    for i, (wt, aa) in enumerate(zip(wt_seq, seq), start=1):
        if wt != aa:
            muts.append(f"{wt}{labels[i-1]}{aa}")
    return muts


def score_candidate(
    wt_seq: str,
    seq: str,
    features: list[ResidueFeature],
    oxidation_hotspots: list[int],
    flexible_positions: list[int],
    profile: list[dict[str, float]] | None,
    asr_profile: list[dict[str, float]] | None,
    family_positive_profile: list[dict[str, float]] | None,
    family_negative_profile: list[dict[str, float]] | None,
    manual_preferred: dict[int, dict[str, float]],
    evolution_positions: list[int] | None = None,
    mutable_positions: list[int] | None = None,
    term_weights: dict[str, float] | None = None,
    position_to_index: dict[int, int] | None = None,
    evolution_position_weights: dict[int, float] | None = None,
    residue_numbers: list[int] | None = None,
    profile_prior_scale: float = 0.35,
    asr_prior_scale: float = 0.45,
    family_prior_scale: float = 0.60,
    topic_name: str | None = None,
    topic_cfg: dict[str, object] | None = None,
) -> ScoreResult:
    notes: list[str] = []
    weights = {
        "oxidation": 1.0,
        "surface": 1.0,
        "manual": 1.0,
        "evolution": 1.0,
        "burden": 1.0,
        "topic_sequence": 1.0,
        "topic_structure": 1.0,
        "topic_evolution": 1.0,
    }
    if term_weights:
        weights.update({k: float(v) for k, v in term_weights.items()})

    components = {
        "oxidation": 0.0,
        "surface": 0.0,
        "manual": 0.0,
        "evolution": 0.0,
        "burden": 0.0,
        "topic_sequence": 0.0,
        "topic_structure": 0.0,
        "topic_evolution": 0.0,
    }
    feat_map = {f.num: f for f in features}
    mutable_set = set(mutable_positions) if mutable_positions is not None else None

    for pos in oxidation_hotspots:
        if mutable_set is not None and pos not in mutable_set:
            continue
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        wt = wt_seq[idx]
        aa = seq[idx]
        if aa == wt:
            components["oxidation"] -= 1.5
            notes.append(f"keeps_hotspot_{pos}")
        elif aa in SAFE_OXIDATION_MAP.get(wt, {}):
            components["oxidation"] += SAFE_OXIDATION_MAP[wt][aa]
            notes.append(f"hardens_hotspot_{pos}")
        elif aa in {"R", "K", "H", "C", "W", "Y", "M"}:
            components["oxidation"] -= 1.4
            notes.append(f"bad_hotspot_choice_{pos}")
        else:
            components["oxidation"] += 0.2

    for pos in flexible_positions:
        if mutable_set is not None and pos not in mutable_set:
            continue
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        wt = wt_seq[idx]
        aa = seq[idx]
        feat = feat_map[pos]
        if aa == wt:
            continue
        if aa in {"E", "D", "Q", "N", "S", "T"} and feat.sasa >= 40:
            components["surface"] += 0.7
            notes.append(f"surface_hydration_{pos}")
        if aa in {"W", "F", "Y", "C", "M"} and feat.sasa >= 40:
            components["surface"] -= 0.8
            notes.append(f"sticky_surface_{pos}")

    for pos, aa_bias in manual_preferred.items():
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        aa = seq[idx]
        if aa in aa_bias:
            components["manual"] += aa_bias[aa]
            notes.append(f"manual_bias_{pos}")

    evo_positions = evolution_positions or sorted(manual_preferred)
    evo = profile_log_score(
        seq,
        profile,
        evo_positions,
        position_to_index=position_to_index,
        position_weights=evolution_position_weights,
    )
    if profile is not None:
        components["evolution"] += profile_prior_scale * evo
        notes.append("evolution_prior")
    asr_evo = profile_log_score(
        seq,
        asr_profile,
        evo_positions,
        position_to_index=position_to_index,
        position_weights=evolution_position_weights,
    )
    if asr_profile is not None:
        components["evolution"] += asr_prior_scale * asr_evo
        notes.append("asr_prior")
    family_evo = differential_profile_score(
        seq,
        family_positive_profile,
        family_negative_profile,
        evo_positions,
        position_to_index=position_to_index,
        position_weights=evolution_position_weights,
    )
    if family_positive_profile is not None and family_negative_profile is not None:
        components["evolution"] += family_prior_scale * family_evo
        notes.append("family_evolution_prior")
    if evolution_position_weights:
        notes.append("template_weighted_evolution")

    muts = mutation_list_with_numbering(wt_seq, seq, residue_numbers=residue_numbers)
    components["burden"] -= 0.15 * max(0, len(muts) - 1)
    if len(muts) <= 2:
        notes.append("low_burden")

    topic_result = score_topic_candidate(
        topic_name=topic_name,
        wt_seq=wt_seq,
        seq=seq,
        features=features,
        mutable_positions=mutable_positions,
        position_to_index=position_to_index,
        profile=profile,
        asr_profile=asr_profile,
        family_positive_profile=family_positive_profile,
        family_negative_profile=family_negative_profile,
        topic_cfg=topic_cfg,
    )
    for key, value in topic_result.components.items():
        components[key] += value
    notes.extend(topic_result.notes)

    total = round(sum(components[k] * weights.get(k, 1.0) for k in components), 3)
    weighted_components = {k: round(components[k] * weights.get(k, 1.0), 3) for k in components}
    return ScoreResult(total, notes, weighted_components)
