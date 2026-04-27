from __future__ import annotations

import math
from typing import Any

# Standard amino acids
STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_BACKGROUND_FREQ = {
    "A": 0.082, "C": 0.019, "D": 0.054, "E": 0.062, "F": 0.039,
    "G": 0.072, "H": 0.022, "I": 0.051, "K": 0.058, "L": 0.092,
    "M": 0.024, "N": 0.042, "P": 0.052, "Q": 0.040, "R": 0.054,
    "S": 0.068, "T": 0.056, "V": 0.067, "W": 0.013, "Y": 0.032,
}


def shannon_entropy(profile: list[dict[str, float]]) -> list[float]:
    """Calculate Shannon entropy at each position in the alignment profile.

    Args:
        profile: List of amino acid frequency dictionaries per position

    Returns:
        List of entropy values (0 = perfectly conserved, higher = more variable)
    """
    entropies = []
    for pos_dict in profile:
        if not pos_dict:
            entropies.append(0.0)
            continue
        total = sum(pos_dict.values())
        if total <= 0:
            entropies.append(0.0)
            continue
        H = 0.0
        for p in pos_dict.values():
            if p > 0:
                norm_p = p / total
                H -= norm_p * math.log(max(norm_p, 1e-10))
        entropies.append(round(H, 6))
    return entropies


def normalized_conservation(profile: list[dict[str, float]]) -> list[float]:
    """Calculate normalized conservation score (0-1 scale).

    Args:
        profile: List of amino acid frequency dictionaries per position

    Returns:
        List of conservation scores (1 = perfectly conserved, 0 = maximally variable)
    """
    max_entropy = math.log(20)  # Maximum possible entropy for 20 amino acids
    entropies = shannon_entropy(profile)
    return [round(1.0 - (H / max_entropy if max_entropy > 0 else 0.0), 6) for H in entropies]


def build_pssm(
    profile: list[dict[str, float]],
    background_freqs: dict[str, float] | None = None,
    pseudocount: float = 0.01,
) -> list[dict[str, float]]:
    """Generate Position-Specific Scoring Matrix from alignment profile.

    Args:
        profile: List of amino acid frequency dictionaries per position
        background_freqs: Background amino acid frequencies (default: standard frequencies)
        pseudocount: Pseudocount to add to prevent log(0) issues

    Returns:
        List of PSSM dictionaries with log-odds scores per position and amino acid
    """
    bg = background_freqs or AA_BACKGROUND_FREQ
    pssm = []
    for pos_dict in profile:
        scores = {}
        total = sum(pos_dict.values()) if pos_dict else 0.0
        if total <= 0:
            for aa in STANDARD_AA:
                scores[aa] = round(math.log(max(pseudocount, bg.get(aa, 0.01)) / bg.get(aa, 0.01)), 4)
        else:
            for aa in STANDARD_AA:
                obs_freq = (pos_dict.get(aa, 0.0) / total) if total > 0 else 0.0
                obs_with_pc = obs_freq + pseudocount
                bg_freq = bg.get(aa, 0.01)
                log_odds = math.log(obs_with_pc / bg_freq)
                scores[aa] = round(log_odds, 4)
        pssm.append(scores)
    return pssm


def conservation_weight(
    profile: list[dict[str, float]],
    weights: dict[int, float] | None = None,
) -> dict[int, float]:
    """Calculate position-specific conservation weights.

    Args:
        profile: List of amino acid frequency dictionaries per position
        weights: Optional existing weights to multiply with conservation

    Returns:
        Dictionary mapping position index to conservation weight
    """
    conservation = normalized_conservation(profile)
    result = {}
    for i, score in enumerate(conservation):
        base_weight = 1.0 + score * 2.0  # Scale conservation to weight
        if weights and i in weights:
            base_weight *= weights[i]
        result[i] = round(base_weight, 4)
    return result


def score_mutant_pssm(
    pssm: list[dict[str, float]],
    mutant_seq: str,
    wt_seq: str,
    position_to_index: dict[int, int] | None = None,
) -> float:
    """Score a mutant sequence using PSSM.

    Args:
        pssm: Position-Specific Scoring Matrix
        mutant_seq: Mutant sequence string
        wt_seq: Wild-type sequence string
        position_to_index: Optional mapping from residue numbers to indices

    Returns:
        Sum of PSSM scores for the mutant
    """
    if len(mutant_seq) != len(wt_seq):
        raise ValueError("mutant_seq and wt_seq must have the same length")

    total_score = 0.0
    for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, mutant_seq)):
        if wt_aa == mut_aa:
            continue
        idx = position_to_index[i] if position_to_index else i
        if idx >= len(pssm):
            continue
        score = pssm[idx].get(mut_aa, 0.0)
        total_score += score
    return round(total_score, 4)


def identify_functional_sites(
    profile: list[dict[str, float]],
    wt_seq: str,
    conservation_threshold: float = 0.7,
    entropy_threshold: float = 0.3,
    positions: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Identify functionally important sites based on conservation patterns.

    Args:
        profile: List of amino acid frequency dictionaries per position
        wt_seq: Wild-type sequence string
        conservation_threshold: Minimum conservation to be considered conserved
        entropy_threshold: Maximum entropy for highly variable sites
        positions: Optional list of specific positions to analyze

    Returns:
        List of dictionaries describing each functional site
    """
    conservation = normalized_conservation(profile)
    entropies = shannon_entropy(profile)

    sites = []
    for i in range(len(profile)):
        if positions is not None and i not in positions:
            continue
        cons = conservation[i]
        H = entropies[i]
        wt_aa = wt_seq[i] if i < len(wt_seq) else ""

        pos_dict = profile[i]
        if pos_dict:
            top_aa = max(pos_dict.items(), key=lambda x: x[1])
            top_aa_freq = top_aa[1] / sum(pos_dict.values())
        else:
            top_aa = (wt_aa, 0.0)
            top_aa_freq = 0.0

        site_type = "variable"
        if cons >= conservation_threshold:
            site_type = "conserved"
        elif H <= entropy_threshold:
            site_type = "conserved"
        elif top_aa_freq >= 0.5:
            site_type = "moderately_conserved"

        sites.append({
            "position": i,
            "wt_aa": wt_aa,
            "conservation": round(cons, 4),
            "entropy": round(H, 4),
            "top_aa": top_aa[0],
            "top_aa_freq": round(top_aa_freq, 4),
            "site_type": site_type,
        })
    return sites


def evolutionary_constraint_score(
    profile: list[dict[str, float]],
    wt_seq: str,
    positions: list[int] | None = None,
    position_to_index: dict[int, int] | None = None,
) -> dict[int, float]:
    """Calculate evolutionary constraint scores for thermostability prediction.

    Sites under strong evolutionary constraint tend to be structurally important
    and mutations there often destabilize the protein.

    Args:
        profile: List of amino acid frequency dictionaries per position
        wt_seq: Wild-type sequence string
        positions: Optional list of positions to analyze
        position_to_index: Optional mapping from residue numbers to indices

    Returns:
        Dictionary mapping position to evolutionary constraint score (0-1)
    """
    if positions is None:
        positions = list(range(len(profile)))

    entropies = shannon_entropy(profile)
    conservation = normalized_conservation(profile)

    constraint_scores = {}
    for pos in positions:
        idx = position_to_index[pos] if position_to_index else pos
        if idx >= len(profile):
            constraint_scores[pos] = 0.0
            continue

        pos_dict = profile[idx]
        H = entropies[idx]
        cons = conservation[idx]

        if not pos_dict or sum(pos_dict.values()) == 0:
            constraint_scores[pos] = 0.5
            continue

        wt_aa = wt_seq[idx] if idx < len(wt_seq) else ""
        wt_freq = pos_dict.get(wt_aa, 0.0) / sum(pos_dict.values())

        # Weight wild-type frequency highly (conservation of functional residues)
        # Low entropy + high WT frequency = high constraint
        wt_freq_weight = 0.4
        entropy_weight = 0.3
        conservation_weight = 0.3

        # Calculate constraint score
        wt_component = wt_freq * wt_freq_weight
        entropy_component = (1.0 - min(H / math.log(20), 1.0)) * entropy_weight
        cons_component = cons * conservation_weight

        constraint = wt_component + entropy_component + cons_component
        constraint_scores[pos] = round(min(1.0, max(0.0, constraint)), 4)

    return constraint_scores


def integrate_thermostability_scores(
    conservation_scores: dict[int, float],
    constraint_scores: dict[int, float],
    structure_weights: dict[int, float] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[int, float]:
    """Integrate conservation and constraint scores for thermostability prediction.

    Args:
        conservation_scores: Position-specific conservation scores
        constraint_scores: Evolutionary constraint scores
        structure_weights: Optional structural importance weights
        weights: Optional weights for each component (conservation, constraint, structure)

    Returns:
        Combined thermostability scores per position
    """
    cfg = {
        "conservation_weight": 0.3,
        "constraint_weight": 0.4,
        "structure_weight": 0.3,
    }
    if weights:
        cfg.update(weights)

    combined = {}
    all_positions = set(conservation_scores.keys()) | set(constraint_scores.keys())
    if structure_weights:
        all_positions |= set(structure_weights.keys())

    for pos in all_positions:
        cons = conservation_scores.get(pos, 0.5)
        constr = constraint_scores.get(pos, 0.5)
        struct = structure_weights.get(pos, 1.0) if structure_weights else 1.0

        # Normalize structure weight to 0-1 range
        struct_norm = min(1.0, max(0.0, struct / 3.0))  # Assuming max weight is 3.0

        score = (
            cons * cfg["conservation_weight"]
            + constr * cfg["constraint_weight"]
            + struct_norm * cfg["structure_weight"]
        )
        combined[pos] = round(min(1.0, max(0.0, score)), 4)

    return combined


def rank_positions_by_importance(
    profiles: list[list[dict[str, float]]] | None,
    wt_seq: str,
    positions: list[int] | None = None,
    position_to_index: dict[int, int] | None = None,
    structure_features: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Rank positions by evolutionary and structural importance.

    Args:
        profiles: List of multiple alignment profiles (for consensus)
        wt_seq: Wild-type sequence string
        positions: Optional list of positions to analyze
        position_to_index: Optional mapping from residue numbers to indices
        structure_features: Optional list of ResidueFeature objects

    Returns:
        Sorted list of positions with importance scores and reasons
    """
    if positions is None:
        positions = list(range(len(wt_seq)))

    # Build consensus profile if multiple profiles provided
    if profiles and len(profiles) > 0:
        consensus_profile = []
        n_profiles = len(profiles)
        seq_len = len(wt_seq)
        for i in range(seq_len):
            combined: dict[str, float] = {}
            for prof in profiles:
                if i < len(prof):
                    for aa, freq in prof[i].items():
                        combined[aa] = combined.get(aa, 0.0) + freq / n_profiles
            consensus_profile.append(combined)
    else:
        consensus_profile = profiles[0] if profiles else [{} for _ in wt_seq]

    # Calculate individual scores
    constraint_scores = evolutionary_constraint_score(consensus_profile, wt_seq, positions, position_to_index)
    conservation = normalized_conservation(conservation_profile if (profiles and len(profiles) > 0) else consensus_profile)
    entropies = shannon_entropy(consensus_profile)
    sites = identify_functional_sites(consensus_profile, wt_seq)

    # Build feature map for structure features
    feat_map = {}
    if structure_features:
        for feat in structure_features:
            feat_map[feat.num] = feat

    # Rank positions
    ranked = []
    for pos in positions:
        idx = position_to_index[pos] if position_to_index else pos
        constraint = constraint_scores.get(pos, 0.5)
        cons = conservation[idx] if idx < len(conservation) else 0.5
        entropy = entropies[idx] if idx < len(entropies) else 0.0
        site = next((s for s in sites if s["position"] == idx), {})

        # Structural contributions
        feat = feat_map.get(pos)
        struct_score = 0.5
        struct_factors = []
        if feat:
            if feat.in_disulfide:
                struct_score = max(struct_score, 0.8)
                struct_factors.append("disulfide")
            if feat.glyco_motif:
                struct_score = max(struct_score, 0.7)
                struct_factors.append("glycosylation")
            if feat.sasa < 10:
                struct_score = max(struct_score, 0.7)
                struct_factors.append("buried")
            if feat.mean_b > 50:
                struct_score = max(struct_score, 0.6)
                struct_factors.append("flexible")

        # Combined importance score
        importance = 0.35 * constraint + 0.35 * cons + 0.3 * struct_score

        reasons = []
        if site.get("site_type") == "conserved":
            reasons.append("highly_conserved")
        if constraint > 0.7:
            reasons.append("high_constraint")
        if feat and feat.in_disulfide:
            reasons.append("disulfide_bridge")
        if feat and feat.glyco_motif:
            reasons.append("glycosylation_site")
        if entropy < 0.2:
            reasons.append("low_variation")

        ranked.append({
            "position": pos,
            "importance_score": round(importance, 4),
            "constraint_score": constraint,
            "conservation_score": cons,
            "structure_score": round(struct_score, 4),
            "reasons": reasons,
            "struct_factors": struct_factors,
        })

    ranked.sort(key=lambda x: -x["importance_score"])
    return ranked


# Alias for backwards compatibility
def conservation_profile(profile: list[dict[str, float]]) -> list[dict[str, float]]:
    """Alias for build_pssm for backwards compatibility."""
    return build_pssm(profile)
