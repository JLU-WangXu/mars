from __future__ import annotations

import math
from typing import Any


def build_ancestral_posterior_field(
    asr_profile: list[dict[str, float]] | None,
    wt_seq: str,
    positions: list[int],
    position_to_index: dict[int, int] | None = None,
    top_k: int = 3,
    min_prob: float = 0.10,
) -> dict[int, dict[str, Any]]:
    if asr_profile is None:
        return {}

    field: dict[int, dict[str, Any]] = {}
    for position in positions:
        idx = position_to_index[position] if position_to_index is not None else position - 1
        probs = asr_profile[idx] if idx < len(asr_profile) else {}
        if not probs:
            continue
        clean_probs = {
            aa: float(prob)
            for aa, prob in probs.items()
            if aa != "-" and float(prob) > 0.0
        }
        if not clean_probs:
            continue
        entropy = -sum(p * math.log(max(p, 1e-8)) for p in clean_probs.values())
        max_entropy = math.log(max(2, len(clean_probs)))
        confidence = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)
        wt = wt_seq[idx]
        ranked = sorted(clean_probs.items(), key=lambda item: (-item[1], item[0]))[: int(top_k)]
        recommendations = {
            aa: round(prob * max(0.25, confidence), 6)
            for aa, prob in ranked
            if aa != wt and prob >= min_prob
        }
        field[position] = {
            "wt_residue": wt,
            "posterior": {aa: round(prob, 6) for aa, prob in ranked},
            "entropy": round(float(entropy), 6),
            "confidence": round(float(confidence), 6),
            "recommendations": recommendations,
        }
    return field
