from __future__ import annotations

from dataclasses import dataclass

from ..evolution import differential_profile_score, profile_log_score


HYDROPHOBIC = set("AILMFWVY")
POLAR = set("NQST")
CHARGED = set("DEKRH")
LOW_COMPLEXITY_CORE = set("QSGNAEKDTP")
OXIDATION_PRONE = set("MWYCH")
BURIED_BREAKERS = set("PGDEKR")
MICROGRAVITY_SURFACE_FAVORABLE = set("NQSTDEKRHA")
MICROGRAVITY_STICKY = set("FWYMLIVC")
MICROGRAVITY_FLEX_RISK = set("PGFWYMC")

SAFE_OXIDATION_TOPIC_MAP = {
    "M": {"L": 3.0, "I": 2.6, "V": 1.8},
    "C": {"S": 2.0, "A": 1.4},
    "W": {"F": 1.6, "Y": 0.8},
    "Y": {"F": 1.2},
    "H": {"Q": 1.1, "N": 0.8},
}


@dataclass
class TopicScoreResult:
    total: float
    components: dict[str, float]
    notes: list[str]


def merge_cfg(defaults: dict[str, object], overrides: dict[str, object] | None) -> dict[str, object]:
    merged = dict(defaults)
    if overrides:
        merged.update(overrides)
    return merged


def fraction(seq: str, alphabet: set[str]) -> float:
    if not seq:
        return 0.0
    return sum(1 for aa in seq if aa in alphabet) / len(seq)


def net_charge(seq: str) -> float:
    charge_map = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}
    return sum(charge_map.get(aa, 0.0) for aa in seq)


def mutated_positions(
    wt_seq: str,
    seq: str,
    mutable_positions: list[int] | None,
    position_to_index: dict[int, int] | None,
) -> list[int]:
    if mutable_positions is None:
        positions = list(range(1, len(wt_seq) + 1))
    else:
        positions = list(mutable_positions)
    changed = []
    for pos in positions:
        idx = position_to_index[pos] if position_to_index is not None else pos - 1
        if wt_seq[idx] != seq[idx]:
            changed.append(pos)
    return changed


def score_profile_bundle(
    seq: str,
    positions: list[int],
    position_to_index: dict[int, int] | None,
    profile: list[dict[str, float]] | None,
    asr_profile: list[dict[str, float]] | None,
    family_positive_profile: list[dict[str, float]] | None,
    family_negative_profile: list[dict[str, float]] | None,
    profile_scale: float,
    asr_scale: float,
    family_scale: float,
) -> float:
    total = 0.0
    if profile is not None:
        total += profile_scale * profile_log_score(seq, profile, positions, position_to_index=position_to_index)
    if asr_profile is not None:
        total += asr_scale * profile_log_score(seq, asr_profile, positions, position_to_index=position_to_index)
    if family_positive_profile is not None and family_negative_profile is not None:
        total += family_scale * differential_profile_score(
            seq,
            family_positive_profile,
            family_negative_profile,
            positions,
            position_to_index=position_to_index,
        )
    return total


def finalize_result(components: dict[str, float], notes: list[str]) -> TopicScoreResult:
    total = round(sum(components.values()), 3)
    return TopicScoreResult(
        total=total,
        components={k: round(v, 3) for k, v in components.items()},
        notes=notes,
    )
