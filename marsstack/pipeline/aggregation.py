from __future__ import annotations


def summarize_aligned_entries(
    entries: list[tuple[str, str]],
    wt_seq: str,
) -> tuple[int, float]:
    """Compute ``(accepted, mean_coverage)`` for ``anchored_align_entries`` output.

    The first entry is treated as the WT reference and is excluded from
    coverage / accepted counts.
    """
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
    """Merge multiple ``{position: {aa: score}}`` recommendation maps.

    Per (position, aa), the *maximum* score across maps wins, then per position
    only the top-``top_k`` AAs by score are kept.
    """
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
