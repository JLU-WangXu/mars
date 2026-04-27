from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marsstack.mars_score import score_candidate
from marsstack.structure_features import ResidueFeature


REPORT_PATH = ROOT / "reports" / "mars_resilience_stack" / "topic_scoring_smoke_test_2026-04-18.md"


def make_profile(length: int, supported: dict[int, dict[str, float]]) -> list[dict[str, float]]:
    profile = [{} for _ in range(length)]
    for pos, mapping in supported.items():
        profile[pos - 1] = dict(mapping)
    return profile


def set_seq_chars(length: int, updates: dict[int, str], fill: str = "A") -> str:
    chars = [fill] * length
    for pos, aa in updates.items():
        chars[pos - 1] = aa
    return "".join(chars)


def build_feature(num: int, aa: str, sasa: float, b: float = 25.0) -> ResidueFeature:
    return ResidueFeature(
        num=num,
        name="ALA",
        aa=aa,
        sasa=sasa,
        mean_b=b,
        min_dist_protected=12.0,
        in_disulfide=False,
        glyco_motif=False,
    )


def run_cld_case() -> tuple[str, float, dict[str, float], list[str]]:
    length = 300
    wt = set_seq_chars(length, {189: "W", 190: "W", 204: "H", 217: "R", 254: "E", 261: "W"})
    seq = set_seq_chars(length, {189: "F", 190: "F", 204: "Q", 217: "R", 254: "E", 261: "F"})
    features = [
        build_feature(189, "W", 18.0),
        build_feature(190, "W", 18.0),
        build_feature(204, "H", 12.0),
        build_feature(217, "R", 26.0),
        build_feature(254, "E", 14.0),
        build_feature(261, "W", 19.0),
    ]
    profile = make_profile(length, {189: {"F": 0.6}, 190: {"F": 0.6}, 204: {"Q": 0.7}, 261: {"F": 0.65}})
    res = score_candidate(
        wt_seq=wt,
        seq=seq,
        features=features,
        oxidation_hotspots=[],
        flexible_positions=[],
        profile=profile,
        asr_profile=profile,
        family_positive_profile=profile,
        family_negative_profile=make_profile(length, {189: {"W": 0.6}, 190: {"W": 0.6}}),
        manual_preferred={},
        evolution_positions=[189, 190, 204, 217, 254, 261],
        mutable_positions=[189, 190, 204, 261],
        position_to_index={i: i - 1 for i in range(1, length + 1)},
        topic_name="cld",
        topic_cfg={
            "enabled": True,
            "name": "cld",
            "cld": {
                "functional_shell_positions": [189, 190, 204, 217, 254, 261],
                "oxidation_guard_positions": [189, 190, 261],
                "distal_gate_positions": [217],
                "proximal_network_positions": [204, 254],
            },
        },
    )
    return "Cld", res.total, res.components, res.notes


def run_drwh_case() -> tuple[str, float, dict[str, float], list[str]]:
    motif = "VLANQDGPAA"
    wt = motif * 12
    seq_list = list(wt)
    for pos, aa in {12: "M", 28: "Q", 39: "L", 58: "N", 72: "E"}.items():
        seq_list[pos - 1] = aa
    seq = "".join(seq_list)
    features = [
        build_feature(12, wt[11], 12.0),
        build_feature(28, wt[27], 44.0),
        build_feature(39, wt[38], 15.0),
        build_feature(58, wt[57], 42.0),
        build_feature(72, wt[71], 46.0),
    ]
    profile = make_profile(len(wt), {12: {"M": 0.5}, 28: {"Q": 0.6}, 39: {"L": 0.55}, 58: {"N": 0.65}, 72: {"E": 0.6}})
    res = score_candidate(
        wt_seq=wt,
        seq=seq,
        features=features,
        oxidation_hotspots=[],
        flexible_positions=[],
        profile=profile,
        asr_profile=profile,
        family_positive_profile=profile,
        family_negative_profile=make_profile(len(wt), {12: {"P": 0.6}}),
        manual_preferred={},
        evolution_positions=[12, 28, 39, 58, 72],
        mutable_positions=[12, 28, 39, 58, 72],
        position_to_index={i: i - 1 for i in range(1, len(wt) + 1)},
        topic_name="drwh",
        topic_cfg={"enabled": True, "name": "drwh", "drwh": {}},
    )
    return "DrwH", res.total, res.components, res.notes


def run_aresg_case() -> tuple[str, float, dict[str, float], list[str]]:
    motif = "QSGNALVAAA"
    wt = motif * 12
    seq_list = list(wt)
    for pos, aa in {15: "Q", 22: "N", 37: "E", 64: "T", 88: "A"}.items():
        seq_list[pos - 1] = aa
    seq = "".join(seq_list)
    features = [
        build_feature(15, wt[14], 46.0),
        build_feature(22, wt[21], 41.0),
        build_feature(37, wt[36], 38.0),
        build_feature(64, wt[63], 40.0),
        build_feature(88, wt[87], 12.0),
    ]
    profile = make_profile(len(wt), {15: {"Q": 0.6}, 22: {"N": 0.65}, 37: {"E": 0.5}, 64: {"T": 0.55}, 88: {"A": 0.7}})
    res = score_candidate(
        wt_seq=wt,
        seq=seq,
        features=features,
        oxidation_hotspots=[],
        flexible_positions=[],
        profile=profile,
        asr_profile=profile,
        family_positive_profile=profile,
        family_negative_profile=make_profile(len(wt), {15: {"W": 0.6}}),
        manual_preferred={},
        evolution_positions=[15, 22, 37, 64, 88],
        mutable_positions=[15, 22, 37, 64, 88],
        position_to_index={i: i - 1 for i in range(1, len(wt) + 1)},
        topic_name="aresg",
        topic_cfg={"enabled": True, "name": "aresg", "aresg": {}},
    )
    return "AresG", res.total, res.components, res.notes


def run_microgravity_case() -> tuple[str, float, dict[str, float], list[str]]:
    motif = "AVLQNTAALA"
    wt = motif * 12
    seq_list = list(wt)
    for pos, aa in {12: "Q", 19: "E", 32: "N", 39: "K", 53: "V"}.items():
        seq_list[pos - 1] = aa
    seq = "".join(seq_list)
    features = [
        build_feature(12, wt[11], 48.0, 41.0),
        build_feature(19, wt[18], 45.0, 35.0),
        build_feature(32, wt[31], 43.0, 34.0),
        build_feature(39, wt[38], 46.0, 39.0),
        build_feature(53, wt[52], 14.0, 18.0),
    ]
    profile = make_profile(len(wt), {12: {"Q": 0.55}, 19: {"E": 0.6}, 32: {"N": 0.55}, 39: {"K": 0.52}, 53: {"V": 0.7}})
    res = score_candidate(
        wt_seq=wt,
        seq=seq,
        features=features,
        oxidation_hotspots=[],
        flexible_positions=[],
        profile=profile,
        asr_profile=profile,
        family_positive_profile=profile,
        family_negative_profile=make_profile(len(wt), {12: {"W": 0.7}, 19: {"F": 0.65}, 39: {"Y": 0.55}}),
        manual_preferred={},
        evolution_positions=[12, 19, 32, 39, 53],
        mutable_positions=[12, 19, 32, 39, 53],
        position_to_index={i: i - 1 for i in range(1, len(wt) + 1)},
        topic_name="microgravity",
        topic_cfg={"enabled": True, "name": "microgravity", "microgravity": {}},
    )
    return "Microgravity", res.total, res.components, res.notes


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [run_cld_case(), run_drwh_case(), run_aresg_case(), run_microgravity_case()]
    lines = [
        "# Topic Scoring Smoke Test",
        "",
        "Date: 2026-04-18",
        "",
        "| module | total_score | topic_sequence | topic_structure | topic_evolution |",
        "| --- | --- | --- | --- | --- |",
    ]
    for module, total, components, _ in rows:
        lines.append(
            f"| {module} | {total:.3f} | {components.get('topic_sequence', 0.0):.3f} | {components.get('topic_structure', 0.0):.3f} | {components.get('topic_evolution', 0.0):.3f} |"
        )
    lines.extend(["", "## Notes", ""])
    for module, total, components, notes in rows:
        lines.append(f"### {module}")
        lines.append(f"- total_score: {total:.3f}")
        lines.append(f"- components: {components}")
        lines.append(f"- notes: {'; '.join(notes[:12])}")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
