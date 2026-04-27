"""Evidence-field aggregation invariants."""

from __future__ import annotations

from marsstack.evidence_field import build_unified_evidence_fields, serialize_evidence_fields
from marsstack.structure_features import ResidueFeature


def make_feature(num: int, aa: str, sasa: float) -> ResidueFeature:
    return ResidueFeature(
        num=num,
        name="ALA",
        aa=aa,
        sasa=sasa,
        mean_b=20.0,
        min_dist_protected=12.0,
        in_disulfide=False,
        glyco_motif=False,
    )


def test_build_unified_evidence_fields_smoke():
    wt_seq = "MAACDEFG"
    design_positions = [3, 5]
    position_to_index = {i: i - 1 for i in range(1, len(wt_seq) + 1)}
    features = [make_feature(i, wt_seq[i - 1], 35.0) for i in range(1, len(wt_seq) + 1)]

    proposal_rows = [
        {
            "candidate_id": "p1",
            "source": "mars_mpnn",
            "source_group": "learned",
            "sequence": "MALCDEFG",
            "ranking_score": 1.0,
            "mars_score": 1.0,
        },
        {
            "candidate_id": "p2",
            "source": "manual",
            "source_group": "manual_control",
            "sequence": "MAQCDEFG",
            "ranking_score": 0.5,
            "mars_score": 0.5,
        },
    ]

    fields = build_unified_evidence_fields(
        wt_seq=wt_seq,
        design_positions=design_positions,
        position_to_index=position_to_index,
        features=features,
        oxidation_hotspots=[],
        flexible_positions=design_positions,
        manual_bias={5: {"E": 0.5}},
        profile=None,
        family_recommendations=None,
        ancestral_field=None,
        retrieval_recommendations=None,
        proposal_rows=proposal_rows,
        top_k_per_position=4,
    )

    assert {f.position for f in fields} == set(design_positions)
    for field in fields:
        assert all(option.score >= 0 or option.residue == field.wt_residue for option in field.options)


def test_serialize_evidence_fields_round_trips_keys():
    wt_seq = "AAAA"
    fields = build_unified_evidence_fields(
        wt_seq=wt_seq,
        design_positions=[2],
        position_to_index={1: 0, 2: 1, 3: 2, 4: 3},
        features=[make_feature(i, "A", 35.0) for i in range(1, 5)],
        oxidation_hotspots=[],
        flexible_positions=[],
        manual_bias={2: {"C": 0.7}},
        profile=None,
        family_recommendations=None,
        ancestral_field=None,
        retrieval_recommendations=None,
        proposal_rows=[],
    )
    payload = serialize_evidence_fields(fields)
    assert payload[0]["position"] == 2
    assert any(option["residue"] == "C" for option in payload[0]["options"])
