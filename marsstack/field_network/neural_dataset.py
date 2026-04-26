from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_ORDER)}


@dataclass
class NeuralTargetBatch:
    target: str
    pipeline_dir: Path
    positions: list[int]
    wt_indices: np.ndarray
    geom_inputs: np.ndarray
    evo_inputs: np.ndarray
    asr_inputs: np.ndarray
    retrieval_inputs: np.ndarray
    env_inputs: np.ndarray
    pair_inputs: dict[tuple[int, int], np.ndarray]
    candidate_indices: np.ndarray
    candidate_features: np.ndarray
    candidate_scores: np.ndarray
    candidate_mars_scores: np.ndarray
    candidate_sources: list[str]
    candidate_mutations: list[str]
    candidate_feature_names: list[str]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _distribution_from_dict(payload: dict[str, Any] | None) -> np.ndarray:
    vec = np.zeros(len(AA_ORDER), dtype=np.float32)
    if not payload:
        return vec
    for aa, prob in payload.items():
        if aa in AA_TO_IDX:
            vec[AA_TO_IDX[aa]] = float(prob)
    return vec


def _recommendation_mass(payload: dict[str, Any] | None) -> tuple[float, float]:
    if not payload:
        return 0.0, 0.0
    values = [float(v) for v in payload.values()]
    if not values:
        return 0.0, 0.0
    return float(sum(values)), float(max(values))


def _split_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [token.strip() for token in text.split(";") if token.strip()]


def _mutation_count(mutations: Any) -> int:
    text = str(mutations or "").strip()
    if not text or text.upper() == "WT":
        return 0
    return len([token for token in text.split(";") if token.strip()])


def _candidate_feature_names() -> list[str]:
    return [
        "mutation_count_norm",
        "is_wt",
        "is_single_mut",
        "support_count_norm",
        "is_local_proposal",
        "is_baseline_mpnn",
        "is_mars_mpnn",
        "is_esm_if",
        "is_manual",
        "is_fusion_decoder",
        "group_heuristic_local",
        "group_learned",
        "group_manual_control",
        "group_decoder",
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
        "ranking_penalty",
        "note_manual_bias",
        "note_evolution_prior",
        "note_template_weighted",
        "note_low_burden",
        "note_decoder",
        "note_topic",
        "note_retrieval",
        "note_ancestral",
        "selection_score_scaled",
        "engineering_score_scaled",
        "ranking_score_raw_scaled",
        "ranking_score_z_scaled",
        "ranking_score_bounded_scaled",
        "ranking_score_mars_calibrated_scaled",
        "selector_rank_prior",
        "selector_rank_fraction",
        "selection_gap_to_best",
        "engineering_gap_to_best",
        "selection_engineering_gap",
    ]


def _candidate_feature_vector(
    row: pd.Series,
    row_index: int,
    total_rows: int,
    best_selection_score: float,
    best_engineering_score: float,
) -> np.ndarray:
    source = str(row.get("source", ""))
    source_group = str(row.get("source_group", ""))
    support_tokens = _split_tokens(row.get("supporting_sources", ""))
    note_tokens = _split_tokens(row.get("notes", ""))
    note_blob = " ".join(note_tokens).lower()
    mutation_count = _mutation_count(row.get("mutations", ""))
    selection_score = _safe_float(row.get("selection_score", row.get("ranking_score", row.get("mars_score", 0.0))))
    engineering_score = _safe_float(row.get("engineering_score", row.get("mars_score", 0.0)))
    ranking_score_raw = _safe_float(row.get("ranking_score_raw", selection_score))
    ranking_score_z = _safe_float(row.get("ranking_score_z", 0.0))
    ranking_score_bounded = _safe_float(row.get("ranking_score_bounded", selection_score))
    ranking_score_mars_calibrated = _safe_float(row.get("ranking_score_mars_calibrated", 0.0))
    selector_rank_prior = 1.0 / float(row_index + 1)
    selector_rank_fraction = 1.0 - (float(row_index) / float(max(1, total_rows - 1)))
    features = np.array(
        [
            mutation_count / 8.0,
            1.0 if mutation_count == 0 else 0.0,
            1.0 if mutation_count == 1 else 0.0,
            min(len(support_tokens), 8) / 8.0,
            1.0 if source == "local_proposal" else 0.0,
            1.0 if source == "baseline_mpnn" else 0.0,
            1.0 if source == "mars_mpnn" else 0.0,
            1.0 if source == "esm_if" else 0.0,
            1.0 if source == "manual" else 0.0,
            1.0 if source == "fusion_decoder" else 0.0,
            1.0 if source_group == "heuristic_local" else 0.0,
            1.0 if source_group == "learned" else 0.0,
            1.0 if source_group == "manual_control" else 0.0,
            1.0 if source == "fusion_decoder" else 0.0,
            _safe_float(row.get("score_oxidation", 0.0)) / 5.0,
            _safe_float(row.get("score_surface", 0.0)) / 5.0,
            _safe_float(row.get("score_manual", 0.0)) / 5.0,
            _safe_float(row.get("score_evolution", 0.0)) / 5.0,
            _safe_float(row.get("score_burden", 0.0)) / 5.0,
            _safe_float(row.get("score_topic_sequence", 0.0)) / 5.0,
            _safe_float(row.get("score_topic_structure", 0.0)) / 5.0,
            _safe_float(row.get("score_topic_evolution", 0.0)) / 5.0,
            _safe_float(row.get("fusion_linear_generator", 0.0)) / 5.0,
            _safe_float(row.get("fusion_linear_structure", 0.0)) / 5.0,
            _safe_float(row.get("fusion_linear_evolution", 0.0)) / 5.0,
            _safe_float(row.get("fusion_linear_consensus", 0.0)) / 5.0,
            _safe_float(row.get("fusion_linear_topic", 0.0)) / 5.0,
            _safe_float(row.get("fusion_linear_context", 0.0)) / 5.0,
            _safe_float(row.get("fusion_linear_misc", 0.0)) / 5.0,
            _safe_float(row.get("fusion_interaction", 0.0)) / 5.0,
            _safe_float(row.get("fusion_raw_feature_norm", 0.0)) / 50.0,
            _safe_float(row.get("ranking_penalty", 0.0)) / 5.0,
            1.0 if "manual_bias" in note_blob else 0.0,
            1.0 if "evolution_prior" in note_blob else 0.0,
            1.0 if "template_weighted" in note_blob else 0.0,
            1.0 if "low_burden" in note_blob else 0.0,
            1.0 if "decoder" in note_blob else 0.0,
            1.0 if "topic" in note_blob else 0.0,
            1.0 if "retrieval" in note_blob or "atlas" in note_blob else 0.0,
            1.0 if "ancestral" in note_blob or "asr" in note_blob else 0.0,
            selection_score / 10.0,
            engineering_score / 12.0,
            ranking_score_raw / 10.0,
            ranking_score_z / 4.0,
            ranking_score_bounded / 5.0,
            ranking_score_mars_calibrated / 2.0,
            selector_rank_prior,
            selector_rank_fraction,
            (best_selection_score - selection_score) / 10.0,
            (best_engineering_score - engineering_score) / 12.0,
            (selection_score - engineering_score) / 12.0,
        ],
        dtype=np.float32,
    )
    return features


def _extract_position_vectors(
    feature_rows: list[dict[str, Any]],
    positions: list[int],
    retrieval_payload: dict[str, Any],
    ancestral_payload: dict[str, Any],
    position_fields: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat_map = {int(row["num"]): row for row in feature_rows}
    field_map = {int(item["position"]): item for item in position_fields}
    retrieval_recs = retrieval_payload.get("recommendations", {})

    geom_vectors = []
    evo_vectors = []
    asr_vectors = []
    retrieval_vectors = []

    for position in positions:
        feat = feat_map[position]
        field_item = field_map[position]
        wt_residue = str(field_item["wt_residue"])

        geom_vectors.append(
            [
                _safe_float(feat.get("sasa")) / 120.0,
                _safe_float(feat.get("mean_b")) / 80.0,
                _safe_float(feat.get("min_dist_protected")) / 20.0,
                1.0 if bool(feat.get("in_disulfide")) else 0.0,
                1.0 if bool(feat.get("glyco_motif")) else 0.0,
                1.0 if wt_residue in {"M", "W", "Y", "H", "C"} else 0.0,
            ]
        )

        evo_vec = np.zeros(len(AA_ORDER), dtype=np.float32)
        for option in field_item.get("options", []):
            residue = str(option["residue"])
            evidence = option.get("evidence_breakdown", {})
            if residue in AA_TO_IDX:
                evo_vec[AA_TO_IDX[residue]] = float(
                    _safe_float(evidence.get("evolution_profile"))
                    + _safe_float(evidence.get("family_differential"))
                )
        evo_vectors.append(evo_vec.tolist())

        ancestral_info = ancestral_payload.get(str(position), ancestral_payload.get(position, {}))
        asr_posterior = _distribution_from_dict(ancestral_info.get("posterior", {}))
        asr_rec_mass, asr_rec_max = _recommendation_mass(ancestral_info.get("recommendations", {}))
        asr_vectors.append(
            np.concatenate(
                [
                    asr_posterior,
                    np.array(
                        [
                            float(_safe_float(ancestral_info.get("confidence", 0.0))),
                            float(_safe_float(ancestral_info.get("entropy", 0.0))) / 3.0,
                            asr_rec_mass / 5.0,
                            asr_rec_max / 5.0,
                        ],
                        dtype=np.float32,
                    ),
                ]
            ).tolist()
        )

        retrieval_dist = _distribution_from_dict(retrieval_recs.get(str(position), retrieval_recs.get(position, {})))
        retrieval_diag = retrieval_payload.get("neighbors", {}).get(str(position), retrieval_payload.get("neighbors", {}).get(position, []))
        neighbor_count = float(len(retrieval_diag))
        top_similarity = max((_safe_float(item.get("similarity", 0.0)) for item in retrieval_diag), default=0.0)
        top_weight = max((_safe_float(item.get("weight", 0.0)) for item in retrieval_diag), default=0.0)
        support_sum = float(sum(_safe_float(item.get("support_count", 0.0)) for item in retrieval_diag))
        unique_targets = float(len({target for item in retrieval_diag for target in item.get("support_targets", [])}))
        retrieval_vectors.append(
            np.concatenate(
                [
                    retrieval_dist,
                    np.array(
                        [
                            neighbor_count / 10.0,
                            top_similarity,
                            top_weight,
                            support_sum / 50.0,
                            unique_targets / 10.0,
                        ],
                        dtype=np.float32,
                    ),
                ]
            ).tolist()
        )

    return (
        np.array(geom_vectors, dtype=np.float32),
        np.array(evo_vectors, dtype=np.float32),
        np.array(asr_vectors, dtype=np.float32),
        np.array(retrieval_vectors, dtype=np.float32),
    )


def _pair_inputs(
    pair_payload: dict[str, Any],
    positions: list[int],
) -> dict[tuple[int, int], np.ndarray]:
    local_index = {pos: idx for idx, pos in enumerate(positions)}
    result: dict[tuple[int, int], np.ndarray] = {}
    for pair_key, values in pair_payload.items():
        pos_i, pos_j = [int(x) for x in str(pair_key).split("-")]
        if pos_i not in positions or pos_j not in positions:
            continue
        total_weight = float(sum(_safe_float(v) for v in values.values()))
        max_weight = float(max((_safe_float(v) for v in values.values()), default=0.0))
        entropy_like = 0.0
        if total_weight > 0:
            for value in values.values():
                p = _safe_float(value) / total_weight
                if p > 0:
                    entropy_like -= p * np.log(p)
        result[(local_index[pos_i], local_index[pos_j])] = np.array(
            [
                total_weight / 10.0,
                max_weight / 10.0,
                entropy_like / 3.0,
            ],
            dtype=np.float32,
        )
    return result


def _candidate_matrix(
    ranked_df: pd.DataFrame,
    positions: list[int],
    position_to_index: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    candidate_indices = []
    candidate_features = []
    candidate_scores = []
    candidate_mars_scores = []
    candidate_sources: list[str] = []
    candidate_mutations: list[str] = []
    feature_names = _candidate_feature_names()
    best_selection_score = float(ranked_df.get("selection_score", ranked_df.get("ranking_score", ranked_df.get("mars_score"))).max())
    best_engineering_score = float(ranked_df.get("engineering_score", ranked_df.get("mars_score")).max())
    total_rows = int(len(ranked_df))

    for row_index, (_, row) in enumerate(ranked_df.iterrows()):
        sequence = str(row["sequence"])
        residues = []
        for position in positions:
            idx = position_to_index[position]
            aa = sequence[idx]
            residues.append(AA_TO_IDX.get(aa, 0))
        candidate_indices.append(residues)
        candidate_features.append(
            _candidate_feature_vector(
                row,
                row_index=row_index,
                total_rows=total_rows,
                best_selection_score=best_selection_score,
                best_engineering_score=best_engineering_score,
            )
        )
        candidate_scores.append(float(row.get("ranking_score", row.get("mars_score", 0.0))))
        candidate_mars_scores.append(float(row.get("mars_score", 0.0)))
        candidate_sources.append(str(row.get("source", "")))
        candidate_mutations.append(str(row.get("mutations", "WT")))

    return (
        np.array(candidate_indices, dtype=np.int64),
        np.array(candidate_features, dtype=np.float32),
        np.array(candidate_scores, dtype=np.float32),
        np.array(candidate_mars_scores, dtype=np.float32),
        candidate_sources,
        candidate_mutations,
        feature_names,
    )


def load_neural_target_batch(pipeline_dir: Path) -> NeuralTargetBatch | None:
    feature_summary_path = pipeline_dir / "feature_summary.json"
    feature_table_path = pipeline_dir / "structure_features.csv"
    ranked_path = pipeline_dir / "combined_ranked_candidates.csv"
    retrieval_path = pipeline_dir / "retrieval_memory_hits.json"
    ancestral_path = pipeline_dir / "ancestral_field.json"
    pairwise_path = pipeline_dir / "pairwise_energy_tensor.json"
    position_fields_path = pipeline_dir / "position_fields.json"
    profile_summary_path = pipeline_dir / "profile_summary.json"
    if not all(path.exists() for path in [feature_summary_path, feature_table_path, ranked_path, retrieval_path, ancestral_path, pairwise_path, position_fields_path]):
        return None

    feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
    profile_summary = json.loads(profile_summary_path.read_text(encoding="utf-8")) if profile_summary_path.exists() else {}
    feature_rows = pd.read_csv(feature_table_path).to_dict(orient="records")
    ranked_df = pd.read_csv(ranked_path)
    if ranked_df.empty:
        return None
    retrieval_payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
    ancestral_payload = json.loads(ancestral_path.read_text(encoding="utf-8"))
    pair_payload = json.loads(pairwise_path.read_text(encoding="utf-8"))
    position_fields = json.loads(position_fields_path.read_text(encoding="utf-8"))

    positions = [int(x) for x in feature_summary["design_positions"]]
    wt_sequence = str(ranked_df.iloc[0]["sequence"])
    residue_numbers = [int(row["num"]) for row in feature_rows]
    position_to_index = {num: idx for idx, num in enumerate(residue_numbers)}
    wt_indices = np.array([AA_TO_IDX.get(wt_sequence[position_to_index[pos]], 0) for pos in positions], dtype=np.int64)
    geom_inputs, evo_inputs, asr_inputs, retrieval_inputs = _extract_position_vectors(
        feature_rows=feature_rows,
        positions=positions,
        retrieval_payload=retrieval_payload,
        ancestral_payload=ancestral_payload,
        position_fields=position_fields,
    )
    env_inputs = np.array(
        [
            float(len(feature_summary.get("oxidation_hotspots", []))) / 20.0,
            float(len(feature_summary.get("flexible_surface_positions", []))) / 40.0,
            float(len(positions)) / 10.0,
            float(_safe_float(profile_summary.get("accepted_homologs", 0.0))) / 20.0,
            float(_safe_float(profile_summary.get("accepted_asr", 0.0))) / 10.0,
            1.0 if bool(profile_summary.get("family_prior_enabled", False)) else 0.0,
            1.0 if bool(profile_summary.get("template_weighting_enabled", False)) else 0.0,
            1.0 if bool(profile_summary.get("asr_prior_enabled", False)) else 0.0,
        ],
        dtype=np.float32,
    )
    pair_inputs = _pair_inputs(pair_payload, positions)
    candidate_indices, candidate_features, candidate_scores, candidate_mars_scores, candidate_sources, candidate_mutations, candidate_feature_names = _candidate_matrix(
        ranked_df=ranked_df,
        positions=positions,
        position_to_index=position_to_index,
    )
    return NeuralTargetBatch(
        target=str(feature_summary.get("protein", pipeline_dir.name.replace("_pipeline", ""))),
        pipeline_dir=pipeline_dir,
        positions=positions,
        wt_indices=wt_indices,
        geom_inputs=geom_inputs,
        evo_inputs=evo_inputs,
        asr_inputs=asr_inputs,
        retrieval_inputs=retrieval_inputs,
        env_inputs=env_inputs,
        pair_inputs=pair_inputs,
        candidate_indices=candidate_indices,
        candidate_features=candidate_features,
        candidate_scores=candidate_scores,
        candidate_mars_scores=candidate_mars_scores,
        candidate_sources=candidate_sources,
        candidate_mutations=candidate_mutations,
        candidate_feature_names=candidate_feature_names,
    )


def load_neural_corpus(outputs_root: Path, include_targets: list[str] | None = None) -> list[NeuralTargetBatch]:
    include = set(include_targets or [])
    batches: list[NeuralTargetBatch] = []
    for pipeline_dir in sorted(outputs_root.glob("*_pipeline")):
        if include:
            pipeline_name = pipeline_dir.name
            bare_name = pipeline_name.replace("_pipeline", "")
            if pipeline_name not in include and bare_name not in include:
                continue
        batch = load_neural_target_batch(pipeline_dir)
        if batch is None:
            continue
        if include and batch.target not in include and pipeline_dir.name not in include and batch.target.lower() not in {item.lower() for item in include}:
            continue
        batches.append(batch)
    return batches


def _serialize_position_fields_like(position_fields: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for field in position_fields:
        if isinstance(field, dict):
            payload.append(field)
            continue
        payload.append(
            {
                "position": int(getattr(field, "position")),
                "wt_residue": str(getattr(field, "wt_residue")),
                "options": [
                    {
                        "residue": str(getattr(option, "residue")),
                        "score": float(getattr(option, "score")),
                        "supporting_sources": list(getattr(option, "supporting_sources", []) or []),
                        "support_strength": float(getattr(option, "support_strength", 0.0)),
                        "evidence_breakdown": dict(getattr(option, "evidence_breakdown", {}) or {}),
                    }
                    for option in list(getattr(field, "options"))
                ],
            }
        )
    return payload


def build_runtime_neural_target_batch(
    target: str,
    pipeline_dir: Path,
    feature_rows: list[dict[str, Any]],
    ranked_rows: list[dict[str, Any]],
    retrieval_payload: dict[str, Any],
    ancestral_payload: dict[str, Any],
    pair_payload: dict[str, Any],
    position_fields: list[Any],
    profile_summary: dict[str, Any] | None = None,
) -> NeuralTargetBatch | None:
    if not ranked_rows:
        return None
    ranked_df = pd.DataFrame(ranked_rows).copy()
    if ranked_df.empty:
        return None
    if "ranking_score" in ranked_df.columns:
        ranked_df = ranked_df.sort_values(["ranking_score", "mars_score"], ascending=[False, False]).reset_index(drop=True)
    position_fields_payload = _serialize_position_fields_like(position_fields)
    positions = [int(item["position"]) for item in position_fields_payload]
    if not positions:
        return None

    wt_sequence = str(ranked_df.iloc[0]["sequence"])
    residue_numbers = [int(row["num"]) for row in feature_rows]
    position_to_index = {num: idx for idx, num in enumerate(residue_numbers)}
    wt_indices = np.array([AA_TO_IDX.get(wt_sequence[position_to_index[pos]], 0) for pos in positions], dtype=np.int64)
    geom_inputs, evo_inputs, asr_inputs, retrieval_inputs = _extract_position_vectors(
        feature_rows=feature_rows,
        positions=positions,
        retrieval_payload=retrieval_payload,
        ancestral_payload=ancestral_payload,
        position_fields=position_fields_payload,
    )
    profile_summary = dict(profile_summary or {})
    env_inputs = np.array(
        [
            float(len(profile_summary.get("oxidation_hotspots", []))) / 20.0,
            float(len(profile_summary.get("flexible_surface_positions", []))) / 40.0,
            float(len(positions)) / 10.0,
            float(_safe_float(profile_summary.get("accepted_homologs", 0.0))) / 20.0,
            float(_safe_float(profile_summary.get("accepted_asr", 0.0))) / 10.0,
            1.0 if bool(profile_summary.get("family_prior_enabled", False)) else 0.0,
            1.0 if bool(profile_summary.get("template_weighting_enabled", False)) else 0.0,
            1.0 if bool(profile_summary.get("asr_prior_enabled", False)) else 0.0,
        ],
        dtype=np.float32,
    )
    pair_inputs = _pair_inputs(pair_payload, positions)
    candidate_indices, candidate_features, candidate_scores, candidate_mars_scores, candidate_sources, candidate_mutations, candidate_feature_names = _candidate_matrix(
        ranked_df=ranked_df,
        positions=positions,
        position_to_index=position_to_index,
    )
    return NeuralTargetBatch(
        target=str(target),
        pipeline_dir=pipeline_dir,
        positions=positions,
        wt_indices=wt_indices,
        geom_inputs=geom_inputs,
        evo_inputs=evo_inputs,
        asr_inputs=asr_inputs,
        retrieval_inputs=retrieval_inputs,
        env_inputs=env_inputs,
        pair_inputs=pair_inputs,
        candidate_indices=candidate_indices,
        candidate_features=candidate_features,
        candidate_scores=candidate_scores,
        candidate_mars_scores=candidate_mars_scores,
        candidate_sources=candidate_sources,
        candidate_mutations=candidate_mutations,
        candidate_feature_names=candidate_feature_names,
    )
