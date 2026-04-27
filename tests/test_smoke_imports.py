"""Smoke test: every documented public symbol imports cleanly."""

from __future__ import annotations


def test_marsstack_core_modules_import():
    import marsstack.ancestral_field  # noqa: F401
    import marsstack.decoder  # noqa: F401
    import marsstack.energy_head  # noqa: F401
    import marsstack.evidence_field  # noqa: F401
    import marsstack.evolution  # noqa: F401
    import marsstack.mars_score  # noqa: F401
    import marsstack.retrieval_memory  # noqa: F401
    import marsstack.structure_features  # noqa: F401
    import marsstack.unified_generator  # noqa: F401


def test_fusion_ranker_public_api():
    from marsstack.fusion_ranker import (
        FusionRankerConfig,
        OutputContext,
        TrainingCorpus,
        apply_learned_fusion_ranking,
        apply_target_score_calibration,
        build_corpus,
        build_feature_dict,
        build_feature_matrix,
        explain_rows,
        feature_group,
        fm_gradients,
        fm_score,
        load_context,
        load_training_tables,
        rank_rows_with_model,
        score_feature_matrix,
        sigmoid,
        train_factor_ranker,
    )
    assert callable(apply_learned_fusion_ranking)
    assert callable(rank_rows_with_model)
    assert callable(fm_score)


def test_topic_score_public_api():
    from marsstack.topic_score import (
        TopicHandlers,
        TopicScoreResult,
        build_topic_local_recommendations,
        register_topic,
        registered_topics,
        score_topic_candidate,
    )
    assert set(registered_topics()) == {"aresg", "cld", "drwh", "microgravity"}
    assert TopicHandlers  # dataclass class is truthy
    assert TopicScoreResult


def test_pipeline_public_api():
    from marsstack.pipeline import (
        ALPHABET,
        build_bias_and_omit,
        build_parsed_index_maps,
        collapse_mpnn_sequence,
        load_parsed_chain_sequence,
        materialize_decoded_candidate_rows,
        merge_recommendation_maps,
        normalize_parsed_names,
        preprocess_pdb,
        project_to_design_positions,
        resolve_project_path,
        restore_template_mismatches,
        summarize_aligned_entries,
    )
    assert ALPHABET == "ACDEFGHIKLMNPQRSTVWYX"


def test_field_network_public_api():
    from marsstack.field_network import (
        EvidenceBundle,
        EvidencePaths,
        FieldBuildResult,
        MarsFieldSystem,
        ProteinDesignContext,
        ScoringInputs,
        build_local_proposal_candidates,
        classify_source_group,
        parse_mpnn_fasta,
        parse_sample_fasta,
        register_candidate,
        score_candidate_rows,
        write_shortlist_fasta,
    )
    assert callable(build_local_proposal_candidates)
    assert callable(classify_source_group)
