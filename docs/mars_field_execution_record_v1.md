# MARS-FIELD Execution Record v1

Use this file as the compact running log for the `v1 -> final` upgrade cycle.

The detailed historical log remains:

- `docs/mars_field_process_log.md`

This file should stay short and execution-oriented.

## 2026-04-18

### Activated roadmap

- Roadmap document created:
  - `docs/mars_field_v1_to_final_roadmap_v1.md`

### Working items now

- `Item 1` neural reranker into benchmark main comparison path
- `Item 4` score semantics unification
- `Item 5` benchmark protocol automation
- `Item 10` version / run metadata manifests

### Immediate implementation goals

- add standardized score fields to pipeline and benchmark outputs
- add benchmark protocol manifest files
- keep neural rerank outputs visible in benchmark summaries

### Completed in this cycle

- `v1.0` upload package generated:
  - `F:\4-15Marsprotein\release_packages\mars_field_engineering_approximation_v1_0`
  - `F:\4-15Marsprotein\release_packages\mars_field_engineering_approximation_v1_0.zip`
- neural reranker wired into benchmark comparison path
- neural outputs recorded in `benchmark_twelvepack`
- benchmark protocol manifest exported
- target pipelines refreshed so score-contract fields are written to candidate tables
- neural comparison summary exported for `benchmark_twelvepack`
- neural comparison figure renderer added
- richer retrieval / ancestry feature channels added to the neural branch
- neural gate-weight diagnostics exported
- ancestry alignment loss added
- retrieval alignment loss added
- gate regularization added
- environment-conditioned neural modulation added
- hybrid neural benchmark policy config added:
  - `configs/benchmark_twelvepack_neural_hybrid.yaml`
- neural-forward benchmark output produced:
  - `outputs/benchmark_twelvepack_neural_hybrid/`
- neural branch diagnostics figure produced:
  - `outputs/paper_bundle_v1/figures/figure_neural_branch_diagnostics_v1.svg`

### Next implementation batch

- continue learned retrieval / ancestry upgrades beyond richer feature channels
- turn neural comparison into a default paper-bundle figure/input
- start the next algorithmic layer: learned pairwise / environment upgrades

### Additional completed work

- refreshed `benchmark_twelvepack` through the new score contract
- refreshed target pipeline outputs so `selection_score` and `engineering_score` are present
- exported neural comparison aggregate:
  - `outputs/benchmark_twelvepack/neural_comparison_summary.csv`
- exported neural comparison figure:
  - `outputs/paper_bundle_v1/figures/figure_neural_comparison_v1.svg`
- expanded neural branch inputs:
  - richer ancestry feature channels
  - richer retrieval feature channels
  - expanded environment vector
- added neural gate diagnostics:
  - `neural_site_gates.json`
- added multi-objective neural training support:
  - regression loss
  - WT recovery loss
  - pairwise consistency loss
- refreshed paper bundle figure manifest and summary to include:
  - `figure2_benchmark_overview_v3.svg`
  - `figure3_decoder_calibration_v3.svg`
  - `figure_neural_comparison_v1.svg`
  - `figure_neural_branch_diagnostics_v1.svg`
  - `figure4_case_1lbt_v2.svg`
  - `figure5_case_tem1_v2.svg`

### Most recent additions

- neural-default benchmark path added:
  - `configs/benchmark_twelvepack_neural_default.yaml`
  - `outputs/benchmark_twelvepack_neural_default/`
- current vs hybrid comparison exported:
  - `outputs/benchmark_twelvepack_neural_hybrid/compare_current_vs_hybrid.md`
- current vs neural-default comparison exported:
  - `outputs/benchmark_twelvepack_neural_default/compare_current_vs_neural.md`
- neural policy comparison figure rendered:
  - `outputs/paper_bundle_v1/figures/figure_policy_compare_v1.svg`
- pipeline-level neural rerank path now writes:
  - `neural_selection_pred`
  - `neural_engineering_pred`
  - `neural_policy_score`
  - `neural_policy_summary.json`
- neural branch now has explicit candidate-level heads:
  - selection head
  - engineering head

### Latest neural-final push

- retrieval / ancestry branches upgraded from plain projections to prototype-memory fusion in:
  - `marsstack/field_network/neural_model.py`
- candidate-level evidence now enters the neural policy head through structured candidate features in:
  - `marsstack/field_network/neural_dataset.py`
- neural training now includes selector-calibration objectives:
  - winner guard loss
  - non-decoder guard loss
  - simplicity guard loss
  - selector-anchor distillation loss
- neural rerank outputs now expose:
  - `neural_policy_pred`
  - `neural_policy_z`
- pipeline candidate exports now carry the expanded neural policy fields:
  - `neural_policy_pred`
  - `neural_policy_z`
  - `neural_policy_score`
- verified pipeline-level neural rerank writeout on:
  - `outputs/1lbt_pipeline/`
- refreshed twelvepack comparison outputs after the neural-final push:
  - `outputs/benchmark_twelvepack/`
  - `outputs/benchmark_twelvepack_neural_hybrid/`
  - `outputs/benchmark_twelvepack_neural_default/`
- refreshed paper-bundle neural figures:
  - `outputs/paper_bundle_v1/figures/figure_neural_comparison_v1.svg`
  - `outputs/paper_bundle_v1/figures/figure_neural_branch_diagnostics_v1.svg`
  - `outputs/paper_bundle_v1/figures/figure_policy_compare_v1.svg`

### Current readout after refresh

- `neural-default` no longer shows broad collapse across the panel
- it now matches current policy exactly on:
  - `CLD_3Q09_TOPIC`
  - `adk_1s3g`
  - `petase_5xfy`
  - `petase_5xh3`
  - `sfgfp_2b3p`
  - `sod_1y67`
  - `subtilisin_2st1`
  - `t4l_171l`
- remaining neural-default regressions are now concentrated in:
  - `1LBT`
  - `CLD_3Q09_NOTOPIC`
  - `esterase_7b4q`
  - `tem1_1btl`
- hybrid remains the safest default transition policy after this batch

### Final-controller refinement

- added stronger selector-prior candidate features in:
  - `marsstack/field_network/neural_dataset.py`
- updated neural policy calibration to mix:
  - learned policy head
  - neural engineering head
  - neural selection head
  - selector prior
  - engineering prior
- esterase hard case is now recovered to the current winner under neural rerank
- TEM1 hard case is now stabilized under neural rerank
- refreshed twelvepack comparison after the final-controller pass:
  - exact policy matches in `neural-default`: `10/12`
  - remaining pure-neural mismatches: `1LBT`, `CLD_3Q09_TOPIC`
  - mean policy delta vs current: about `-0.026`
- added final benchmark entrypoint:
  - `configs/benchmark_twelvepack_final.yaml`
- added technical route note:
  - `docs/mars_field_final_technical_route_v2.md`
- tightened hybrid final-controller rule with a small selection tolerance so neural cannot replace a stable incumbent on marginal score loss
- verified final safe controller output:
  - `outputs/benchmark_twelvepack_final/compare_current_vs_final.md`
  - current vs final policy alignment on twelvepack: `12/12`

### End-to-end neural field pass

- added runtime neural batch construction:
  - `marsstack/field_network/neural_dataset.py`
- added neural field generator utilities:
  - `marsstack/field_network/neural_generator.py`
- added decoder-field supervision to neural training:
  - `decoder_field_loss`
- pipeline now builds and exports:
  - `neural_position_fields.json`
  - `neural_pairwise_energy_tensor.json`
  - `neural_decoder_preview.json`
  - `neural_field_runtime_summary.json`
- pipeline now supports a learned `neural_decoder` branch that feeds generated candidates back into the main ranking path
- benchmark runner now propagates `--neural-rerank true` into pipeline execution, so the full benchmark path includes the neural field decoder
- refreshed `benchmark_twelvepack_final` under the new end-to-end path

### Current end-to-end readout

- neural decoder enabled across twelvepack: `12/12` targets
- neural decoder injected novel candidates on: `5/12` targets
- total neural decoder preview candidates: `373`
- total neural decoder novel retained candidates: `34`
- total neural decoder rejected candidates: `215`
- policy delta vs the previous current benchmark:
  - positive on `9/12`
  - negative on `3/12`
  - mean approximately `-0.001`

### Manuscript drafting assets

- richer experiment matrix added:
  - `docs/mars_field_paper_experiment_matrix_v1.md`
- expanded manuscript draft added:
  - `docs/mars_field_methods_results_draft_v2.md`
- Word draft with figures and tables added:
  - `outputs/paper_bundle_v1/MARS_FIELD_Methods_Results_Draft_v2.docx`
- more Nature-like Word layout added:
  - `outputs/paper_bundle_v1/MARS_FIELD_Nature_Style_Methods_Results_v3.docx`
- reusable Word-build scripts added:
  - `scripts/build_mars_field_word_draft.py`
  - `scripts/build_mars_field_word_draft_v3.py`
- figure redesign planning added:
  - `docs/mars_field_figure_1_4_masterplan_v1.md`
  - `docs/mars_field_figure_1_4_cn_storyboard_v1.md`
- introduction and discussion draft added:
  - `docs/mars_field_intro_discussion_draft_v1.md`
- Chinese technical report added:
  - `docs/mars_field_technical_report_cn_v1.md`
  - `outputs/paper_bundle_v1/MARS_FIELD_中文技术报告_v1.docx`
- new benchmark claim and mechanism figures rendered:
  - `outputs/paper_bundle_v1/figures/figure2_benchmark_claim_v4.png`
  - `outputs/paper_bundle_v1/figures/figure3_mechanism_limitations_v4.png`
- selected references list added:
  - `docs/mars_field_selected_references_v1.md`
- submission-style manuscript draft with references added:
  - `outputs/paper_bundle_v1/MARS_FIELD_Submission_Style_v5.docx`
