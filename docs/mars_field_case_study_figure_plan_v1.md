# MARS-FIELD Case-Study Figure Plan v1

## Goal

This document defines the figure structure for the protein-level case studies.

The aim is to avoid showing isolated nice-looking structures. Each case-study figure must connect:

- benchmark evidence
- field / decoder behavior
- structural interpretation
- final engineering decision

## Selected Primary Cases

### Figure 4. `1LBT`

Role in the paper:

- clean compact benchmark case
- easiest target for explaining the residue field idea
- strongest bridge between geometric evidence, decoder behavior, and final winner selection

Recommended panels:

- `A` target structure with protected and design positions
- `B` local residue field around positions `249 / 251 / 298`
- `C` decoder preview vs final ranked candidates
- `D` retrieval memory / motif support snapshot
- `E` final winner rationale: `R249Q;A251S;M298L`
- `F` compact table of top 5 candidates and calibration notes

Primary assets:

- `outputs/1lbt_pipeline/viz_bundle/scene.pml`
- `outputs/1lbt_pipeline/position_fields.json`
- `outputs/1lbt_pipeline/pairwise_energy_tensor.json`
- `outputs/1lbt_pipeline/retrieval_memory_hits.json`
- `outputs/1lbt_pipeline/pipeline_summary.md`

### Figure 5. `TEM1 / 1BTL`

Role in the paper:

- strongest multi-site engineering case
- demonstrates that the system remains engineering-consistent instead of collapsing to unsafe learned winners

Recommended panels:

- `A` structure with oxidative or exposed positions highlighted
- `B` candidate-source comparison: Mars / ESM-IF / local / final
- `C` selector calibration view showing why local winner remains preferred
- `D` mutation map for `H153N;M155L;W229F;M272L`
- `E` benchmark context against other targets

Primary assets:

- `outputs/tem1_1btl_pipeline/viz_bundle/scene.pml`
- `outputs/tem1_1btl_pipeline/position_fields.json`
- `outputs/tem1_1btl_pipeline/pairwise_energy_tensor.json`
- `outputs/tem1_1btl_pipeline/pipeline_summary.md`

### Figure 6. `PETase / 5XFY + 5XH3`

Role in the paper:

- family-transfer story
- paired structures let us show consistency across related cutinase backbones

Recommended panels:

- `A` paired structure overview of `5XFY` and `5XH3`
- `B` aligned design windows and shared mutation logic
- `C` benchmark summary for both PETase targets
- `D` candidate agreement and divergence across the pair
- `E` field evidence supporting `Y3F / Y40F / Y41F / Y117*`
- `F` short interpretation of transfer consistency

Primary assets:

- `outputs/petase_5xfy_pipeline/viz_bundle/scene.pml`
- `outputs/petase_5xh3_pipeline/viz_bundle/scene.pml`
- `outputs/petase_5xfy_pipeline/position_fields.json`
- `outputs/petase_5xh3_pipeline/position_fields.json`
- `outputs/petase_5xfy_pipeline/pipeline_summary.md`
- `outputs/petase_5xh3_pipeline/pipeline_summary.md`

### Figure 7. `CLD / 3Q09`

Role in the paper:

- strongest ancestry-aware biology bridge
- best target for justifying the `Ancestral Lineage Encoder`
- topic / no-topic pair gives an internal control

Recommended panels:

- `A` Cld structure with oxidation shell / design positions
- `B` ancestral field visualization or posterior summary
- `C` topic vs no-topic benchmark comparison
- `D` retrieval + ancestry evidence convergence
- `E` final design comparison around `W155 / W156 / M167 / M212 / W227`
- `F` summary diagram linking lineage evidence to final selection

Primary assets:

- `outputs/cld_3q09_topic_pipeline/viz_bundle/scene.pml`
- `outputs/cld_3q09_notopic_pipeline/viz_bundle/scene.pml`
- `outputs/cld_3q09_topic_pipeline/ancestral_field.json`
- `outputs/cld_3q09_notopic_pipeline/ancestral_field.json`
- `outputs/cld_3q09_topic_pipeline/position_fields.json`
- `outputs/cld_3q09_notopic_pipeline/position_fields.json`

## Secondary Cases

Reserve for supplement or rebuttal:

- `esterase_7b4q`
- `t4l_171l`
- `subtilisin_2st1`
- `adk_1s3g`
- `sod_1y67`
- `sfgfp_2b3p`

These are useful for:

- decoder calibration diagnostics
- family-prior examples
- robustness checks

## Figure Design Rules

Each case-study figure should include:

- one structure panel
- one model-internal evidence panel
- one candidate ranking panel
- one short interpretation panel

Avoid:

- large raw tables in the main figure
- too many sequence strings
- unannotated PyMOL screenshots

## Current Recommendation

The main manuscript should center the narrative around:

1. `Figure 1`: architecture
2. `Figure 2`: twelvepack benchmark overview
3. `Figure 3`: decoder / calibration analysis
4. `Figure 4`: `1LBT`
5. `Figure 5`: `TEM1`
6. `Figure 6`: `PETase`
7. `Figure 7`: `CLD`
