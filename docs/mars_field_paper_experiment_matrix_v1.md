# MARS-FIELD Paper Experiment Matrix v1

This document turns the current codebase and outputs into a paper-facing experiment plan.

It is deliberately split into:

- experiments already supported by current outputs
- experiments that are recommended for a stronger submission

## Central claim

The paper should be organized around this claim:

`MARS-FIELD` is a unified evidence-to-sequence controller that absorbs geometry, phylogeny, ancestry, retrieval memory, and environment-conditioned engineering constraints into a shared residue field, and uses a neuralized controller-decoder stack to propose and select calibrated protein designs.

In the current repository state, the strongest claim we can support is:

- unified field controller: yes
- neural field decoder in the main pipeline: yes
- fully joint proposal-generator / field / decoder training: not yet

## Primary benchmark panel

Use the twelve-target panel as the main benchmark:

| Target | Family | Primary role in paper |
| --- | --- | --- |
| `1LBT` | `lipase_b` | hard safety/control case |
| `tem1_1btl` | `beta_lactamase` | high-scoring engineering case |
| `petase_5xfy` | `cutinase` | stable canonical redesign case |
| `petase_5xh3` | `cutinase` | replicate PETase benchmark |
| `sfgfp_2b3p` | `gfp_like` | fluorescence-protein style surface redesign |
| `t4l_171l` | `lysozyme` | neural-decoder-assisted policy shift case |
| `subtilisin_2st1` | `subtilisin` | neural-decoder-assisted policy shift case |
| `adk_1s3g` | `adenylate_kinase` | strong engineering improvement case |
| `esterase_7b4q` | `lipase_esterase` | family-conditioned calibration case |
| `sod_1y67` | `superoxide_dismutase` | neural-decoder-assisted policy shift case |
| `CLD_3Q09_NOTOPIC` | `cld` | no-topic control case |
| `CLD_3Q09_TOPIC` | `cld` | topic-conditioned case |

## Main comparison arms

### Arm A. Current baseline benchmark

Purpose:

- define the incumbent engineering pipeline
- provide paired deltas for every target

Use:

- `outputs/benchmark_twelvepack/benchmark_summary.csv`
- `outputs/benchmark_twelvepack_current_summary.md` if needed for narrative backfill

### Arm B. Final hybrid controller

Purpose:

- primary method result for the paper
- safest deployable controller
- includes neural reranker and neural field decoder in the main pipeline

Use:

- `outputs/benchmark_twelvepack_final/benchmark_summary.csv`
- `outputs/benchmark_twelvepack_final/compare_current_vs_final.md`

### Arm C. Neural branch diagnostics

Purpose:

- show that the neural branch is active and interpretable
- support the “not just a heuristic voter” claim

Use:

- `outputs/benchmark_twelvepack_final/neural_comparison_summary.csv`
- `outputs/paper_bundle_v1/figures/figure_neural_comparison_v1.svg`
- `outputs/paper_bundle_v1/figures/figure_neural_branch_diagnostics_v1.svg`
- `outputs/paper_bundle_v1/figures/figure_policy_compare_v1.svg`

### Arm D. End-to-end neural decoder utilization

Purpose:

- demonstrate that the neural field is not only reranking but also decoding

Use:

- `outputs/benchmark_twelvepack_final/benchmark_summary.csv`
- per-target `neural_decoder_preview.json`
- per-target `neural_field_runtime_summary.json`

Current readout from the latest benchmark:

- neural decoder enabled on `12/12` targets
- neural decoder retained novel candidates on `5/12` targets
- total neural decoder preview candidates: `373`
- total retained novel neural decoder candidates: `34`
- total rejected neural decoder candidates: `215`

## Primary metrics

For the main text, use these metrics:

1. `policy_selection_score`
2. `policy_engineering_score` (`mars_score`)
3. number of targets with positive paired delta vs current
4. number of targets with negative paired delta vs current
5. decoder utilization:
   - enabled targets
   - preview candidate count
   - retained novel candidate count
   - rejected candidate count

For the latest final run, the current paired result is:

- policy score improved on `9/12`
- policy score decreased on `3/12`
- mean paired delta approximately `-0.001`

This is strong enough for a main-results narrative because:

- the end-to-end neural controller is active
- the average effect is approximately neutral
- gains are distributed across many targets
- remaining failures are explicit and analyzable

## Figures and tables

### Figure 1

Concept:

- method overview
- evidence streams -> shared residue field -> neural controller/decoder -> calibrated selection

Assets:

- `docs/mars_field_figure1_spec_v2.md`
- `outputs/paper_bundle_v1/figures/figure1_mars_field_architecture_v1.svg`

### Figure 2

Concept:

- benchmark overview on twelvepack
- current vs final paired comparison
- family-stratified summary

Assets:

- `outputs/paper_bundle_v1/figures/figure2_benchmark_overview_v3.svg`
- `outputs/benchmark_twelvepack_final/compare_current_vs_final.md`

### Figure 3

Concept:

- decoder calibration and neural-branch diagnostics
- show both acceptance gating and end-to-end neural decoder involvement

Assets:

- `outputs/paper_bundle_v1/figures/figure3_decoder_calibration_v3.svg`
- `outputs/paper_bundle_v1/figures/figure_neural_comparison_v1.svg`
- `outputs/paper_bundle_v1/figures/figure_neural_branch_diagnostics_v1.svg`
- `outputs/paper_bundle_v1/figures/figure_policy_compare_v1.svg`

### Figures 4-7

Recommended main case studies:

- `1LBT`
- `tem1_1btl`
- `petase_5xfy`
- `CLD_3Q09_TOPIC`

Recommended supplementary case studies:

- `esterase_7b4q`
- `t4l_171l`
- `subtilisin_2st1`
- `sod_1y67`

### Main text tables

Table 1:

- twelvepack benchmark panel
- target, family, design positions, priors enabled

Table 2:

- current vs final paired benchmark result
- one row per target
- include policy mutation, policy score, engineering score, delta sign

Table 3:

- neural decoder utilization table
- preview count, retained count, rejected count, best neural decoder candidate

Supplementary Table S1:

- per-target protocol manifest summary

Supplementary Table S2:

- per-target neural gate means and diagnostics

## Methods experiments section

The Methods section should explicitly describe these benchmarks:

### M1. Leave-one-target-out neural controller training

- each target is held out from the neural controller training set
- all other pipeline outputs are used as supervision

### M2. Final hybrid controller evaluation

- benchmark uses `selection_policy: hybrid`
- neural rerank is enabled
- neural decoder branch is enabled in the pipeline path

### M3. End-to-end neural decoder accounting

- count neural preview candidates
- count retained novel neural decoder candidates
- count safety-filtered neural decoder candidates

### M4. Case-study structural analysis

- use the current figure-grade structural panels
- compare incumbent winner, best learned candidate, and best neural-decoder-derived candidate when present

## Results section outline

### R1. Overall benchmark result

Headline:

- the final MARS-FIELD controller remains competitive with the incumbent pipeline while introducing a genuine neural field decoder into the main path

### R2. Neural decoder contributes meaningful novel proposals

Headline:

- the neural field is not only reranking but generating decode-time candidates inside the main pipeline

### R3. Hard-target stability and safe fallback

Headline:

- the hybrid controller preserves safety on hard cases such as `1LBT` while still allowing neural-driven gains on selected targets

### R4. Case studies

Case roles:

- `1LBT`: conservative safety-preserving controller behavior
- `TEM1`: high-scoring engineering landscape and decoder-assisted learned branch
- `PETase`: stable canonical redesign reproduced across two structures
- `CLD`: topic-conditioned vs non-topic behavior under a unified controller

## Recommended additional experiments before submission

These are not blockers for drafting the paper, but they would strengthen the submission:

1. Refresh a pure `neural-default` benchmark under the new end-to-end decoder path.
2. Add a dedicated ablation that disables neural decoder injection while keeping neural rerank on.
3. Add a dedicated ablation that disables ancestry memory and retrieval memory separately.
4. Add runtime/throughput accounting for:
   - baseline pipeline
   - final hybrid controller
   - end-to-end neural decoder path

## Recommendation

We should now proceed in this order:

1. write Methods and Results from current outputs
2. freeze the main-paper benchmark on `benchmark_twelvepack_final`
3. only then decide which supplementary ablations are worth adding before submission
