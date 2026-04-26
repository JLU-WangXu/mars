# GitHub Release Manifest v1

## Release Identity

Recommended GitHub release name:

`MARS-FIELD engineering approximation v1`

Recommended short description:

`A benchmarked protein engineering research prototype with retrieval, evolution, ancestry, decoder, calibration, and figure-generation support.`

## Honest Status

This release is ready to share as:

- a research prototype
- a reproducible engineering workflow
- a benchmarked codebase with manuscript-oriented figures

This release is **not** ready to claim:

- a fully learned end-to-end neural field model
- a finished production package
- final algorithmic closure on every planned module

## Recommended To Include In The GitHub Upload

- `README.md`
- `.gitignore`
- `configs/`
- `docs/`
- `marsstack/`
- `scripts/`
- `datasets/README.md`
- `inputs/benchmark/`
- `inputs/*` files actually required by the public examples
- `outputs/paper_bundle_v1/`
- `vendors/ProteinMPNN/`
- `vendors/esm-main/`

## Recommended To Exclude If You Want A Cleaner Public Repo

- `outputs/*_pipeline/`
- `outputs/benchmark_triplet/`
- `outputs/benchmark_sixpack/`
- `outputs/benchmark_ninepack/`
- `outputs/benchmark_twelvepack/`
- `outputs/neural_field_training/`
- `.cache/`
- `vendors/ProteinMPNN_tmp_extract/`
- large local checkpoints or temporary render products

## Minimum Public Figure Set

If you want the GitHub repo to already show the paper direction, keep:

- `outputs/paper_bundle_v1/figures/figure2_benchmark_overview_v2.svg`
- `outputs/paper_bundle_v1/figures/figure3_decoder_calibration_v3.svg`
- `outputs/paper_bundle_v1/figures/figure4_case_1lbt_v1.svg`
- `outputs/paper_bundle_v1/figures/figure5_case_tem1_v1.svg`
- `outputs/paper_bundle_v1/figures/figure6_case_petase_v1.svg`
- `outputs/paper_bundle_v1/figures/figure7_case_cld_v1.svg`

## Suggested Repo Tagline

Use one of these:

1. `Protein engineering with a shared residue decision field`
2. `A retrieval-, evolution-, and ancestry-aware protein design prototype`
3. `Benchmark-driven protein engineering with calibrated decoding`

## Suggested Release Notes Skeleton

### Included

- unified `field_network` engineering stack
- motif-atlas retrieval
- explicit ancestral field support
- pairwise decoder and calibration
- ninepack and twelvepack benchmark support
- publication-oriented paper bundle and figure generation

### Current limitation

- fully neural end-to-end field training remains future work

### Main outputs

- benchmark summaries
- decoder/calibration diagnostics
- case-study composite figures
- PyMOL/PSE structure assets
