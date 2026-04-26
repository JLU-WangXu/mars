# MARS-FIELD Experiment Plan v1

## 1. Purpose

This document defines the next-stage experiment plan after the current `ninepack -> twelvepack` system stabilization.

Its role is to bridge:

- the codebase
- the benchmark outputs
- the figure set
- the paper narrative

## 2. Current Readiness

The project now has:

- a canonical `field_network` abstraction
- motif-atlas retrieval
- explicit ancestral field objects
- unified field construction
- pairwise decoder support
- calibrated ranking
- a `benchmark_twelvepack` configuration
- structure visualization bundle generation

This means the next experiments should focus on:

- demonstrating system-level consistency
- selecting strong representative case studies
- preparing publication-grade figures

## 3. Benchmarks To Lock

### A. Core Benchmark

Primary computational benchmark:

- `benchmark_twelvepack`

This should become the main result table for the current phase.

### B. Control Benchmark

Retain:

- `benchmark_ninepack`
- `benchmark_twelvepack`
- `v20 decoder_off`
- current unified-path `decoder_on`

These define the before/after computational evidence.

## 4. Primary Evaluation Questions

### Q1. Does MARS-FIELD improve engineering direction?

Readouts:

- overall winner quality
- best learned winner quality
- fraction of targets whose current unified-path score improves over `v20`

### Q2. Does the model remain engineering-consistent?

Readouts:

- whether top winners obey oxidation hardening intuition
- whether unsafe decoder outcomes are suppressed
- whether WT collapse is avoided

### Q3. Does the field help family transfer?

Readouts:

- family-level means
- held-out family summaries
- performance on added `CLD` and `PETase 5XH3` cases

## 5. Figure Set

### Figure 1

MARS-FIELD architecture figure

Source:

- `docs/mars_field_figure1_blueprint_v1.md`

### Figure 2

Benchmark overview

Contents:

- twelvepack target list
- family assignments
- current score summary

### Figure 3

Decoder and calibration analysis

Contents:

- decoder injected vs rejected counts
- engineering-prior consistency
- examples of corrected winner selection

### Figure 4

Case study 1: `1LBT`

Contents:

- structure view
- design positions
- top candidates
- final winner justification

### Figure 5

Case study 2: `TEM1` or `PETase`

Contents:

- benchmark evidence
- field-driven candidate selection
- structural interpretation

### Figure 6

Case study 3: `CLD`

Contents:

- ancestry-aware design branch
- oxidation shell redesign
- topic/no-topic comparison

## 6. Visualization Asset Targets

The following targets should definitely receive `viz_bundle` assets:

- `1LBT`
- `tem1_1btl`
- `petase_5xfy`
- `petase_5xh3`
- `cld_3q09_notopic`
- `cld_3q09_topic`

Secondary tier:

- `t4l_171l`
- `subtilisin_2st1`
- `adk_1s3g`

## 7. Immediate Execution Order

1. review twelvepack benchmark summary
2. select strongest 3 to 4 case-study proteins
3. generate figure-grade structural bundles for those proteins
4. draft Figure 1 architecture figure
5. draft Figure 2 benchmark summary figure

## 8. Current Recommendation

Recommended high-priority case studies:

1. `1LBT`
2. `TEM1`
3. `PETase`
4. `CLD`

Rationale:

- `1LBT`: clean benchmark and oxidation-hardening narrative
- `TEM1`: strong multi-site engineering example
- `PETase`: family-transfer and cutinase generalization
- `CLD`: strongest ancestry/evolution/biology bridge
