# MARS-FIELD Manuscript Table Plan v1

## Goal

This document defines the manuscript-ready tables that should be exported from the current codebase.

The tables should support the paper narrative directly and reduce manual spreadsheet work later.

## Main-Text Tables

### Table 1. Twelvepack Benchmark Overview

Purpose:

- summarize the full computational panel
- show overall winner per target
- make the scale of evaluation obvious

Required columns:

- target
- family
- overall source
- overall mutations
- overall score
- overall mars score
- best learned source
- best learned mutations
- best learned score
- decoder novel count
- decoder rejected count
- homolog count
- ASR enabled
- family prior enabled

Primary source:

- `outputs/benchmark_twelvepack/benchmark_summary.csv`

### Table 2. Family-Level Transfer Summary

Purpose:

- support the generalization claim
- summarize held-out family behavior

Required columns:

- family
- number of targets
- mean overall score
- mean best learned score
- number of family-prior targets
- number of ASR-active targets

### Table 3. Decoder and Calibration Summary

Purpose:

- show that decoder injection is controlled rather than naive
- support the engineering-consistency claim

Required columns:

- target
- decoder enabled
- decoder injected
- decoder novel count
- decoder rejected count
- best decoder candidate
- best decoder ranking score
- final overall source
- final overall score

## Supplementary Tables

### Supplementary Table S1. Full Candidate-Level Ranking

One file per case-study target is acceptable.

Expected source:

- `outputs/*_pipeline/combined_ranked_candidates.csv`

### Supplementary Table S2. Visualization Asset Inventory

Purpose:

- keep structure figure preparation reproducible
- track PyMOL scene and palette files

Required columns:

- target
- pipeline directory
- viz manifest path
- scene path
- palette path
- position field path
- pairwise tensor path
- retrieval hits path
- ancestral field path

### Supplementary Table S3. Case-Study Manifest

Purpose:

- freeze which targets belong to which figure
- preserve companion targets and control examples

Required columns:

- figure label
- case id
- primary target
- companion targets
- narrative role
- key output assets

## Export Policy

All paper tables should be generated automatically into a dedicated output bundle.

Recommended output directory:

- `outputs/paper_bundle_v1`

This bundle should contain:

- benchmark overview table
- family summary table
- decoder summary table
- case-study manifest
- asset inventory
- one markdown summary for human review

## Immediate Next Step

Implement one script that reads:

- `outputs/benchmark_twelvepack/benchmark_summary.csv`
- target pipeline output folders

and writes:

- manuscript-ready CSV tables
- figure manifests
- a short markdown summary
