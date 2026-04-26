# Held-Out Family Split v0

## Goal

Prevent the method story from collapsing into a collection of protein-specific case studies.

The benchmark should report:

- in-family aggregate performance
- held-out-family performance
- proposal-source-stratified results (`best learned` vs `overall`)

## Current family assignments

- `lipase_b`: `CALB / 1LBT`
- `beta_lactamase`: `TEM-1 / 1BTL`
- `cutinase`: `PETase / 5XFY`
- `gfp_like`: `sfGFP / 2B3P`
- `lysozyme`: `T4 lysozyme / 171L`
- `subtilisin`: `subtilisin / 2ST1`

## Expansion targets already present in local inputs

- `adenylate_kinase`: `1S3G`
- `lipase_esterase`: `7B4Q`
- `superoxide_dismutase`: `1Y67`

These three are the fastest path from six proteins to nine proteins.

## Recommended split rule

Use leave-one-family-out reporting:

1. build benchmark tables on the full panel
2. group targets by family
3. for each family, report aggregate performance on all remaining families as context and the held-out family as the main transfer result

## Immediate next implementation step

- add benchmark configs for `1S3G`, `7B4Q`, and `1Y67`
- keep their family labels explicit in config metadata
- export a `family_summary.csv` from the benchmark runner
