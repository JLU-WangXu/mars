# Mars Benchmark Standard

## Purpose

This benchmark package is meant to support two different but connected jobs:

1. learn adaptation priors from naturally stress-tolerant proteins
2. test whether common proteins can be engineered toward Mars-relevant stress tolerance without losing core function

The benchmark therefore has two curated tables plus one shared assay panel:

- `inputs/benchmark/reference_dataset.csv`
- `inputs/benchmark/transfer_dataset.csv`
- `inputs/benchmark/stress_panel.csv`

## Dataset Layers

### 1. Reference set

The reference set is for learning and interpretation.

Each row should represent a naturally occurring protein with:

- an experimental structure
- a recoverable canonical sequence from UniProt or the RCSB reference chain
- a clear biological role
- a known adaptation axis such as psychrophily, radioresistance, oxidative resilience, or perchlorate relevance

This layer is not mainly about "winning" a design contest. It is where we learn reusable sequence and structure patterns.

### 2. Transfer set

The transfer set is for engineering.

Each row should represent a practical scaffold with:

- low assay cost
- common expression systems
- simple functional readout
- good structural coverage
- useful historical mutational or engineering literature

This layer is where MarsStack should be judged prospectively or retrospectively.

## Inclusion Rules

Only include proteins that satisfy most of the following:

- experimental structure available by X-ray, cryo-EM, or NMR
- sequence traceable to UniProt or the deposited experimental structure
- assay readout available in a low-cost standard format
- protein family is broad enough to support transfer or comparative analysis
- at least one clear Mars-relevant stress axis can be defined

Preferred extras:

- homologous cold or heat adapted relatives
- published mutational scan, stability panel, or engineering history
- simple soluble expression
- direct residual-activity measurements after stress

## Benchmark Tasks

We standardize five task groups.

### A. Cold adaptation

Goal:
retain activity at `4 C` and `10 C` while preserving standard-condition function.

### B. Oxidative and radiation resilience

Goal:
retain function after `H2O2`, `UV-C`, or ionizing radiation exposure.

### C. Freeze-thaw and dry-rehydration resilience

Goal:
recover function after physical stress expected in storage or transport.

### D. Perchlorate and salt tolerance

Goal:
maintain function in `Mg/Na perchlorate` or related high-ionic-strength conditions.

### E. Baseline function retention

Goal:
avoid trading all stress gains for loss of expression or native activity.

This is a hard gate, not a cosmetic metric.

### F. Low-shear / microgravity proxy resilience

Goal:
retain function and low-aggregation behavior after a reduced-convection or low-shear exposure.

This is an optional extension task, not part of the mandatory core package.

## Phase-1 Core Collection

The current "enough to start and good enough to scale" core is:

### Reference core

- adenylate kinase triad: psychrophile, mesophile, thermophile
- psychrophilic malate dehydrogenase
- cold-active citrate synthase
- `D. radiodurans` RecA
- `D. radiodurans` DdrB
- `D. radiodurans` DdrC
- `D. radiodurans` Mn-SOD
- chlorite dismutase as the Mars perchlorate module anchor

### Transfer core

- CALB
- TEM-1 beta-lactamase
- avGFP
- T4 lysozyme
- subtilisin
- alpha-amylase
- alcohol dehydrogenase
- chlorite dismutase as a Mars-specific extension target

## Split Rules

Do not split train and test by random mutations from the same family.

Preferred splits:

- by protein family
- by organism lineage within family
- by assay generation or publication group for leak checks

## Scoring Rules

### Hard gates for transfer work

Any engineered variant should normally satisfy both:

- soluble expression or usable purified yield `>= 0.30 x WT`
- baseline activity at the standard assay condition `>= 0.50 x WT`

Variants below these gates can still be informative for mechanistic analysis, but they should not count as practical Mars-hardened wins.

### Recommended transfer score

For general enzymes, use:

- gate: `EXP01`
- gate: `BASE01`
- weighted panel: `LT04 + FT03 + OXH01 + PER01`
- optional add-on: `LYO01` or `MGR01`
- radiation-focused swap: replace `PER01` with `UV01` or `RAD01`

If the claim is explicitly about low-shear or microgravity readiness, append `MGR01` as an optional add-on and renormalize the weighted score rather than promoting it to a universal core term.

If all weighted assays are normalized to WT=`1.0`, a practical first-pass score is:

`MarsTransferScore = 0.35*BASE01 + 0.25*LT04 + 0.15*FT03 + 0.15*OXH01 + 0.10*PER01`

For reporter proteins, swap `BASE01` from enzymatic activity to fluorescence or signal output at the standard condition.

### Reference analysis outputs

The reference set is not judged by the same scalar score.

Preferred outputs are:

- family-wise feature deltas
- mutation enrichment maps
- structure feature distributions
- position-level transfer priors
- assay-matched comparative plots

## Column Guidance

### `reference_dataset.csv`

Key columns:

- `benchmark_id`: stable short identifier
- `priority_tier`: `core` or `expansion`
- `task_group`: main biological use
- `protein_family`: family-level grouping for split control
- `representative_pdb`: anchor experimental structure
- `sequence_source`: `UniProt reviewed`, `RCSB reference sequence`, or `local validated sequence`
- `mutational_data_class`: `homolog-comparative`, `scan-rich`, `engineering-rich`, or `sparse`

### `transfer_dataset.csv`

Key columns:

- `design_goal`: what MarsStack should improve
- `assay_anchor`: cheapest robust assay for that scaffold
- `mandatory_panels`: default required stress panels
- `optional_panels`: useful expansion assays
- `status`: `active_seed`, `active_extension`, or `backlog`

## Practical Default Conditions

Use the same WT-normalized readouts across proteins whenever possible.

Recommended defaults:

- standard temperature: `25 C`, unless the community-standard assay temperature is strongly entrenched
- low temperature panel: `4 C` first, `10 C` optional
- freeze-thaw: `3` cycles minimum
- dry-rehydration: one lyophilization cycle minimum
- microgravity proxy: `24-72 h` low-shear clinostat or rotating-wall-vessel preincubation plus a matched control
- oxidative stress: `1 to 5 mM H2O2` pilot-calibrated per scaffold
- UV panel: dose chosen so WT is stressed but not destroyed
- perchlorate panel: start with sublethal `NaClO4` or `Mg(ClO4)2` titration rather than a single extreme dose

## Immediate Use

This package is designed for immediate use in `MarsStack`:

- the reference set defines what we learn from
- the transfer set defines what we engineer on
- the stress panel defines how results stay comparable across projects

If we keep adding rows without changing these three contracts, the benchmark can grow without needing a redesign.
