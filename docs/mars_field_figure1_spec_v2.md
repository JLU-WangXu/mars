# MARS-FIELD Figure 1 Spec v2

## Goal

This figure is the main principle figure for the paper.

It should read as a machine-learning systems figure, not as an operations flow chart.
The figure must communicate one idea clearly:

**MARS-FIELD maps heterogeneous structural, evolutionary, ancestral, retrieval, and environment-conditioned evidence into a shared residue energy field, then decodes calibrated protein designs under engineering constraints.**

## Final Title

**MARS-FIELD: A Unified Evidence-to-Sequence Network**

## One-Sentence Figure Claim

`MARS-FIELD` does not vote among generators. It builds a shared residue decision space whose native objects are site-wise residue energies `U(i, a)` and pairwise coupling energies `C(i, j, a, b)`.

## Layout

The figure should have three visual sections.

### Section I. Multi-Modal Evidence Encoders

Five encoder blocks enter from the left:

1. `Geometric Encoder`
   - backbone graph
   - local geometry
   - protected/design masks
   - backbone-conditioned compatibility

2. `Phylo-Sequence Encoder`
   - homolog MSA
   - conservation
   - family differential priors
   - structure-aware evolutionary weighting

3. `Ancestral Lineage Encoder`
   - ASR posterior
   - ancestor depth
   - posterior entropy
   - lineage confidence

4. `Retrieval Memory Encoder`
   - motif atlas
   - prototype memory
   - local structural neighbors
   - residue-support prototypes

5. `Environment Hypernetwork`
   - oxidation
   - low temperature
   - freeze-thaw / dry-rehydration
   - perchlorate / ionic stress

### Section II. Shared Residue Energy Field

This is the center of gravity of the figure.

The central object must be shown as two coupled layers:

- site-wise residue energy layer `U(i, a)`
- pairwise coupling layer `C(i, j, a, b)`

Preferred visual language:

- residue field slabs or tensor slices
- multi-layer energy landscape
- pairwise coupling arcs or structured overlays
- no flat "candidate box" depiction

The figure should visually imply that all evidence streams are projected into the same residue decision manifold.

### Section III. Decoding and Calibration

The right side should show:

1. `Structured Decoder`
   - field-to-sequence decoding
   - constrained beam search
   - energy-guided search

2. `Calibrated Selector`
   - target-wise normalization
   - prior consistency
   - safety gating
   - uncertainty-aware shortlist

3. `Final Outputs`
   - ranked designs
   - case-study candidates
   - benchmark tables
   - structure visualization bundles

## Core Math To Surface In The Figure

Use a compact equation block near the center or lower caption area:

```text
E(x) = sum_i U(i, x_i) + sum_(i,j in N) C(i, j, x_i, x_j)
```

Optional small caption line:

```text
Evidence streams parameterize a shared residue energy field rather than a generator vote.
```

## Arrow Language

Use these phrases on arrows or small callouts:

- `geometry-conditioned compatibility`
- `phylogenetic adaptation statistics`
- `ancestral posterior constraints`
- `motif memory retrieval`
- `environment-conditioned modulation`
- `project to shared residue field`
- `decode under engineering constraints`
- `calibrate before final ranking`

## Mapping To Current Code

Every box in the figure should map to real modules:

- `marsstack/field_network/encoders.py`
- `marsstack/evolution.py`
- `marsstack/ancestral_field.py`
- `marsstack/retrieval_memory.py`
- `marsstack/field_network/residue_field.py`
- `marsstack/energy_head.py`
- `marsstack/decoder.py`
- `marsstack/fusion_ranker.py`
- `marsstack/field_network/system.py`

## Implementation Status Callout

The manuscript text should be accurate about what is already implemented.

Implemented now:

- multi-modal evidence bundle construction
- motif-atlas retrieval
- explicit ancestral field objects
- shared residue field construction
- pairwise engineering energy
- structured decoding
- calibrated selection
- unified benchmark and visualization outputs

Still on the neural upgrade path:

- fully learned geometric encoder
- differentiable prototype memory
- stronger environment modulation
- fully learned pairwise head

## Recommended Visual Cases

Use these proteins as inset exemplars or side callouts:

1. `1LBT`
   - compact three-site design window
   - clean oxidation-hardening narrative

2. `TEM1 / 1BTL`
   - robust multi-site engineering example
   - strong benchmark winner stability

3. `PETase / 5XFY` and `5XH3`
   - family-transfer pair
   - cross-structure narrative

4. `CLD / 3Q09`
   - strongest ancestry-aware branch
   - topic / no-topic comparison

## Color Semantics

- geometry: slate / blue
- phylogeny: green
- ancestry: teal
- retrieval: amber / gold
- environment: orange / coral
- field core: layered spectral gradient
- decoder: magenta / crimson
- calibration: charcoal / neutral dark gray

## Suggested Caption

**Figure 1 | MARS-FIELD architecture.** `MARS-FIELD` projects geometric, phylogenetic, ancestral, retrieval-based, and environment-conditioned evidence into a shared residue energy field. The geometric encoder captures backbone-conditioned compatibility, the phylo-sequence encoder captures conservation and family adaptation statistics, the ancestral lineage encoder represents posterior residue preferences and uncertainty from reconstructed lineage states, the retrieval branch queries a motif atlas of structurally similar local patterns, and the environment branch modulates the field according to engineering-relevant stress objectives. These evidence streams parameterize site-wise residue energies and pairwise coupling energies, which are decoded into constrained sequence designs and calibrated before final ranking.
