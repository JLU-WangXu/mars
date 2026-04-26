# MARS-FIELD Figure 1 Blueprint v1

## Figure Goal

This figure is not a workflow chart.

It should communicate the core computational principle of `MARS-FIELD` as a unified evidence-to-sequence network:

- multi-modal evidence streams
- shared residue energy field
- structured decoding
- calibrated engineering-aware selection

The figure should look like a machine-learning systems figure, not a pipeline operations diagram.

## Figure Title

**MARS-FIELD: A Unified Evidence-to-Sequence Network**

## Panel Structure

### Panel A. Multi-Modal Evidence Encoders

Five streams should enter from the left:

1. `Geometric Encoder`
   - backbone graph
   - local geometry
   - protected/design masks

2. `Phylo-Sequence Encoder`
   - homolog MSA
   - family differential priors
   - structure-aware evolutionary weighting

3. `Ancestral Lineage Encoder`
   - ASR posterior
   - ancestor depth
   - uncertainty/confidence

4. `Retrieval Memory Encoder`
   - motif atlas / prototype memory
   - structural motif retrieval
   - residue-support prototypes

5. `Environment Hypernetwork`
   - oxidation
   - low temperature
   - freeze-thaw / dry-rehydration
   - perchlorate / ionic stress

### Panel B. Shared Residue Energy Field

This must be the visual center of the figure.

The field should be represented as:

- one layer for site-wise residue energy `U(i, a)`
- one overlaid layer for pairwise coupling `C(i, j, a, b)`

Desired visual language:

- energy surface
- structured tensor slice
- colored residue field slabs

What to emphasize:

- evidence streams are projected into the same decision space
- the model’s main object is residue energy, not candidate voting

### Panel C. Structured Decoder and Calibration

Right side of the figure should show:

1. structured decoder
   - field-to-sequence decoding
   - constrained beam / energy-guided search

2. calibrated selector
   - target-wise normalization
   - prior consistency
   - safety gating

3. final design outputs
   - ranked variants
   - uncertainty-aware shortlist

## Core Message

The figure must communicate:

**MARS-FIELD does not vote among generators. It constructs a unified residue energy field from geometry, evolution, ancestry, retrieval, and environment-conditioned evidence, then decodes calibrated sequences under engineering constraints.**

## Engineering Mapping To Current Code

This figure should be traceable to real modules in the codebase:

- `marsstack/field_network/encoders.py`
- `marsstack/field_network/residue_field.py`
- `marsstack/energy_head.py`
- `marsstack/decoder.py`
- `marsstack/fusion_ranker.py`
- `marsstack/retrieval_memory.py`
- `marsstack/ancestral_field.py`

## Current Best Visual Cases

Recommended proteins for structural visuals:

1. `1LBT`
   - compact design window
   - clean engineering narrative

2. `TEM-1 / 1BTL`
   - multi-site oxidation mitigation
   - strong benchmark case

3. `CLD / 3Q09`
   - explicit ancestry + oxidation-shell engineering
   - strongest example for evolutionary/ancestral branch

4. `PETase / 5XFY` or `5XH3`
   - useful for family transfer and cutinase narrative

## Recommended Color Logic

Not literal implementation colors, but semantic colors:

- geometry: blue / slate
- phylogeny: green
- ancestry: teal / emerald
- retrieval memory: amber / gold
- environment: orange / coral
- residue field core: multi-layer gradient, no flat blocks
- decoder: magenta / crimson
- selector/calibration: neutral dark gray with highlight accents

## Caption Skeleton

**Figure 1 | MARS-FIELD architecture.**
`MARS-FIELD` maps geometric, phylogenetic, ancestral, retrieval-based, and environment-conditioned evidence into a shared residue energy field. The geometric encoder captures backbone-conditioned structural compatibility; the phylo-sequence encoder captures conservation and family adaptation statistics; the ancestral lineage encoder represents posterior residue preferences and uncertainty from reconstructed lineage states; the retrieval encoder queries a motif atlas of structurally similar local patterns; and the environment hypernetwork conditions the model on stress-relevant engineering objectives. These evidence streams are fused into site-wise residue energies and pairwise coupling energies, which are decoded into constrained sequence designs and calibrated before final ranking.
