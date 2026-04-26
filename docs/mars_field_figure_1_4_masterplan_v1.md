# MARS-FIELD Figure 1-4 Masterplan v1

This document replaces the previous figure logic with a more publication-oriented narrative structure.

The main problem with the earlier figure set was not only visual polish. It was that the figures behaved more like pipeline reports than like a high-level scientific argument. The new figure logic should therefore be organized around claims rather than modules.

## Global principles

The four main-text figures should jointly answer four scientific questions:

1. What is the method?
2. Does it improve engineering decisions across a broad benchmark?
3. Is the neural branch genuinely active and mechanistically meaningful?
4. Do representative targets reveal interpretable controller behavior rather than arbitrary winner changes?

The figures should not look like:

- software workflow diagrams
- generic dashboard panels
- collections of weak scatterplots
- method-vote cartoons

They should look like:

- concise claim-driven scientific composites
- integrated summary figures
- targeted mechanistic case studies

## Figure 1

### Figure title

`MARS-FIELD integrates heterogeneous evidence into a shared residue field`

### Scientific claim

`MARS-FIELD` is not a vote among generators. It is a shared residue-field controller that absorbs geometry, phylogeny, ancestry, retrieval memory, and environment context into a single residue and pairwise decision space.

### Panel design

Panel A:

- high-level architecture schematic
- evidence streams on the left
- shared field object in the center
- controller and decoder on the right
- compact equation block:
  - `E(x) = sum_i U(i, x_i) + sum_(i,j) C(i, j, x_i, x_j)`

Panel B:

- field-object zoom
- show site-wise residue energies and pairwise couplings as the real native objects of the method
- emphasize that decoder and selector operate on this object

Panel C:

- mapping from evidence streams to actual code / modules
- very compact, almost legend-like
- should reassure readers that the architecture is real rather than rhetorical

Panel D:

- one concise target-level micro-example
- for example 1LBT position field and pairwise edge
- enough to make the field object concrete

### Visual direction

- this figure should be mostly schematic, but scientifically dense
- avoid toy 3D blocks without meaning
- use restrained color semantics by evidence stream

## Figure 2

### Figure title

`The final MARS-FIELD controller remains benchmark-stable while activating neural decode-time generation`

### Scientific claim

The main controller is broad-panel competitive with the incumbent system, while the neural decoder is active and contributes retained novel candidates.

### Panel design

Panel A:

- paired delta strip or lollipop plot
- one row per target
- x-axis: paired policy delta vs incumbent
- color by sign
- direct target labels
- this should become the benchmark headline panel

Panel B:

- compact bar summary:
  - positive targets
  - negative targets
  - mean paired delta
  - neural decoder enabled count
  - retained neural-decoder target count

Panel C:

- family-level summary heatmap or dot matrix
- rows: family
- columns:
  - mean final score
  - best learned score
  - family prior active
  - ASR prior active

Panel D:

- neural decoder utilization panel
- preview / retained / rejected counts by target
- retained targets highlighted
- this should visually prove that the decoder is doing work rather than sitting idle

### Tables linked to Figure 2

- benchmark summary
- family summary
- neural decoder utilization table

## Figure 3

### Figure title

`Ablations and neural diagnostics reveal what drives controller behavior`

### Scientific claim

The method is not a black-box neural add-on. Oxidation and evolution remain dominant constraints, while the neural branch has measurable and interpretable contributions.

### Panel design

Panel A:

- ablation sensitivity summary
- three perturbations:
  - no oxidation
  - no surface
  - no evolution
- report:
  - number of changed top candidates
  - mean score effect

Panel B:

- neural gate composition across targets
- stacked bar or heatmap:
  - geometry
  - phylogeny
  - ancestry
  - retrieval
  - environment

Panel C:

- neural decoder retention map
- retained vs rejected candidates by target
- should reveal the selective, target-dependent nature of decode-time generation

Panel D:

- failure / limitation mini-panel
- show the 3 regression targets:
  - CLD_3Q09_NOTOPIC
  - CLD_3Q09_TOPIC
  - subtilisin_2st1
- explain whether the issue is:
  - over-exploration
  - score compression
  - aggressive learned alternative

### Purpose

This figure is crucial for making the paper look serious. Nature-level papers do not only show successes; they show mechanism and limits.

## Figure 4

### Figure title

`Representative case studies illustrate distinct controller regimes`

### Scientific claim

Different targets reveal different operating modes of the unified controller:

- conservative safety preservation
- stable incumbent plus learned alternative
- reproducible canonical redesign
- calibration stress test

### Panel design

Panel A: 1LBT

- emphasize preservation of `M298L`
- neural decoder active but filtered
- message:
  - stable safety controller

Panel B: TEM1

- emphasize incumbent retained, neural-decoder learned alternative surfaced
- message:
  - stable top solution with useful neural branch exploration

Panel C: PETase

- use one compact two-structure comparison
- message:
  - reproducibility across related scaffolds

Panel D: CLD

- topic vs no-topic
- incumbent vs neural alternative
- message:
  - stress test for calibration and evidence-conditioned tradeoff

### Structure images

- close-ups should be tighter than in previous drafts
- only show residues that support the claim
- no decorative extra angles
- use cleaner residue labels and less clutter

## What should move to Supplementary

The following should not crowd the main figures:

- overly detailed branch diagnostics
- full per-target scatter dashboards
- duplicate benchmark tables
- visually weak acceptance/rejection plots without mechanism

These can move to Supplementary:

- secondary neural comparison plots
- full decoder diagnostics
- full held-out family tables
- exhaustive asset inventories

## Immediate implementation recommendation

The next concrete rendering pass should prioritize:

1. rebuild Figure 2 as a real benchmark claim figure
2. rebuild Figure 3 as a mechanism + limitation figure
3. keep Figure 1 mostly schematic but scientifically denser
4. compress Figure 4 into four strong case-study panels rather than many small report-style snippets
