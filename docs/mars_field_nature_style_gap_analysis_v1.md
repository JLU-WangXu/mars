# MARS-FIELD Nature-Style Gap Analysis v1

## Why this document exists

The current manuscript and figure set are not yet operating at the density, coherence, and claim-driven structure expected from a top-tier Nature / Nature Biotechnology / Nature Machine Intelligence style methods paper.

This document compares:

- what those papers usually do
- where the current MARS-FIELD draft is thin
- what we must add before the manuscript looks serious

## Reference papers consulted

### 1. AlphaFold (Nature 2021)

- Jumper, J. et al. *Highly accurate protein structure prediction with AlphaFold*. Nature 596, 583-589 (2021).
- URL: https://www.nature.com/articles/s41586-021-03819-2

Why relevant:

- archetypal method paper with a very strong principle figure
- main text is tightly claim-driven
- heavy technical detail is supported by extensive Methods / Supplementary material

### 2. Foldseek (Nature Biotechnology 2024)

- van Kempen, M. et al. *Fast and accurate protein structure search with Foldseek*. Nat Biotechnol 42, 243-246 (2024).
- URL: https://www.nature.com/articles/s41587-023-01773-0

Why relevant:

- computational structural biology tool paper
- extremely compressed main text
- every figure and paragraph carries a concrete benchmark or algorithmic claim

### 3. LASE / ASR representation learning (Nature Machine Intelligence 2024)

- Matthews, D. S. et al. *Leveraging ancestral sequence reconstruction for protein representation learning*. Nat Mach Intell 6, 1542-1555 (2024).
- URL: https://www.nature.com/articles/s42256-024-00935-2

Why relevant:

- directly relevant to our ancestry branch
- useful example of how to turn an evolutionary/representation method into a full machine-learning paper

## What these papers have in common

### 1. The paper is built around 2 to 4 hard claims

These papers do not try to say everything at once.
Instead, they compress the whole manuscript into a few claims, for example:

- a new representation
- a measurable benchmark improvement
- a mechanistic explanation
- an explicit limitation or scope boundary

Current MARS-FIELD problem:

- too much content still behaves like project notes
- the draft often explains the system rather than proving a small number of sharp claims

### 2. Figures are claim-first, not dashboard-first

The strongest papers do not use figures as output summaries.
They use figures to answer explicit questions:

- what is the method?
- does it outperform or stabilize the baseline?
- why does it work?
- where does it fail?

Current MARS-FIELD problem:

- several figures still read like pipeline monitoring dashboards
- too many panels explain activity rather than proving a specific scientific statement

### 3. Methods are not short because the work is small

Top-tier papers often have:

- compressed main text
- but very dense Methods
- with architecture description
- training objective details
- benchmark protocol details
- ablations
- data curation details
- implementation details

Current MARS-FIELD problem:

- our current Methods draft is still too narrative
- it does not yet enumerate the system in a paper-grade technical breakdown
- losses, branches, supervision signals, and benchmark protocol need a much more explicit presentation

### 4. Technical detail is organized, not scattered

In strong papers, technical content is grouped into clear subsections such as:

- data and benchmark construction
- model architecture
- objective functions
- inference and decoding
- calibration / selection
- evaluation protocol

Current MARS-FIELD problem:

- architecture, training, decoder, and benchmark details are still mixed together
- this makes the work feel lighter than it really is

### 5. Every benchmark metric is interpreted

Good method papers do not only state numbers.
They explain what the numbers mean.

Current MARS-FIELD problem:

- we have several important metrics
- but they are not yet integrated into a strong benchmark narrative with enough interpretation

## The biggest current gaps in MARS-FIELD

### Gap A. The paper still looks like a project report instead of a method paper

Symptoms:

- too much system explanation
- not enough formal method decomposition
- insufficiently sharp subsection boundaries

What to add:

- a dedicated Architecture subsection
- a dedicated Objective Functions subsection
- a dedicated Neural Field Decoder subsection
- a dedicated Benchmark Protocol subsection

### Gap B. Methods are under-specified

Right now we need much more explicit detail on:

1. evidence tensor construction
2. shared field construction
3. candidate feature construction
4. memory-bank ancestry and retrieval branches
5. controller heads
6. all training losses
7. teacher-forced neural field decoder
8. hybrid final policy

Without these, the paper reads like a concept note rather than a serious ML/biology method.

### Gap C. Figures 2 and 3 were previously too presentation-like

This is already being improved, but the underlying rule should be explicit:

- no generic KPI cards unless they support a strong benchmark statement
- no redundant plot panels
- no panels whose only function is "showing there is data"

### Gap D. Figure 4 must carry much more of the visual appeal

If the paper wants to look publication-ready, the structure-driven case-study figure has to do more work.

That means:

- high-quality PSE / PyMOL renders
- tighter close-ups
- clearer local mechanism
- less overlaid text

### Gap E. References and related-work positioning are still too thin

The draft currently needs a stronger framing around:

- structure prediction / inverse folding
- protein engineering representation learning
- retrieval-based structural search
- ancestral sequence reconstruction in learning systems

## What a stronger MARS-FIELD manuscript should contain

### Main text

The main paper should be organized around four claims:

1. MARS-FIELD defines a unified evidence-to-sequence residue-field controller.
2. The final controller remains benchmark-stable while introducing decode-time neural generation.
3. Neural decoder contributions are real, selective, and mechanistically interpretable.
4. Remaining failures are concentrated and reveal the next research frontier rather than invalidating the method.

### Methods

The Methods should have at least the following subsections:

1. Benchmark panel and target curation
2. Structural feature extraction and design-window definition
3. Phylogenetic and family-differential priors
4. Ancestral lineage field construction
5. Retrieval motif-atlas and prototype-memory construction
6. Shared residue field representation
7. Candidate controller architecture
8. Training objectives
9. Teacher-forced neural field decoder
10. Final hybrid selection policy
11. Evaluation metrics and paired comparison protocol

### Figures

The main figures should answer:

- Figure 1: what the method is
- Figure 2: why the benchmark claim is credible
- Figure 3: why the method works and where it fails
- Figure 4: how representative targets expose distinct controller regimes

## Concrete action items

The manuscript is not ready to look like a top-tier submission until we do the following:

1. Expand Methods into a real technical methods section with formulas, supervision signals, and protocol detail.
2. Rebuild Figure 1 as a true principle figure rather than a decorative system diagram.
3. Make Figure 4 a stronger PSE-driven visual centerpiece.
4. Add a proper Related Work / References backbone.
5. Write figure legends as if they were already going to production.

## Bottom line

The issue is not that MARS-FIELD lacks enough underlying work.
The issue is that the manuscript and figures still under-represent the actual amount of work done.

In other words:

- the project is deeper than the paper currently looks
- the current task is to convert system depth into paper density
