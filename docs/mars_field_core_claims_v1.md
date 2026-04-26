# MARS-FIELD Core Claims v1

## Why this file exists

The manuscript should not try to make ten small claims.
It should compress the entire paper into a few hard claims that the figures, methods, and benchmarks all reinforce.

## Claim 1

`MARS-FIELD` is a unified evidence-to-sequence controller that maps structural, phylogenetic, ancestral, retrieval-based, and environment-conditioned signals into a shared residue field.

Why this matters:

- this is the conceptual leap beyond stitched-together proposal tools
- this is the basis for Figure 1

Supported by:

- field architecture
- unified residue-field representation
- encoder and memory branches

## Claim 2

The final controller remains benchmark-stable after introducing decode-time neural generation.

Why this matters:

- this is the main benchmark claim
- it tells the reader the system did not collapse when neural complexity increased

Supported by:

- 12-target panel
- 9/12 improved
- 3/12 negative
- mean paired delta approximately -0.001

This is the basis for Figure 2.

## Claim 3

The neural branch is a real proposal source, not only a reranker, and its contribution is selective rather than indiscriminate.

Why this matters:

- it upgrades the method from rerank-only to controller-decoder
- it supports the “end-to-end controller” narrative

Supported by:

- neural decoder enabled on 12/12 targets
- retained neural-decoder candidates on 5/12 targets
- total retained novel neural-decoder candidates = 34

This is supported by Figure 2 and Figure 3.

## Claim 4

The behavior of the controller is mechanistically interpretable and its remaining failures are concentrated rather than diffuse.

Why this matters:

- top-tier methods papers need both mechanism and boundary
- this is where we show the method is serious rather than flashy

Supported by:

- oxidation and evolution ablations
- neural gate composition
- concentrated limitation cases
- case-study structural panels

This is the basis for Figure 3 and Figure 4.

## Mapping claims to figures

- Figure 1 -> Claim 1
- Figure 2 -> Claim 2 and Claim 3
- Figure 3 -> Claim 4
- Figure 4 -> Claim 4 in concrete target-level form
