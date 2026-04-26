# MARS-FIELD Core Claims v2

## Claim 1

`MARS-FIELD` is a unified evidence-to-sequence controller that projects structural, phylogenetic, ancestral, retrieval-based, and environment-conditioned signals into a shared residue field.

Why it matters:

- it defines the method as a field controller rather than a generator vote
- it gives the paper a clear algorithmic object

Figure anchor:

- Figure 1

## Claim 2

The final controller remains benchmark-stable after neural decode-time generation is introduced into the main path.

Why it matters:

- it is the main benchmark claim
- it shows that neuralization does not destabilize the panel

Supported by:

- 12-target panel
- 9/12 targets improved
- 3/12 targets decreased
- mean paired policy delta approximately -0.001

Figure anchor:

- Figure 2

## Claim 3

The neural branch is a real proposal source, not only a reranker, and its contribution is selective rather than indiscriminate.

Why it matters:

- it upgrades the system from rerank-only to controller-decoder
- it supports an end-to-end controller narrative without overclaiming full joint training

Supported by:

- neural decoder enabled on 12/12 targets
- retained neural-decoder candidates on 5/12 targets
- 34 retained novel neural-decoder candidates

Figure anchor:

- Figure 2
- Figure 3

## Claim 4

Controller behavior is mechanistically interpretable and remaining failures are concentrated rather than diffuse.

Why it matters:

- it gives the paper a mechanism-and-boundary layer
- it turns regressions into explicit limitations rather than hidden weaknesses

Supported by:

- oxidation and evolution ablations
- target-specific gate profiles
- concentrated regression targets
- case-study structural panels

Figure anchor:

- Figure 3
- Figure 4
