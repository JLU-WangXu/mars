# MARS-FIELD Figure Legends v3

## Figure 1 | MARS-FIELD integrates heterogeneous evidence into a shared residue field

**a,** Multi-modal evidence streams entering the controller, including geometry-conditioned structural information, phylogenetic context, ancestral lineage priors, retrieval-based motif memory, and environment-conditioned engineering context.  
**b,** Central representation of the method as a shared residue field defined by site-wise residue energies `U(i, a)` and pairwise couplings `C(i, j, a, b)`.  
**c,** Downstream controller-decoder actions on the field. The structured decoder performs constrained field-to-sequence search, whereas the calibrated selector enforces target-wise normalization, prior consistency, and safety gating before final ranking.  
**d,** Example target-level field instantiation showing how residue preferences and pairwise couplings are represented on a concrete design window.

## Figure 2 | The final controller remains benchmark-stable while activating neural decode-time generation

**a,** Paired policy-score differences between the final controller and the incumbent benchmark across the twelve-target panel. Positive values indicate targets whose final policy score increased relative to the incumbent pipeline.  
**b,** Headline benchmark metrics summarizing improved targets, decreased targets, mean paired policy delta, and retained neural-decoder-derived candidates.  
**c,** Neural decoder utilization across the benchmark, decomposed into preview, rejected, and retained novel candidates.  
**d,** Family-level summary showing final controller behavior across the represented protein families and prior regimes.

## Figure 3 | Ablations and neural diagnostics reveal both the drivers and the boundaries of MARS-FIELD

**a,** Component ablation analysis showing sensitivity to oxidation-aware, surface-aware, and evolutionary evidence.  
**b,** Target-specific neural gate profiles across geometry, phylogeny, ancestry, retrieval, and environment branches.  
**c,** Concentrated negative paired-delta targets, illustrating that the remaining failures are localized rather than diffuse.  
**d,** Decoder selectivity map relating preview counts to retained-novel fractions.

## Figure 4 | Representative case studies reveal distinct controller regimes

**a,** `1LBT` as a safety-preserving regime in which the final controller maintains the incumbent solution despite active neural decoding.  
**b,** `TEM1` as a stable-incumbent regime with a neural-decoder-derived learned alternative that remains below the final policy winner.  
**c,** `PETase` as a reproducibility regime across related structural contexts.  
**d,** `CLD` as a calibration stress test in which the incumbent remains stable while the neural branch continues to surface a nearby alternative.
