# MARS-FIELD Figure Legends v2

## Figure 1 | MARS-FIELD integrates heterogeneous evidence into a shared residue field

**a,** Multi-modal evidence streams entering the controller, including geometry-conditioned structural information, phylogenetic sequence context, ancestral lineage priors, retrieval-based motif memory, and environment-conditioned engineering context.  
**b,** Central representation of the method as a shared residue field whose native objects are site-wise residue energies `U(i, a)` and pairwise coupling energies `C(i, j, a, b)`.  
**c,** Downstream controller-decoder actions on the shared field. The structured decoder performs constrained field-to-sequence search, whereas the calibrated selector normalizes target-level scores and applies prior consistency and safety-gating constraints before final ranking.  
**d,** Example target-level field instantiation illustrating how residue preferences and pairwise couplings are represented on a concrete design window.

## Figure 2 | The final controller remains benchmark-stable while activating neural decode-time generation

**a,** Paired policy-score differences between the final controller and the incumbent benchmark across the twelve-target panel. Positive values indicate targets whose final policy score increased relative to the incumbent pipeline.  
**b,** Headline benchmark metrics summarizing the number of improved targets, the number of decreased targets, the mean paired policy delta, and the total number of retained neural-decoder-derived candidates.  
**c,** Neural decoder utilization across the benchmark, decomposed into preview, rejected, and retained novel candidates for each target.  
**d,** Family-level summary showing final controller performance and family-prior usage across the ten represented protein families.

## Figure 3 | Ablations and neural diagnostics reveal both the drivers and the boundaries of MARS-FIELD

**a,** Component ablation analysis showing the sensitivity of the final system to oxidation-aware, surface-aware, and evolutionary evidence.  
**b,** Target-specific neural gate profiles illustrating how the controller redistributes weight across geometry, phylogeny, ancestry, retrieval, and environment branches.  
**c,** The concentrated set of negative paired-delta targets, showing that the remaining failures are localized rather than broadly distributed across the benchmark.  
**d,** Decoder selectivity map relating preview counts to retained-novel fractions, showing that neural decode-time generation is gated rather than indiscriminate.

## Figure 4 | Representative case studies reveal distinct controller regimes

**a,** `1LBT` as a safety-preserving regime, in which the final controller maintains the incumbent solution despite active neural decode-time generation.  
**b,** `TEM1` as a stable-incumbent regime with a neural-decoder-derived learned alternative that remains below the final policy winner.  
**c,** `PETase` as a reproducibility regime across related structural contexts, showing that the controller can preserve a canonical redesign rather than forcing unnecessary novelty.  
**d,** `CLD` as a calibration stress test in which the incumbent final policy remains stable while the neural branch continues to surface a nearby alternative with stronger local engineering support.
