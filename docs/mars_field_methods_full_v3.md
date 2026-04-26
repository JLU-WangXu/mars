# MARS-FIELD Methods v3

## Overview

`MARS-FIELD` is a unified evidence-to-sequence controller for protein engineering. The method constructs a shared residue field over a target-specific design window, uses that field to score incumbent candidates, and decodes new candidates under explicit engineering constraints. The current implementation contains three coupled layers:

1. multi-modal evidence construction
2. residue-field construction
3. controller-decoder inference with calibrated final selection

The present system is not yet a fully joint proposal-generator / field / decoder model. However, the benchmark-time main path includes both a neural controller and a decode-time neural field generator.

## Benchmark panel and target curation

The main benchmark contains twelve targets spanning ten protein families. Each target is specified by:

- an input structure
- a chain identifier
- a wild-type sequence
- a local design window
- a protected-position set
- optional homolog, family, template, or ASR inputs

The benchmark includes:

- `1LBT`
- `tem1_1btl`
- `petase_5xfy`
- `petase_5xh3`
- `sfgfp_2b3p`
- `t4l_171l`
- `subtilisin_2st1`
- `adk_1s3g`
- `esterase_7b4q`
- `sod_1y67`
- `CLD_3Q09_NOTOPIC`
- `CLD_3Q09_TOPIC`

Three targets use family priors, two use ancestral priors, and all twelve use template-aware weighting in the final benchmark path. Neural training is leave-one-target-out at the target level.

## Structural feature extraction and design-window definition

For each target, the structure-analysis stage extracts residue-level geometric and engineering features, including:

- solvent-accessible surface area
- local flexibility proxy
- minimum distance to protected positions
- disulfide participation
- glycosylation motif membership

Two structural masks are then derived:

- oxidation hotspots
- flexible surface positions

The design window is read directly from the target configuration, and the resulting target bundle contains the wild-type sequence, design positions, protected positions, residue feature table, oxidation-hotspot set, and flexible-surface set.

## Phylogenetic and family priors

The phylogenetic branch constructs residue priors from homolog or family sequence collections. The current implementation supports:

1. homolog profile priors
2. family differential priors
3. template-aware weighting

For each design position `i`, the homolog profile defines a residue distribution over the amino-acid alphabet. Family differential priors represent residue enrichment or depletion relative to a reference family distribution. Template-aware weighting modulates the importance of these priors according to structural context.

## Ancestral lineage field

The ancestral branch converts ancestral reconstruction outputs into a position-wise lineage field. For each design position, the current representation may include:

- posterior residue distribution
- posterior confidence
- posterior entropy
- recommendation mass
- maximum recommendation support

These signals contribute both explicit residue support in the engineering field and a dedicated ancestry branch in the neural controller.

## Retrieval motif atlas and prototype memory

The retrieval branch encodes local structural memory rather than acting only as a nearest-neighbor annotation tool. For each design position, the retrieval representation may include:

- residue-support distribution
- neighbor count
- top structural similarity
- top retrieval weight
- support sum
- unique supporting target count

These quantities are projected into the hidden space and fused with a learned retrieval-memory bank, allowing the controller to use retrieval as a structured evidence source.

## Shared residue field

The shared residue field is the native decision object of the method. It has two components:

- site-wise residue energies `U(i, a)`
- pairwise couplings `C(i, j, a, b)`

For a sequence `x`, the field energy is:

`E(x) = sum_(i in D) U(i, x_i) + sum_((i,j) in N) C(i, j, x_i, x_j)`

The engineering field is assembled from structural, evolutionary, ancestral, retrieval, and proposal-derived evidence. The neural field is produced by a multi-branch neural model that outputs unary residue logits and pairwise interaction tensors.

## Candidate controller architecture

The neural controller contains five evidence branches:

- geometry
- phylogeny
- ancestry
- retrieval
- environment

Each branch is projected into a shared hidden space. The ancestry and retrieval branches are additionally fused with learned memory banks. The environment branch produces both an environment token and branch-specific modulation parameters.

Branch fusion is implemented with learned gates. If the branch embeddings at site `i` are
`h_geom(i)`, `h_phylo(i)`, `h_asr(i)`, `h_retr(i)`, and `h_env(i)`, then:

`alpha(i) = softmax(G([h_geom(i), h_phylo(i), h_asr(i), h_retr(i), h_env(i)]))`

and the fused site representation is:

`h_i = MLP([alpha_geom(i) h_geom(i), alpha_phylo(i) h_phylo(i), alpha_asr(i) h_asr(i), alpha_retr(i) h_retr(i), alpha_env(i) h_env(i)])`

At the candidate level, the controller combines sequence-conditioned residue-prototype context, candidate-specific evidence features, and pairwise summary features. The candidate embedding feeds three heads:

- selection head
- engineering head
- policy head

## Training objectives

The controller is trained with multiple objectives.

### Selection regression

`L_sel = MSE(s_hat, s_norm)`

### Engineering regression

`L_eng = MSE(e_hat, e_norm)`

where `e_norm` is the standardized engineering score.

### Policy regression

The target policy score is defined as:

`p_target = 0.30 s_norm + 0.70 e_norm`

with supervision:

`L_pol = MSE(p_hat, p_target)`

### Pairwise policy ranking

Pairwise ranking supervision is applied to candidate pairs whose target policy values differ by a margin:

`L_rank = mean softplus(-(p_hat_i - p_hat_j))`

### Decoder-field supervision

The unary field is supervised toward the empirical residue distribution induced by high-quality candidates:

`L_dec = CE(U, q_site)`

### Additional regularization and auxiliary losses

The current implementation also includes:

- recovery loss
- ancestry alignment loss
- retrieval alignment loss
- environment reconstruction loss
- pairwise consistency loss
- winner-guard loss
- non-decoder-guard loss
- simplicity-guard loss
- selector-anchor loss
- gate-entropy and gate-prior regularization

The total objective is a weighted sum of these terms.

## Teacher-forced neural field decoder

The neural field decoder extends the system beyond reranking. For each held-out target, the benchmark-time pipeline:

1. constructs a runtime neural batch from the live target state
2. trains a leave-one-target-out neural field model using the remaining targets
3. produces learned unary and pairwise field outputs
4. combines these outputs with evidence-derived prior fields
5. decodes `neural_decoder` candidates under constrained beam search

This teacher-forced design prevents the learned field from drifting too far from biologically and engineering-wise supported regions while still allowing decode-time novelty.

## Final hybrid selection policy

The final controller uses a hybrid policy. If `x_current` is the incumbent policy candidate and `x_neural` is the neural-policy candidate, then `x_neural` is adopted only if:

1. `engineering_score(x_neural) >= engineering_score(x_current)`
2. `selection_score(x_neural) - selection_score(x_current) >= -tau`

In the current benchmark implementation:

- `tau = 0.10`

This policy keeps the system conservative on hard targets while still permitting neural branch adoption when adequately supported.

## Evaluation protocol

The primary evaluation quantities are:

- policy selection score
- policy engineering score
- paired policy delta relative to the incumbent benchmark
- neural decoder utilization

Neural decoder utilization is quantified by:

- whether the decoder was enabled
- preview candidate count
- retained novel candidate count
- rejected candidate count
- best retained neural-decoder-derived candidate

The benchmark is interpreted as a paired engineering panel rather than a single-endpoint statistical trial. The key quantities are therefore:

- number of improved targets
- number of decreased targets
- whether regressions are concentrated or diffuse
- whether neural decoding contributes retained non-redundant proposals

## Scope

The present implementation supports the claim that `MARS-FIELD` is a unified residue-field controller with an active decode-time neural branch in the main benchmark path. It does not yet support the stronger claim of fully joint proposal-generator / field / decoder optimization.
