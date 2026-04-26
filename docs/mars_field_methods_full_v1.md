# MARS-FIELD Methods (Full Draft) v1

## Overview

`MARS-FIELD` is formulated as a unified evidence-to-sequence controller for protein engineering. Rather than combining independent proposal engines and hand-tuned post hoc heuristics, the method projects heterogeneous evidence streams into a shared residue field over the design positions, then uses a neural controller and a constrained decoder to rank incumbent candidates and generate new ones under engineering constraints.

At the current stage, the method has three interacting layers:

1. multi-modal evidence construction
2. shared residue-field construction
3. controller-decoder inference and calibrated selection

The present implementation is not yet a fully jointly optimized proposal-generator / field / decoder system. However, the benchmark-time main path now includes both a learned neural controller and a decode-time neural field generator, making the current system meaningfully closer to an end-to-end controller-decoder architecture than to a simple candidate reranker.

## Benchmark panel and target curation

The main benchmark consists of twelve targets spanning ten protein families. Each target is defined by a local design window, a fixed wild-type structure, a wild-type sequence, a set of protected positions, and optional evolutionary, family, ancestral, or environmental prior files. The targets are instantiated through per-target YAML configuration files, and benchmark execution is driven by benchmark-level YAML manifests.

The benchmark panel includes:

- `1LBT` (`lipase_b`)
- `tem1_1btl` (`beta_lactamase`)
- `petase_5xfy` and `petase_5xh3` (`cutinase`)
- `sfgfp_2b3p` (`gfp_like`)
- `t4l_171l` (`lysozyme`)
- `subtilisin_2st1` (`subtilisin`)
- `adk_1s3g` (`adenylate_kinase`)
- `esterase_7b4q` (`lipase_esterase`)
- `sod_1y67` (`superoxide_dismutase`)
- `CLD_3Q09_NOTOPIC` and `CLD_3Q09_TOPIC` (`cld`)

Three targets use family priors (`adk_1s3g`, `esterase_7b4q`, `sod_1y67`), two use ancestral priors (`CLD_3Q09_NOTOPIC`, `CLD_3Q09_TOPIC`), and template-aware weighting is enabled for all twelve targets in the final benchmark path.

All neural training used in the benchmark is leave-one-target-out at the target level. For a held-out target `t`, the neural controller and neural field generator are trained using the remaining benchmark targets and are then applied to `t`.

## Structural feature extraction and design-window definition

Each target begins with a structure-processing step that extracts per-residue geometric and engineering features from the input structure. These include:

- solvent-accessible surface area
- mean B-factor or equivalent local flexibility proxy
- minimum distance to protected positions
- disulfide participation
- glycosylation motif membership

Two engineering-relevant structural masks are then derived:

- oxidation hotspots
- flexible surface positions

Oxidation hotspots are detected using structure-derived exposure and protected-distance thresholds together with residue-type sensitivity. Flexible surface positions are detected using exposure-driven heuristics, and are subsequently merged with explicitly designable positions when required by the target configuration.

The design window is specified as the set of mutable positions declared in the target configuration. The final structure bundle therefore contains:

- wild-type sequence
- design positions
- protected positions
- structural feature table
- oxidation-hotspot set
- flexible-surface set

## Phylogenetic and family priors

The phylogenetic branch uses homolog or family-aligned sequence sets to construct residue-level priors over the design positions. The current implementation supports:

- homolog profile priors
- family differential priors
- template-aware weighting

For each design position `i`, the homolog profile defines a residue distribution over the amino-acid alphabet. Family differential priors provide branch- or family-specific enrichment relative to a reference sequence population. Template-aware weighting modulates the importance of evolutionary support according to structural context, allowing the system to reweight profile evidence when local structural conditions justify stronger or weaker prior influence.

These components are represented both as explicit evidence terms in the engineering field and as inputs to the neural controller. In the current neural batch representation, evolutionary inputs include residue-level profile mass and family-differential support encoded as a per-position vector.

## Ancestral lineage field

The ancestral lineage branch converts ancestral sequence reconstruction outputs into a position-wise ancestral field. For each design position `i`, the ancestral representation may include:

- posterior residue distribution
- posterior confidence
- posterior entropy
- recommendation mass
- maximum recommendation support

This branch is important because it injects lineage-derived probability structure into the residue field rather than treating ASR as a one-off hard prior. In the engineering field, ancestry contributes position-wise residue recommendations. In the neural controller, ancestry is represented as a dedicated branch whose latent site embeddings are fused with a learned lineage-memory bank.

Formally, if `p_anc(i, a)` is the ancestral posterior over amino acid `a` at site `i`, then ancestry contributes to both:

- explicit residue support in the engineering field
- latent ancestry branch input in the neural field

The current benchmark includes both ASR-free and ASR-enabled targets, allowing us to evaluate whether ancestry helps selectively rather than uniformly.

## Retrieval motif atlas and prototype memory

The retrieval branch represents a structure-derived local memory rather than a simple nearest-neighbor annotation table. In the engineering field, retrieval contributes residue-level support via structure-local motif memory and local neighbor support. In the neural controller, retrieval is represented as a dedicated latent branch with a learned retrieval-memory bank.

The current retrieval input representation includes:

- residue-support distribution from retrieval neighbors
- neighborhood count
- top structural similarity
- top retrieval weight
- support sum
- unique supporting target count

These quantities are encoded as a per-position retrieval vector and then passed through a retrieval projection layer, followed by a learned prototype-memory fusion step. The purpose of this design is to avoid treating retrieval as a fixed external rule. Instead, the controller can decide when retrieval support should dominate, when it should be moderated, and when it should be ignored.

## Shared residue field

The conceptual core of `MARS-FIELD` is the shared residue field. The field has two coupled components:

- a site-wise residue-energy layer `U(i, a)`
- a pairwise coupling layer `C(i, j, a, b)`

For a candidate sequence `x`, the field energy is written as:

`E(x) = sum_i U(i, x_i) + sum_(i,j in N) C(i, j, x_i, x_j)`

where `N` denotes the set of design-position pairs for which a coupling term is defined.

In the engineering field, `U(i, a)` is constructed from the unified evidence fields produced by structural, evolutionary, ancestral, retrieval, and proposal-derived evidence. Pairwise terms are constructed from proposal-derived pairwise support over top-ranked candidates and filtered by structural proximity.

In the neural field, `U(i, a)` is produced by a learned site encoder and residue-prototype scoring head, while `C(i, j, a, b)` is produced by a low-rank pairwise head conditioned on latent site embeddings and pair features.

## Candidate controller architecture

The neural controller contains five evidence branches:

- geometry branch
- phylogeny branch
- ancestry branch
- retrieval branch
- environment branch

Each branch is first projected into a shared hidden dimension. The ancestry and retrieval branches are then each fused with a learned memory bank:

- lineage memory for ancestral signals
- retrieval memory for motif-atlas and neighborhood signals

The environment branch produces both an environment token and branch-specific modulation parameters. These parameters multiplicatively and additively modulate the geometry, phylogeny, ancestry, and retrieval branch embeddings before branch fusion.

Branch fusion is implemented through a learned gating mechanism. Let the branch embeddings be:

- `h_geom(i)`
- `h_phylo(i)`
- `h_asr(i)`
- `h_retr(i)`
- `h_env(i)`

Then the controller computes branch weights:

`alpha(i) = softmax(G([h_geom(i), h_phylo(i), h_asr(i), h_retr(i), h_env(i)]))`

and forms a fused site representation:

`h_i = MLP([alpha_geom(i) * h_geom(i), alpha_phylo(i) * h_phylo(i), alpha_asr(i) * h_asr(i), alpha_retr(i) * h_retr(i), alpha_env(i) * h_env(i)])`

This site representation drives:

- residue-wise unary logits
- auxiliary ancestry alignment logits
- auxiliary retrieval alignment logits
- environment reconstruction head

At the candidate level, the controller builds a candidate embedding by combining:

- residue-prototype-conditioned sequence context
- candidate-specific evidence features
- pairwise summary features

Candidate-specific features include:

- source identity
- source-group identity
- support-count context
- mutation-count context
- engineering component scores
- fusion-ranker contribution terms
- selector-prior features
- gap-to-best features
- note-derived flags

The final candidate embedding is passed to three heads:

- selection head
- engineering head
- policy head

## Training objectives

The training objective is multi-term and combines direct supervision, pairwise supervision, calibration supervision, and auxiliary reconstruction losses.

### 1. Selection regression

The controller predicts target-wise normalized selection scores:

`L_sel = MSE(s_hat, s_norm)`

where `s_hat` is the predicted selection score and `s_norm` is the standardized target-wise selection score.

### 2. Engineering regression

The controller predicts standardized engineering scores:

`L_eng = MSE(e_hat, e_norm)`

where `e_norm` is the standardized `mars_score`.

### 3. Policy regression

A target policy score is constructed as:

`p_target = 0.30 * s_norm + 0.70 * e_norm`

and supervised using:

`L_pol = MSE(p_hat, p_target)`

### 4. Pairwise policy ranking

To stabilize ranking behavior, pairwise margin-style ranking supervision is applied:

`L_rank = mean softplus(-(p_hat_i - p_hat_j))`

over target pairs `(i, j)` whose target policy scores differ by at least a margin.

### 5. Decoder-field residue supervision

The unary field is also supervised toward the empirical residue distribution induced by high-quality candidate sequences:

`L_dec = CE(U, q_residue)`

where `q_residue` is an empirical site-wise residue distribution weighted by candidate-level policy support.

This term is critical because it pushes the field itself, not only the candidate heads, toward realistic decode-time residue preferences.

### 6. Recovery loss

The unary field is regularized toward wild-type recovery:

`L_rec = CE(U, x_wt)`

### 7. Ancestry and retrieval alignment losses

Auxiliary ancestry and retrieval heads are trained against the ancestry posterior and retrieval distribution, respectively:

- `L_asr`
- `L_retr`

### 8. Environment reconstruction loss

The global environment representation is trained to reconstruct the target environment vector:

`L_env = MSE(env_hat, env_target)`

### 9. Pairwise consistency loss

If empirical pairwise residue support is available from candidate tables, the pairwise tensor is regularized toward this empirical distribution:

`L_pair = MSE(C_hat, C_empirical)`

### 10. Conservative guard losses

Several controller-calibration losses are used to prevent unstable winner promotion:

- winner-guard loss
- non-decoder-guard loss
- simplicity-guard loss
- selector-anchor distillation loss

These losses serve different purposes:

- suppress obviously worse alternatives
- prevent decoder branches from overwhelming better incumbent engineering priors
- avoid needless mutation burden
- preserve the incumbent selector’s stability on hard targets

### 11. Gate regularization

The mean branch gate vector is regularized by:

- gate-entropy regularization
- gate-prior regularization

The gate prior depends on the presence or absence of family, template, and ASR support, encouraging the controller to shift attention according to the available biological context.

### Total loss

The total loss is a weighted sum:

`L_total = λ_reg L_reg + λ_sel L_sel + λ_eng L_eng + λ_pol L_pol + λ_rank L_rank + λ_dec L_dec + λ_guard L_guard + λ_rec L_rec + λ_pair L_pair + λ_asr L_asr + λ_retr L_retr + λ_env L_env + λ_gate L_gate`

with fixed coefficients defined in the implementation.

## Teacher-forced neural field decoder

The neural field decoder extends the controller from reranking to decode-time proposal generation.

For a held-out target, the pipeline:

1. builds a runtime neural batch from the live target state
2. trains a leave-one-target-out neural field model using all other targets
3. generates neural unary and pairwise field outputs
4. combines these learned outputs with evidence-derived prior fields
5. decodes `neural_decoder` candidates under constrained beam search

The critical design choice is that the neural field is not decoded in isolation. Instead:

- learned unary scores are mixed with evidence-derived prior residue scores
- learned pairwise scores are mixed with prior pairwise support

This teacher-forced design is deliberate. It allows the decoder to benefit from learned structure while remaining anchored to the broader evidence regime, thereby reducing unstable or biologically implausible decode-time behavior.

The decoder produces:

- preview neural-decoder candidates
- retained novel candidates after engineering and safety filtering
- per-target neural field runtime summaries

## Final hybrid selection policy

The final paper-facing controller uses a hybrid policy.

If `x_current` is the incumbent current-policy candidate and `x_neural` is the neural-policy candidate, then the final controller adopts `x_neural` only if:

1. its engineering score is at least as strong as the incumbent
2. its selection score does not fall more than a small tolerance below the incumbent

In the current implementation, this second tolerance is a small negative margin rather than a strict non-decrease constraint. This design reflects the current strongest honest deployment setting:

- neural rerank is active
- neural decoder is active
- but the controller remains conservative on hard targets where full neural replacement would still be risky

## Evaluation protocol

The primary evaluation quantities are:

- policy selection score
- policy engineering score (`mars_score`)
- paired delta relative to the incumbent benchmark
- neural decoder utilization

Neural decoder utilization is explicitly accounted for using:

- whether the decoder was enabled
- number of preview candidates
- number of retained novel candidates
- number of rejected candidates
- best retained neural-decoder candidate

The benchmark should be interpreted as a paired engineering panel rather than a classical single-endpoint statistical trial. Accordingly, the key quantities for interpretation are:

- how many targets improve
- how many targets regress
- whether regressions are concentrated or diffuse
- whether the neural decoder contributes retained non-redundant candidates
- whether related structures retain chemically consistent winners

## What this Methods section allows us to claim

With the present implementation, the strongest accurate claim is:

`MARS-FIELD` is a unified evidence-to-sequence residue-field controller with an active decode-time neural field branch in the main benchmark path.

The strongest claim it does **not yet** support is:

- fully joint proposal-generator / field / decoder optimization

That distinction should remain explicit in the final manuscript.
