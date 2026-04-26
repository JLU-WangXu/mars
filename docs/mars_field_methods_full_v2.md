# MARS-FIELD Methods (Full Draft) v2

## Method overview

`MARS-FIELD` is formulated as a unified evidence-to-sequence controller for protein engineering. Rather than combining several proposal engines followed by manual or heuristic post hoc filtering, the method projects heterogeneous evidence streams into a shared residue field over a target-specific design window and then uses a neural controller-decoder stack to rank incumbent candidates and decode new ones under engineering constraints.

At the current stage, the method consists of three coupled layers:

1. multi-modal evidence construction
2. shared residue-field parameterization
3. controller-decoder inference with calibrated final selection

The current implementation is not yet a fully jointly optimized proposal-generator / field / decoder model. However, the benchmark-time main path now includes both a learned neural controller and a decode-time neural field generator, making the system materially closer to an end-to-end controller-decoder architecture than to a pure reranking layer.

## Notation

For a target protein with design positions `i in D` and amino-acid alphabet `A`, the method parameterizes a shared residue field with:

- site-wise residue energies `U(i, a)`
- pairwise coupling energies `C(i, j, a, b)`

For a sequence `x`, the field energy is:

`E(x) = sum_(i in D) U(i, x_i) + sum_((i,j) in N) C(i, j, x_i, x_j)`

where `N` is the set of design-position pairs for which a coupling term is defined.

This field is used both for candidate-level evaluation and for decode-time neural proposal generation.

## Benchmark panel and target curation

The main evaluation panel contains twelve targets spanning ten protein families. Each target is defined by:

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

Three targets use family priors (`adk_1s3g`, `esterase_7b4q`, `sod_1y67`), two use ancestral priors (`CLD_3Q09_NOTOPIC`, `CLD_3Q09_TOPIC`), and template-aware weighting is enabled for all twelve targets in the final benchmark path.

Neural controller training is leave-one-target-out at the target level. For a held-out target `t`, the neural model is trained using the remaining benchmark targets and is then evaluated on `t`.

## Structural feature extraction and design-window definition

Each target is first processed by a structure-analysis stage that extracts residue-level geometric and engineering features. The current feature set includes:

- solvent-accessible surface area
- mean B-factor or equivalent local flexibility proxy
- minimum distance to protected positions
- disulfide participation
- glycosylation motif membership

From these residue-level features, two target-specific structural masks are derived:

- oxidation hotspots
- flexible surface positions

Oxidation hotspots are identified using structure-exposure and protected-distance thresholds in combination with residue-type sensitivity. Flexible surface positions are defined using exposure-driven heuristics and are merged with the design positions when the target configuration indicates that exposed adaptability is desirable.

The design window is specified explicitly in the target YAML configuration. The resulting target bundle therefore contains:

- the wild-type sequence
- the design positions
- the protected positions
- the residue feature table
- the oxidation-hotspot set
- the flexible-surface set

## Phylogenetic and family priors

The phylogenetic branch constructs per-position residue priors from homolog or family sequence collections. The current implementation supports three evolution-aware sources:

1. homolog profile prior
2. family differential prior
3. template-aware evolutionary weighting

For each design position `i`, the homolog profile defines a residue distribution over the amino-acid alphabet. Family differential priors capture enrichment or depletion relative to a reference family distribution. Template-aware weighting modulates evolutionary support according to structural context, allowing profile-derived evidence to be amplified or downweighted depending on local backbone or environment conditions.

In the engineering field, these components contribute directly to residue support. In the neural controller, they are encoded into a phylogenetic feature vector that participates in the branch-fused site representation.

## Ancestral lineage field

The ancestral branch transforms ancestral sequence reconstruction outputs into a position-wise lineage field. For each design position `i`, the current ancestral representation may include:

- posterior residue distribution
- posterior confidence
- posterior entropy
- recommendation mass
- maximum recommendation support

This branch is important because ancestry is not treated as a one-time hard filter. Instead, ancestral information enters the controller as both:

- explicit residue-level support in the engineering field
- a dedicated latent ancestry branch in the neural controller

In the current neural parameterization, the ancestry branch input dimension is 24:

- 20 posterior amino-acid values
- confidence
- entropy
- recommendation mass
- maximum recommendation support

## Retrieval motif atlas and prototype memory

The retrieval branch encodes local structural memory rather than acting as a stand-alone nearest-neighbor annotation tool. In the engineering field, retrieval contributes residue-level support via retrieved motif analogues and neighborhood evidence. In the neural controller, retrieval becomes a learned branch linked to a dedicated prototype-memory bank.

For each design position, the current retrieval representation includes:

- residue-support distribution
- neighbor count
- top structural similarity
- top retrieval weight
- support sum
- unique supporting target count

This yields a 25-dimensional retrieval input:

- 20 residue-support values
- 5 neighborhood diagnostics

These features are projected into the hidden space and fused with a learned retrieval-memory bank, allowing the controller to decide when retrieval evidence should dominate and when it should be moderated by other branches.

## Shared residue field

The conceptual core of `MARS-FIELD` is the shared residue field. The field has two layers:

- a site-wise residue-energy layer `U(i, a)`
- a pairwise coupling layer `C(i, j, a, b)`

### Engineering field

In the engineering field, position-wise residue support is constructed from:

- structural hotspot signals
- flexible-surface signals
- homolog profile priors
- family differential priors
- ancestral recommendations
- retrieval memory
- proposal-derived residue evidence

Pairwise energies are then constructed from proposal-derived co-occurrence patterns over high-ranking candidate sets, filtered by structural proximity.

### Neural field

In the neural field, the residue field is produced from a multi-branch neural model with:

- geometry branch
- phylogeny branch
- ancestry branch
- retrieval branch
- environment branch

The current neural model uses:

- hidden dimension = 64
- pair rank = 16
- memory slots = 12
- branch dropout probability = 0.10

The model outputs:

- unary residue logits
- pairwise interaction tensors
- site hidden states
- branch gates
- ancestry alignment logits
- retrieval alignment logits
- environment reconstruction

## Candidate controller architecture

### Site encoding

Each branch is projected into a shared hidden space:

- geometry input dimension = 6
- phylogeny input dimension = 20
- ancestry input dimension = 24
- retrieval input dimension = 25
- environment input dimension = 8

The ancestry and retrieval branch embeddings are then fused with learned memory banks:

- lineage memory
- retrieval memory

The environment branch produces an environment token and branch-specific modulation parameters. These parameters multiplicatively and additively modulate the geometry, phylogeny, ancestry, and retrieval embeddings before branch fusion.

### Branch fusion

Let the branch embeddings at site `i` be:

- `h_geom(i)`
- `h_phylo(i)`
- `h_asr(i)`
- `h_retr(i)`
- `h_env(i)`

The model computes branch weights:

`alpha(i) = softmax(G([h_geom(i), h_phylo(i), h_asr(i), h_retr(i), h_env(i)]))`

and then forms a fused site representation:

`h_i = MLP([alpha_geom(i) h_geom(i), alpha_phylo(i) h_phylo(i), alpha_asr(i) h_asr(i), alpha_retr(i) h_retr(i), alpha_env(i) h_env(i)])`

This site representation drives both unary residue scoring and pairwise interaction generation.

### Candidate embedding

At the candidate level, the controller combines:

- sequence-conditioned residue-prototype context
- candidate-specific evidence features
- pairwise summary features

The current candidate feature vector contains 51 scalar features spanning:

- source and source-group indicators
- support count
- mutation burden
- component-wise engineering scores
- fusion-ranker contribution features
- note-derived flags
- selector-prior features
- gap-to-best features

The candidate embedding feeds three heads:

- selection head
- engineering head
- policy head

## Training objectives

The controller is trained with multiple objectives, reflecting the fact that the model must behave both as a scorer and as a calibrated controller.

### 1. Selection regression

The controller predicts target-wise standardized selection scores:

`L_sel = MSE(s_hat, s_norm)`

### 2. Engineering regression

The controller predicts standardized engineering scores:

`L_eng = MSE(e_hat, e_norm)`

where `e_norm` is the standardized `mars_score`.

### 3. Policy regression

A target policy score is defined as:

`p_target = 0.30 s_norm + 0.70 e_norm`

and supervised using:

`L_pol = MSE(p_hat, p_target)`

### 4. Pairwise policy ranking

Pairwise ranking supervision is applied to candidate pairs whose target policy scores differ by a margin:

`L_rank = mean softplus(-(p_hat_i - p_hat_j))`

### 5. Decoder-field residue supervision

The unary field is supervised toward the empirical residue distribution induced by high-quality candidates:

`L_dec = CE(U, q_site)`

where `q_site` is a site-wise residue distribution weighted by candidate-level policy support.

This loss is especially important because it pushes the field itself toward realistic decode-time residue preferences.

### 6. Recovery loss

The unary field is regularized toward wild-type recovery:

`L_rec = CE(U, x_wt)`

### 7. Ancestry and retrieval alignment

Auxiliary heads align latent site representations with ancestry and retrieval targets:

- `L_asr`
- `L_retr`

### 8. Environment reconstruction

The environment representation is regularized by:

`L_env = MSE(env_hat, env_target)`

### 9. Pairwise consistency

If empirical pairwise residue support can be estimated from candidate tables, the pairwise tensor is regularized by:

`L_pair = MSE(C_hat, C_empirical)`

### 10. Conservative guard losses

The current controller uses several conservative guard terms:

- winner-guard loss
- non-decoder-guard loss
- simplicity-guard loss
- selector-anchor loss

These losses suppress:

- obviously weaker alternatives
- decoder branches that overwhelm better incumbent engineering priors
- unnecessary mutation burden
- target-specific replacement of stable incumbents

### 11. Gate regularization

Branch usage is regularized by:

- entropy regularization over mean gate usage
- gate-prior regularization conditioned on available family / template / ASR support

### Total objective

The total loss is a weighted sum:

`L_total = λ_reg L_reg + λ_sel L_sel + λ_eng L_eng + λ_pol L_pol + λ_rank L_rank + λ_dec L_dec + λ_guard L_guard + λ_rec L_rec + λ_pair L_pair + λ_asr L_asr + λ_retr L_retr + λ_env L_env + λ_gate L_gate`

with fixed coefficients defined in the implementation.

## Teacher-forced neural field decoder

The neural field decoder extends `MARS-FIELD` beyond reranking.

For a held-out target, the benchmark-time pipeline:

1. builds a runtime neural batch from the target’s live pipeline state
2. trains a leave-one-target-out neural field model using the remaining targets
3. produces learned unary and pairwise field outputs
4. combines these outputs with evidence-derived prior position fields and prior pairwise support
5. decodes `neural_decoder` candidates under constrained beam search

The key design choice is teacher-forcing at the field level:

- learned unary scores are mixed with evidence-derived position priors
- learned pairwise scores are mixed with prior pairwise support

In the current implementation, the default mixing weights are:

- prior field weight = 0.55
- prior pair weight = 0.35

This prevents the neural field from drifting into implausible regions while still allowing learned decode-time novelty.

## Final hybrid selection policy

The final paper-facing controller uses a hybrid policy.

If `x_current` is the incumbent policy candidate and `x_neural` is the top neural-policy candidate, then the controller adopts `x_neural` only if:

1. `engineering_score(x_neural) >= engineering_score(x_current)`
2. `selection_score(x_neural) - selection_score(x_current) >= -tau`

where `tau` is a small tolerance allowing minor selection-score degradation when the engineering prior is improved. In the current benchmark implementation:

- `tau = 0.10`

This policy reflects the current strongest honest deployment setting:

- neural rerank is active
- neural decoder is active
- but the final controller remains conservative on hard targets

## Evaluation protocol

The primary evaluation quantities are:

- policy selection score
- policy engineering score (`mars_score`)
- paired policy delta relative to the incumbent benchmark
- neural decoder utilization

Neural decoder utilization is explicitly quantified using:

- whether the neural decoder was enabled
- number of preview candidates
- number of retained novel candidates
- number of rejected candidates
- best retained neural-decoder-derived candidate

The benchmark should be interpreted as a paired engineering panel rather than a single-endpoint statistical trial. Accordingly, the key inferential quantities are:

- number of targets improved
- number of targets decreased
- whether failures are concentrated or diffuse
- whether neural decoding contributes retained non-redundant candidates
- whether related structures retain chemically plausible redesign logic

## Scope of the current Methods

The present implementation supports the following claim:

`MARS-FIELD` is a unified evidence-to-sequence residue-field controller with an active decode-time neural field branch in the main benchmark path.

It does not yet support the stronger claim of:

- fully joint proposal-generator / field / decoder optimization

That distinction should remain explicit in the final manuscript.
