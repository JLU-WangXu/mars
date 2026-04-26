# MARS-FIELD Methods and Results Draft v2

## Working title

MARS-FIELD: A unified evidence-to-sequence controller for protein engineering

## Summary

This draft expands the previous manuscript skeleton into a more publication-oriented Methods and Results document using the latest `benchmark_twelvepack_final` outputs. The central message is that `MARS-FIELD` is no longer only a multi-branch engineering stack with a neural reranker appended at the end. Instead, the current implementation integrates a shared residue field, a calibrated neural controller, and a decode-time neural field generator inside the benchmark-time main path.

The strongest supported benchmark statements from the current repository state are:

- twelve-target benchmark spanning ten protein families
- neural decoder enabled on `12/12` targets
- retained novel neural-decoder candidates on `5/12` targets
- total retained neural-decoder candidates: `34`
- paired policy score improved on `9/12` targets relative to the incumbent benchmark
- paired policy score decreased on `3/12` targets
- mean paired policy delta approximately `-0.001`

These numbers support a paper narrative centered on a stable, neuralized controller-decoder system rather than on a purely heuristic proposal-and-vote stack.

## Results

### The main benchmark now evaluates a coupled field-controller-decoder system

The primary result of the current codebase is architectural: the benchmark-time main path now contains a coupled field-controller-decoder loop. Each target is processed by a pipeline that builds structural, evolutionary, ancestral, retrieval, and environment-conditioned evidence; fuses these into a shared residue field; trains a leave-one-target-out neural field model; and uses that model both to score incumbent candidates and to decode `neural_decoder` candidates inside the same target run. This is a substantial shift from earlier versions of the project, where the neural component only reranked an externally generated candidate list.

Across the twelve-target panel, the neural decoder was enabled for all targets, generated 373 preview candidates in total, retained 34 novel decoded candidates after filtering, and contributed retained candidates on 5 targets. These counts show that the neural branch is not merely diagnostic. It is participating in design-space exploration as a real proposal source.

### The final controller remains benchmark-stable after adding neural decode-time generation

Relative to the incumbent benchmark, the final controller improved policy score on 9 of 12 targets and decreased it on 3 targets, with a mean paired delta of approximately `-0.001`. This is an important result because it indicates that adding a neural field decoder to the main path does not globally destabilize the benchmark. Instead, the final controller remains approximately neutral at the panel level while still delivering a majority of positive paired shifts.

Positive paired shifts were observed in `1LBT`, `adk_1s3g`, `esterase_7b4q`, `petase_5xfy`, `petase_5xh3`, `sfgfp_2b3p`, `sod_1y67`, `t4l_171l`, and `tem1_1btl`. Some of these gains correspond to better policy scores for unchanged incumbent-like winners, whereas others correspond to changed winners that remain chemically interpretable. For example, `1LBT` preserved the known safe `M298L` winner while increasing its paired policy score. `adk_1s3g` shifted to `Y24F;H28Q;M103I;H109V`, a higher-scoring engineering variant. `esterase_7b4q` and `sod_1y67` also moved to alternatives that remain plausible under the engineering scoring model.

The remaining negative paired shifts were concentrated in `CLD_3Q09_NOTOPIC`, `CLD_3Q09_TOPIC`, and `subtilisin_2st1`. This pattern is preferable to a diffuse benchmark collapse, because it identifies a small set of explicit calibration-limited targets that can be discussed as method boundaries.

### Neural decoder contribution is selective and target-dependent

The retained neural-decoder contribution was not uniform across the panel. Instead, the decoder contributed retained novel candidates in a subset of targets with more favorable field geometry and support structure. In the current final benchmark, retained neural-decoder candidates were observed in `tem1_1btl`, `petase_5xh3`, `t4l_171l`, `subtilisin_2st1`, and `sod_1y67`. The largest retained count occurred in `t4l_171l`, where 13 neural-decoder candidates passed filtering and the final policy shifted to the neural-decoder-derived `Y88S;W126F;Y139S;W158F`.

This selectivity is a useful property, not a weakness. A good engineering controller should not force decode-time novelty on every target. Instead, it should allow neural generation where the field and downstream filters provide sufficient support, while preserving incumbent stability where they do not. `1LBT` is the clearest example of this desired behavior: 32 neural-decoder previews were produced, but none passed the final retention filters, and the incumbent `M298L` solution was preserved.

### Family-stratified and prior-aware behavior remains interpretable

The twelve-target benchmark spans ten protein families, including cutinases, lysozyme-like proteins, beta-lactamase, subtilisin, adenylate kinase, GFP-like proteins, and chlorite dismutase. Three targets used family priors and two targets used ASR priors, while template-aware weighting remained active for all twelve targets. This distribution supports the claim that the method is operating across a heterogeneous evidence regime rather than on a narrow, homogeneous benchmark.

Family-level summaries further suggest that the controller is not only benefiting from a single privileged family. The final panel includes strong-performing families such as adenylate kinase, beta-lactamase, lipase-esterase, GFP-like proteins, and cutinases, while the `cld` family explicitly exposes one of the main remaining calibration challenges. This is a useful narrative for the paper because it avoids overselling universality while still supporting a broad-method contribution.

### Component ablations show that oxidation and evolutionary information remain dominant constraints

Component ablations provide a second experimental axis beyond the headline benchmark comparison. Removing oxidation information changed the top candidate on 10 of 12 targets and produced a mean score drop of approximately `3.695`, indicating that oxidation-aware engineering remains one of the strongest stabilizing constraints in the current system. Removing surface terms changed only 2 of 12 top candidates, suggesting that the present panel is less sensitive to surface terms than to oxidation and evolutionary guidance. Removing evolutionary terms changed the top candidate on 6 of 12 targets and substantially altered the score landscape, showing that profile-derived priors remain critical for ranking and calibration.

These ablations are valuable because they demonstrate that the benchmark is not driven by a single neural branch. Instead, the final controller is genuinely multi-evidence and remains strongly dependent on chemically and evolutionarily meaningful constraints.

### Case studies illustrate different operating regimes of the controller

`1LBT` should be framed as a conservative control case. The final controller preserved `M298L` as the top policy mutation while still enabling full neural decoder execution. This is exactly the type of target on which a good controller should remain cautious, because historical iterations showed that decoder-heavy behavior could produce attractive but unreliable multi-site combinations.

`TEM1` should be framed as a high-scoring engineering landscape in which the incumbent solution remains stable but the neural decoder contributes credible learned alternatives. The final policy remained `H153N;M155L;W229F;M272L`, but the best learned candidate became the neural-decoder-derived `H153Q;M155L;W229Q;M272I`. This is a strong example of neural generation contributing useful alternatives without forcing immediate replacement of the incumbent.

`PETase` should be framed as a reproducibility case across two structural contexts. In both `5XFY` and `5XH3`, the canonical aromatic redesign remained the top policy solution, supporting the view that the controller can reproduce chemically sensible engineering outcomes even after the addition of a neural field decoder.

`CLD_3Q09_TOPIC` should be framed as a calibration stress test. The final policy retained the incumbent `W155F;W156F;M167L;M212L;W227F`, while the neural branch continued to favor a nearby alternative with stronger local engineering signal but weaker final policy support. This target is therefore well suited for discussing the limits of the present controller and the need for future fully joint optimization.

## Methods

### Benchmark design

The primary benchmark consists of twelve targets spanning ten protein families. Each target defines a local design window, an associated wild-type structure, and a target-specific engineering context. The benchmark includes both structure-only and prior-augmented targets. Family priors are enabled for `adk_1s3g`, `esterase_7b4q`, and `sod_1y67`, while ASR priors are enabled for `CLD_3Q09_NOTOPIC` and `CLD_3Q09_TOPIC`. Template-aware weighting is enabled for all twelve targets.

All neural controller training is performed in a leave-one-target-out manner. For a given held-out target, the neural field model is trained using the remaining targets and is then applied to the held-out target for reranking and field decoding.

### Evidence streams

`MARS-FIELD` integrates five evidence streams:

1. geometry-conditioned structural evidence
2. phylogenetic profile evidence
3. ancestral lineage evidence
4. retrieval-based motif memory
5. environment-conditioned engineering context

Structural evidence is derived from residue-level geometric features including solvent exposure, flexibility, protected distances, and hotspot annotations. Evolutionary evidence is derived from homolog profiles and family-differential sequence preferences. Ancestral evidence is represented as posterior residue recommendations and confidence-aware lineage summaries. Retrieval evidence is represented as structure-derived motif memory. Environment evidence is represented as target-level contextual tokens, including hotspot burden, design-window size, and prior-availability indicators.

### Shared residue field

The residue field encodes the mutable positions as residue-wise and pairwise preferences. In the engineering field, these preferences are assembled from evidence-derived position fields and proposal-derived pairwise energies. In the neural field, residue-wise preferences are represented by learned unary logits over amino acids at each design position, and pairwise preferences are represented by learned interaction tensors over design-position pairs.

The neural field contains geometry, phylogeny, ancestry, retrieval, and environment branches. Ancestry and retrieval each interact with learned memory banks, enabling the model to fuse residue representations with lineage memory and retrieval prototype memory. Environment evidence modulates the branch embeddings before branch fusion. This branch-fused site representation drives the unary and pairwise outputs.

### Candidate controller

The candidate controller combines sequence-conditioned residue embeddings with candidate-specific evidence features. These features include source identity, support count, mutation burden, component-wise engineering scores, selector-rank priors, gaps to the best incumbent, and other calibration-aware signals. Pairwise summaries are fused into the same candidate representation. The controller outputs candidate-level selection, engineering, and policy predictions.

Training uses multiple objectives: selection-score regression, engineering-score regression, candidate-level policy regression, pairwise policy ranking, decoder-field supervision, winner-guard loss, non-decoder guard loss, simplicity guard loss, selector-anchor distillation, recovery loss, ancestry alignment, retrieval alignment, pairwise consistency, environment reconstruction, and gate regularization. This combination encourages the controller to remain conservative on unstable branches while still allowing learned improvements.

### Neural field decoder

For each target, the pipeline constructs a runtime neural batch from the current target state and trains a leave-one-target-out neural field model using the remaining benchmark outputs. The learned unary and pairwise outputs are then converted into decode-ready fields. Importantly, the neural field is teacher-forced by combining learned unary and pairwise preferences with evidence-derived prior fields and prior pairwise terms before decoding. This produces a calibrated neural field that can generate `neural_decoder` candidates while remaining anchored to the broader evidence regime.

Decoded neural candidates are passed through the same downstream engineering scorer and safety filters as other candidate sources. They are retained only if they satisfy support, engineering, and prior-gap criteria.

### Final hybrid policy

The paper-facing final controller uses a hybrid policy. Neural reranking and neural field decoding are active, but the final controller only replaces the incumbent when the learned alternative remains sufficiently aligned with incumbent selection-score stability while matching or improving the engineering prior. This setting represents the strongest honest deployment path in the current repository state.

### Primary metrics

The primary target-level metrics are:

- `policy_selection_score`
- `policy_engineering_score` (`mars_score`)
- paired policy delta relative to the incumbent benchmark
- neural decoder utilization statistics

Neural decoder utilization is summarized using:

- whether the neural decoder was enabled
- number of preview candidates
- number of retained novel candidates
- number of rejected candidates
- identity of the best retained neural-decoder-derived candidate

### Statistical interpretation

The benchmark is intended as a paired engineering panel rather than a classical statistical trial. Accordingly, the key interpretation is not a single p-value but the pattern of paired effects across targets. The most important quantities for the present paper are therefore:

- directionality of paired deltas
- concentration versus diffuseness of regressions
- neural decoder utilization breadth
- consistency of chemically plausible winners across related structures

## Recommended manuscript message

The most defensible paper message at this stage is:

`MARS-FIELD` is a unified neuralized evidence-to-sequence controller that now includes a decode-time neural field generator inside the main benchmark path. The controller remains benchmark-stable at the panel level, improves paired policy score on most targets, and generates retained novel neural-decoder proposals on a subset of targets, while still exposing a small number of explicit calibration-limited cases.
