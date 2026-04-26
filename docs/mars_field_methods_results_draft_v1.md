# MARS-FIELD Methods and Results Draft v1

This draft is written directly against the current repository state and benchmark outputs.

It is intended as a paper-starting draft, not a final polished manuscript.

## Working title

MARS-FIELD: A unified evidence-to-sequence controller for protein engineering

## Working abstract skeleton

Engineering robust protein variants typically requires combining structural intuition, evolutionary context, and proposal generators that are often deployed as separate modules. We present `MARS-FIELD`, a unified evidence-to-sequence controller that projects geometry-conditioned structural signals, phylogenetic profiles, ancestral priors, retrieval memory, and environment-conditioned engineering context into a shared residue field and uses a calibrated neural controller-decoder stack to propose and select protein designs. In the current implementation, `MARS-FIELD` integrates a learned neural residue field, a teacher-anchored selection policy, and a neural field decoder inside the main benchmark path. Across a twelve-target benchmark spanning ten protein families, the final controller improved policy score on 9 of 12 targets relative to the incumbent engineering pipeline, with a near-neutral mean paired delta of approximately -0.001, while enabling neural decoder generation on all targets and retaining novel decoded candidates on 5 of 12 targets. These results support `MARS-FIELD` as a unified, neuralized protein engineering controller that moves beyond heuristic score stacking toward an evidence-conditioned field model.

## Methods

### Method overview

`MARS-FIELD` is designed as a unified evidence-to-sequence controller rather than a collection of independent proposal tools. The pipeline absorbs five classes of evidence: structural geometry, phylogenetic sequence context, ancestral lineage information, retrieval-based motif memory, and environment-conditioned engineering constraints. These signals are combined into a shared residue-level field over the design positions, from which both candidate evaluation and decode-time proposal generation are derived.

In the current implementation, the system contains three interacting layers. First, an evidence layer extracts geometry-conditioned, evolutionary, ancestral, retrieval, and environmental signals from the target structure and associated sequence context. Second, a shared field layer constructs residue-level and pairwise interaction preferences over the mutable positions. Third, a calibrated controller layer ranks incumbent candidates, optionally decodes new candidates from the field, and applies a hybrid final policy that preserves stable incumbents on hard targets while allowing neural branch gains when sufficiently supported.

### Structural, evolutionary, ancestral, retrieval, and environment evidence

Structural evidence is derived from per-residue geometric features, including solvent exposure, flexibility, protected distances, and hotspot annotations. Evolutionary evidence is derived from homolog profiles and family-differential sequence preferences. Ancestral evidence is represented as posterior residue preferences and confidence-weighted lineage-derived recommendations. Retrieval evidence is represented as structure-derived residue memory and motif-atlas support. Environment evidence is encoded as target-level contextual tokens, including oxidation hotspot burden, flexible-surface burden, design-window size, and prior availability flags.

These signals are represented in the codebase by the field-network encoders and associated bundle contracts. In particular, ancestry and retrieval are no longer treated as independent external recommendation tables but are mapped into learned latent branches that participate in the neural controller.

### Shared residue field

The shared residue field represents the design problem as residue-wise and pairwise preferences over the mutable positions. In the engineering baseline, this field is assembled from unified evidence fields and proposal-derived pairwise energies. In the neuralized controller, the field is represented by learned unary logits over amino acids at each design position and learned pairwise interaction tensors over design-position pairs.

The current neural field contains geometry, phylogeny, ancestry, retrieval, and environment branches. Ancestry and retrieval are each connected to learned memory banks, allowing the model to fuse latent residue representations with lineage memory and retrieval prototype memory. Environment evidence modulates the branch representations before branch fusion. The resulting site hidden states drive both unary residue preferences and low-rank pairwise interaction terms.

### Candidate-level neural controller

Candidate-level decision making is performed by a neural controller that combines sequence-conditioned residue embeddings with candidate-specific evidence features. These candidate features include source type, support count, mutation burden, component-wise engineering terms, rank-calibrated selector features, and selector-prior context. Pairwise summaries are also fused into the candidate embedding. The controller outputs candidate-level selection, engineering, and policy predictions.

To stabilize the neural controller, training uses multiple calibration losses. These include direct regression to target-wise normalized selection scores, engineering-score regression, pairwise policy ranking, a decoder-field residue supervision term, and several conservative guard losses that discourage unstable winners, over-aggressive decoder branches, and unnecessary mutation burden. A selector-anchor distillation loss additionally preserves the incumbent selector’s stability on targets where a purely neural replacement remains risky.

### Neural field decoder

The latest version of the pipeline introduces a neural field decoder into the main execution path. For each target, the pipeline constructs a runtime neural batch from the live pipeline state, trains a leave-one-target-out neural field model using the remaining targets, and converts the learned unary and pairwise outputs into decode-ready residue fields. These fields are then used by a constrained beam decoder to generate `neural_decoder` candidates.

To keep the decoder grounded, the decoded neural field is not used in isolation. Instead, the neural unary and pairwise terms are combined with evidence-derived prior fields and pairwise priors before decoding. This produces a teacher-forced neural field that remains biologically and engineering-wise anchored while still enabling decode-time neural proposal generation.

### Final hybrid selection policy

The final paper-facing controller is a hybrid policy. Neural reranking is active, and the neural field decoder is active, but the final controller only adopts a neural branch replacement when the learned alternative does not materially violate incumbent selection-score stability while preserving or improving engineering prior. This choice reflects the current repository’s strongest honest deployment setting.

### Benchmark protocol

The main benchmark consists of twelve targets spanning ten protein families. Neural controller training is leave-one-target-out at the target level. The main deployment arm used in the current draft is `benchmark_twelvepack_final`, which runs the final hybrid controller with neural field decoding enabled in the pipeline path. Paired comparisons are evaluated against the current incumbent engineering benchmark. The primary target-level metrics are policy selection score and engineering score (`mars_score`), with additional accounting for neural decoder utilization.

## Results

### A neuralized field-controller architecture supports decode-time proposal generation in the main pipeline

The first result is architectural rather than purely numeric: `MARS-FIELD` is no longer limited to reranking externally generated candidates. In the current implementation, the pipeline constructs `neural_position_fields`, `neural_pairwise_energy_tensor`, and `neural_decoder_preview` assets for each target, and decoded neural candidates can be reinjected into the main ranking path. This converts the system from a pure candidate reranker into a field-controller-decoder loop.

Across the twelve-target benchmark, the neural decoder was enabled on all 12 targets. It produced 373 preview candidates in total, retained 34 novel decoded candidates after safety and engineering filtering, and injected retained neural-decoder candidates on 5 of 12 targets. These numbers indicate that the neural field is participating as a genuine generator rather than only as a passive auxiliary scorer.

### The final controller remains competitive with the incumbent benchmark while introducing end-to-end neural decoding

We next compared the final controller against the incumbent benchmark on the twelve-target panel. Relative to the current benchmark, the final controller improved policy score on 9 of 12 targets and decreased it on 3 of 12 targets, with a mean paired delta of approximately -0.001. This near-neutral mean effect is important: it shows that the neuralized end-to-end controller can be inserted into the pipeline without causing a broad performance collapse, even though the system is now substantially more learned than the original engineering approximation.

Several targets showed clear positive movement. On `1LBT`, the incumbent winner remained `M298L`, but the final controller increased the policy score from 4.715593 to 4.99918, indicating that the end-to-end controller strengthened the incumbent rather than destabilizing it. On `ADK`, the controller shifted the overall and policy winner to `Y24F;H28Q;M103I;H109V`, with a paired policy delta of +0.160602 relative to the incumbent `Y24F;H28Q;M103L;H109Q`. On `PETase 5XFY`, the canonical four-site aromatic redesign `Y3F;Y40F;Y41F;Y117F` remained the top solution, but its policy score increased from 4.671612 to 4.797701. `SFGFP` and `SOD` similarly retained chemically plausible winners while achieving positive paired deltas.

The remaining regressions were concentrated rather than diffuse. `CLD_3Q09_NOTOPIC` and `CLD_3Q09_TOPIC` remained on the same top policy sequence as the incumbent but showed lower policy scores, while `subtilisin_2st1` shifted from `Y21F;Y104F;W241F;Y262F` to `Y21F;Y104F;W241F;Y262Q`, producing a negative paired delta. These targets should be treated as explicit limitations of the present controller rather than hidden failure modes.

### Case studies illustrate the balance between stability and neural proposal generation

`1LBT` serves as a conservative control case. The final controller preserved `M298L` as the policy winner, which is desirable because this target has consistently exposed over-eager decoder behavior in previous iterations. At the same time, the pipeline now builds neural field decode assets for the target and reports 32 neural decoder preview candidates, even though none passed the retained-candidate filter. This behavior is precisely what we want from a safety-preserving target: the neural branch is active, but the controller does not force adoption of weak neural-decoder proposals.

`TEM1` serves as a high-scoring engineering case with meaningful neural decoder contribution. The incumbent policy winner remained `H153N;M155L;W229F;M272L`, but the best learned candidate in the final pipeline became the neural-decoder-derived `H153Q;M155L;W229Q;M272I`, with ranking score 4.836415. The pipeline summary additionally shows that the neural field decoder retained 5 novel decoded candidates on this target. This is a useful case for illustrating that the controller can preserve a stable incumbent while still surfacing decoder-derived learned alternatives worth discussion.

`PETase 5XFY` and `5XH3` provide stable replicate cases. In both structures, the canonical aromatic redesign remained the top policy solution under the final controller. This is important for the paper because it shows that the controller does not need to change the top sequence to add value; instead, it can reproduce a chemically sensible design across two structural contexts while still operating through a neuralized field-decoder stack.

`CLD_3Q09_TOPIC` provides a topic-conditioned case with explicit room for further calibration. The top policy sequence remained `W155F;W156F;M167L;M212L;W227F`, but the neural top candidate favored the more aggressive `W155F;W156F;M167I;M212L;W227F`, which has a higher engineering score but weaker final selection score. This is precisely the type of tradeoff we should discuss in the paper: the neural field can surface alternative high-engineering local optima, but the final controller still needs calibrated gating to decide when such alternatives should replace the incumbent.

### Decoder utilization is target-specific rather than uniformly beneficial

A useful secondary result is that neural decoder contribution is selective. The decoder contributed retained novel candidates in `TEM1`, `PETase 5XH3`, `T4L`, `subtilisin`, and `SOD`, but not in all targets. This is favorable from an engineering perspective because it suggests that the controller is not simply generating additional diversity for its own sake. Instead, decode-time generation is selectively allowed where the field and downstream filters support it. In other words, the neural field decoder behaves like a calibrated proposal source rather than a blind generator.

## Suggested manuscript framing

The current draft should be framed this way:

- The main method contribution is a unified evidence-to-sequence controller.
- The main engineering contribution is integrating a neural field decoder into the benchmark-time main path.
- The main benchmark result is that the end-to-end controller remains competitive while activating decode-time neural generation.
- The main limitation is that full proposal-generator / field / decoder joint optimization has not yet been implemented.

## Immediate next writing tasks

1. Convert the benchmark comparison into a polished Results subsection with a compact table.
2. Expand the case-study text for `1LBT`, `TEM1`, `PETase`, and `CLD`.
3. Write the Methods subsection around the actual code modules:
   - `neural_model.py`
   - `neural_training.py`
   - `neural_generator.py`
   - `run_mars_pipeline.py`
4. Add a brief limitations paragraph clarifying that the current system is end-to-end at the controller-decoder level but not yet fully jointly trained at the proposal-generator level.
