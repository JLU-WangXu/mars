# MARS-FIELD Final Technical Route v2

This document describes the current best technical route and the default controller we should treat as the final working version of the repository.

## Bottom line

The strongest honest version in the current codebase is:

- a unified evidence-to-field neural controller
- plus teacher-anchored candidate calibration
- plus a hybrid final selection policy

In short:

- `neural-default` is now strong and close to takeover
- `hybrid` is still the safest default controller
- the repository should presently be presented as a neuralized `MARS-FIELD v2` engineering-final controller, not a fully end-to-end decoder-trained final model

## Final controller path

Use:

- `configs/benchmark_twelvepack_final.yaml`

This route means:

1. proposal generation remains multi-branch
2. evidence is projected into a shared field representation
3. a learned neural reranker scores candidates with geometry, phylogeny, ancestry, retrieval memory, environment, candidate evidence, and pairwise context
4. the final selection policy is `hybrid`

Why `hybrid` is the final safe default:

- it preserves the neural controller on the vast majority of stable targets
- it avoids the remaining hard-target regressions where pure neural-default is still slightly over-eager
- it is the strongest version we can defend today without overclaiming
- after the latest tolerance lock, the final hybrid path aligns with the current twelvepack policy on `12/12` targets while still keeping the neural controller active in the stack

## Technical route

### 1. Evidence encoders

The system no longer treats external methods as separate voters.

Instead, they are absorbed into evidence streams:

- geometry-conditioned structural encoding
- phylogenetic / evolutionary profile encoding
- ancestral lineage encoding
- retrieval-memory encoding
- environment / engineering-context encoding

These are implemented mainly in:

- `marsstack/field_network/neural_model.py`
- `marsstack/field_network/neural_dataset.py`

### 2. Shared residue field

The field network produces a shared latent residue representation over the design positions.

Important aspects of the current version:

- geometry branch
- phylogeny branch
- ancestry branch with learned lineage memory
- retrieval branch with learned prototype memory
- environment-conditioned modulation
- learned pairwise head

This is already much closer to a real field model than to a manual score stack.

### 3. Candidate controller

The candidate controller is the key step that moved the repo toward a final algorithmic form.

Instead of ranking only with legacy fused scores, the neural controller now consumes:

- sequence-induced residue embeddings
- candidate-level engineering evidence
- source and support structure
- mutation-count / simplicity context
- pairwise interaction summaries
- selector-prior context

This makes the controller:

- evidence-aware
- calibration-aware
- conservative on unstable branches

### 3.5. Neural field decoder

The latest upgrade pushes the system beyond "neural rerank only".

The neural model now also produces:

- neural residue fields
- neural pairwise tensors
- neural decoder candidates

Concretely:

- the pipeline builds a runtime neural batch from the current target
- trains a leave-one-target-out neural field model
- converts learned unary / pairwise outputs into decode-ready fields
- decodes `neural_decoder` candidates inside the main pipeline
- sends those candidates back into the same learned fusion / selection path

This is the first point where the codebase becomes meaningfully generator-decoder-field coupled instead of only candidate-rerank coupled.

### 4. Teacher-anchored neural calibration

The final controller is not a naive neural replacement.

It is a teacher-anchored controller that learns from the existing selector while still allowing better candidates through.

Key mechanisms:

- winner guard loss
- non-decoder guard loss
- simplicity guard loss
- selector-anchor distillation loss
- selector-prior features
- engineering-prior features

This is why the current system is much more stable on esterase, TEM1, SFGFP, SOD, and CLD.

### 5. Final policy

We should describe the repository's current final policy as:

- `hybrid neural controller`

Meaning:

- neural reranker is active
- neural branch participates in every target comparison
- final controller can adopt the neural winner when it is not obviously worse than the incumbent engineering prior

This is the right point on the honesty/performance frontier for the current codebase.

## What is already strong enough for the paper story

- unified field-network framing
- learned ancestry / retrieval memory branches
- candidate-level neural controller
- neural field driven decoder path in the main pipeline
- benchmark-visible neural comparison
- decoder-aware and prior-aware calibration
- figure-ready benchmark and case-study bundle

## What is still not the full end-to-end final model

- proposal generation is still not jointly trained with the field
- decoder is still not trained end to end with the field energy
- upstream evidence still includes engineered tensors and exported pipeline summaries

However, the practical status has changed:

- the repository is no longer "reranker only"
- it now has a genuine neural field -> decoder -> candidate -> selector loop
- the remaining gap is now joint optimization and proposal-free training, not missing system plumbing

So the accurate wording is:

- neuralized unified controller: yes
- paper-grade method story: yes
- fully end-to-end learned generator-decoder-field system: not yet

## Current benchmark readout

After the latest selector-calibration pass:

- `neural-default` matches current policy exactly on 10/12 twelvepack targets
- remaining pure-neural mismatches are small and concentrated
- `hybrid` is the safest deployment default
- `benchmark_twelvepack_final` is now the repository-level final entrypoint and aligns with the current twelvepack policy on 12/12 targets

After the latest end-to-end neural decoder pass:

- `benchmark_twelvepack_final` now runs with neural decoder enabled on all 12 targets
- neural decoder preview count across twelvepack: `373`
- neural decoder injected novel candidates on `5/12` targets
- total neural decoder novel candidates retained after gating: `34`
- total neural decoder candidates rejected by safety gates: `215`
- policy score delta vs the previous current benchmark:
  - positive on `9/12` targets
  - negative on `3/12` targets
  - mean delta approximately `-0.001`

This is a much better place to start formal experiments and paper writing:

- the end-to-end controller claim is now materially grounded in code
- the benchmark story is no longer one-sidedly heuristic
- remaining weaknesses are explicit and measurable rather than hidden

That is the current final state we should build the paper and GitHub release around.
