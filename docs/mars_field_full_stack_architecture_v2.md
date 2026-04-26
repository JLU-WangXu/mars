# MARS-FIELD Full-Stack Architecture v2

## 1. Why This Document Exists

This document defines the full intended architecture of `MARS-FIELD` before more large-scale benchmark reruns are performed.

The purpose is to prevent the project from drifting into repeated local fixes around individual targets while never converging to a coherent, publication-grade system.

From this point onward, the codebase should be interpreted as an implementation trajectory toward the architecture defined here.

## 2. Core Principle

`MARS-FIELD` is not a multi-method voting pipeline.

It is a unified evidence-to-sequence network in which heterogeneous evidence streams are projected into a shared residue energy field, then decoded under explicit engineering constraints.

The primary computational object is therefore:

- site-wise residue energy `U(i, a)`
- pairwise coupling energy `C(i, j, a, b)`

and not:

- a candidate list
- a branch vote
- a reranked proposal table

## 3. Top-Level System

The complete system is organized into the following layers:

1. `Input Layer`
   - target structure
   - target sequence
   - protected/design masks
   - homolog MSA
   - ASR posterior or aligned ancestor set
   - retrieval memory source
   - environment / engineering context

2. `Evidence Encoder Layer`
   - geometric encoder
   - phylo-sequence encoder
   - ancestral lineage encoder
   - retrieval memory encoder
   - environment hypernetwork

3. `Unified Residue Field Layer`
   - projects all evidence into residue-wise latent field
   - produces `U(i, a)`

4. `Pairwise Energy Layer`
   - models site coupling and compatibility
   - produces `C(i, j, a, b)`

5. `Structured Decoder`
   - searches sequence space under design constraints

6. `Calibrated Selector`
   - target-wise calibration
   - prior consistency
   - safety gating

7. `Benchmark + Reporting Layer`
   - target-level
   - family-level
   - held-out-family
   - ablation
   - case-study exports

## 4. Module Definitions

### 4.1 Geometric Encoder

Function:

- convert target backbone geometry and structural context into site-level latent representations

Expected outputs:

- local geometric state per residue
- hotspot / exposure / flexibility context
- geometry-conditioned residue preference signals

Current engineering approximation:

- `structure_features.py`

Future target:

- explicit learned geometric encoder

### 4.2 Phylo-Sequence Encoder

Function:

- encode homolog MSA and family adaptation signals into site-level evolutionary evidence

Expected outputs:

- residue conservation field
- family differential preference field
- evolution-conditioned site weights

Current engineering approximation:

- `evolution.py`

Future target:

- learned sequence-evolution encoder

### 4.3 Ancestral Lineage Encoder

Function:

- encode ancestor posterior, depth, and uncertainty into lineage-aware residue constraints

Expected outputs:

- posterior residue field
- lineage confidence
- uncertainty-aware ASR recommendations

Current engineering approximation:

- `ancestral_field.py`

Future target:

- direct lineage latent branch with depth-aware embeddings

### 4.4 Retrieval Memory Encoder

Function:

- query structurally similar local motifs from a memory bank
- return residue priors and motif neighbors

Expected outputs:

- retrieval-conditioned residue evidence
- local motif support traces

Current engineering approximation:

- `retrieval_memory.py`
- small structure-template atlas built from local inputs

Future target:

- larger motif atlas
- clustering / prototype memory
- retrieval-conditioned latent embedding rather than direct score injection

### 4.5 Environment Hypernetwork

Function:

- encode stress and engineering context as conditioning signals

Expected outputs:

- context modulation vector
- environment-specific residue preference modulation

Current engineering approximation:

- score weights, heuristic environment logic, topic hooks

Future target:

- explicit environment-conditioned hypernetwork

## 5. Unified Residue Field

For each design position `i` and amino acid `a`, MARS-FIELD constructs a shared evidence state:

`r_i(a) = f_geom + f_phylo + f_asr + f_retr + f_env + f_generator`

but this should be interpreted as a network-level latent aggregation, not as a final paper formula.

The field must satisfy:

- all branches contribute in the same residue decision space
- branch outputs are comparable only after field projection
- no branch should directly own the final design decision

## 6. Pairwise Energy

The system must include a pairwise compatibility head.

Purpose:

- model residue co-occurrence compatibility
- discourage impossible or brittle multi-mutation combinations
- allow decoder to optimize sequence-level compatibility, not only site-wise greediness

Current engineering approximation:

- `energy_head.py`

Future target:

- learned pairwise head
- sparse structural neighborhood coupling

## 7. Structured Decoder

Decoder requirements:

- operate on residue field + pairwise energy
- satisfy protected positions
- satisfy mutation burden limits
- support explicit safety gating
- support future upgrade to stronger decoding algorithms

Current engineering approximation:

- constrained beam decoder

## 8. Calibration Layer

Calibration is not the main algorithm.

It exists to:

- normalize target-specific score scales
- penalize unreasonable deviations from strong engineering priors
- suppress unsafe decoder outcomes

Long-term goal:

- keep calibration lightweight while shifting modeling responsibility into the field and energy layers

## 9. Code Mapping

This architecture should be reflected in code through explicit modules:

- `marsstack/field_network/contracts.py`
- `marsstack/field_network/encoders.py`
- `marsstack/field_network/residue_field.py`
- `marsstack/field_network/system.py`

These are not replacement scripts; they are the canonical abstraction layer that future code should align with.

## 10. Definition Of Done

The architecture can be considered "publication-ready engineering version" only when:

1. retrieval is no longer a branch-specific hack
2. ASR is a first-class lineage branch
3. decoder consumes explicit field + pairwise energy
4. calibration no longer has to rescue major failure modes
5. full benchmark shows stable target-level and family-level gains
