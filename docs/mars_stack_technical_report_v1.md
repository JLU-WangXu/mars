# MarsStack Technical Report v1

## 1. Project definition

`MarsStack` is not defined as a modified `ProteinMPNN`.

The project is defined as:

**a general protein engineering framework for extreme-environment hardening that combines multiple proposal generators, a Mars-oriented risk objective, and evolutionary priors, then validates transfer across protein families.**

This framing matters because:

- it is broader than a single model tweak
- it supports both learned and heuristic proposal branches
- it creates a clearer path to a method paper plus a later biology paper

## 2. Scientific motivation

The biological target is not one specific stress only.

The stack is meant to capture a shared design logic behind:

- oxidative stress tolerance
- freeze-thaw resilience
- lyophilization / dry-rehydration resilience
- low-temperature function retention

The present implementation emphasizes surface chemistry and oxidation-aware substitutions because these are:

- structurally interpretable
- experimentally actionable
- transferable across common soluble proteins

## 3. Current method stack

### 3.1 Input layer

Each design run starts from:

- one structure
- one target chain
- one wild-type sequence
- protected positions
- a design window
- optional homolog FASTA or aligned family sequences

The pipeline also supports preprocessing of noncanonical residues in structures.

Current implemented example:

- `sfGFP / 2B3P` is preprocessed by rewriting the mature chromophore residue `CRO` to a design-time `TYR` template

### 3.2 Structure feature layer

For each residue, the pipeline computes:

- approximate `SASA`
- mean `B-factor`
- distance to protected residues
- disulfide membership
- glycosylation motif flag

From these residue-wise features, the stack derives:

- oxidation hotspots
- flexible exposed positions

### 3.3 Proposal layer

The proposal layer is now explicitly multi-branch.

Current branches:

- `baseline_mpnn`: constrained `ProteinMPNN`
- `mars_mpnn`: constrained `ProteinMPNN` with Mars residue bias and omit masks
- `local_proposal`: chemistry-aware local enumerative proposal branch
- `manual`: rational control branch

Prepared but not yet active:

- `esm_if`: `ESM-IF1` integration point already wired into the pipeline, pending environment completion

### 3.4 Chemistry-aware local proposal branch

This branch was added because reranking alone cannot rescue substitutions that never enter the candidate pool.

Its role is to inject chemistry-safe proposals directly into the search space, especially for oxidation-prone residues.

Current logic:

- prefers safe oxidation substitutions such as `M -> L/I/V`
- promotes aromatic hardening patterns such as `Y -> F` and `W -> F/Y`
- allows selected polar replacements for exposed histidines
- uses profile information when available, but profile-derived residues are gated by Mars chemistry rules at oxidation hotspots

This branch is not the final learned solution, but it is strategically useful because it exposes where the current learned generators are weak.

### 3.5 Evolutionary prior layer

The current evolutionary prior is profile-based.

Implemented pipeline:

- collect homolog FASTA
- anchor-align homologs to the wild-type sequence
- build a per-position frequency profile
- use profile log score as one term in reranking

This is already active for all six current benchmark targets.

Newly added homolog-backed targets:

- `sfGFP`
- `T4 lysozyme`
- `subtilisin`

### 3.6 Mars objective layer

The current `MarsScore v0` includes:

- oxidation hardening reward
- oxidation-bad substitution penalty
- surface hydration reward
- sticky exposed hydrophobe penalty
- manual position preference bias
- evolutionary profile prior
- mutation burden penalty

Important rule:

- oxidation and flexibility terms are only evaluated on positions that are open for design

### 3.7 Reporting layer

The benchmark output is now split into two complementary views:

- `overall winner`
- `best learned winner`

This change is important because otherwise the control branches can hide whether the learned generators are actually improving.

## 4. Engineering advances already completed

### 4.1 Gap-aware template handling

Some structures parsed by `ProteinMPNN` contain internal placeholder gaps in the parsed chain sequence.

This caused a critical issue in `TEM-1 / 1BTL`, where:

- parsed template length exceeded actual residue count
- fixed positions, residue bias, and output scoring were misaligned

This is now fixed by:

- mapping real residue numbers to parsed indices
- applying masks and bias in parsed sequence space
- collapsing generated sequences back to actual residue space before scoring

### 4.2 Noncanonical residue preprocessing

The pipeline now supports structure preprocessing before design.

Current validated use case:

- `sfGFP / 2B3P` with chromophore normalization

This is important for later extension to proteins with:

- mature chromophores
- ligated cofactors
- modified residues in deposited structures

### 4.3 Family-level reporting

The codebase now includes a first family-level summary output:

- `outputs/benchmark_sixpack/family_summary.csv`

This is the first step toward true held-out-family reporting.

## 5. Current benchmark state

### 5.1 Six-protein panel

The current panel is:

- `CALB / 1LBT`
- `TEM-1 / 1BTL`
- `PETase / 5XFY`
- `sfGFP / 2B3P`
- `T4 lysozyme / 171L`
- `subtilisin / 2ST1`

### 5.2 Current benchmark summary

Current topline:

- `CALB`: overall `A251E;M298L`; best learned `R249Q;A251S;M298L`
- `TEM-1`: overall `H153Q;M155L;W229F;M272L`; best learned `H153Q;M155I;M272L`
- `PETase`: overall and best learned both `Y3F;Y40F;Y41F;Y117F`
- `sfGFP`: overall `H25Q;H139Q;Y182F;H231Q`; best learned `Y182F;H231E`
- `T4L`: overall `Y88F;W126F;Y139F;W158F`; best learned `Y88F;Y139F;W158F`
- `subtilisin`: overall `Y21F;Y104F;W241F;Y262F`; best learned `Y21F;Y104D;Y262F`

### 5.3 Interpretation

These results imply:

- the stack already produces meaningful cross-protein signal
- the reranking layer is no longer the main bottleneck
- the main bottleneck is the weakness of the learned proposal branch relative to the chemistry-aware local branch

In other words:

- the objective is useful
- the reporting is now informative
- the learned generators still need strengthening

## 6. Main technical conclusion

The project should not spend the next phase primarily on more hand-tuning of `MarsScore`.

The main technical gap is:

**stronger learned proposal generation**

That is why the next best step is:

- integrate `ESM-IF1` as a second learned inverse-folding generator

rather than:

- continuing to overfit more score heuristics

## 7. Why ESM-IF is the right next step

`ESM-IF1` is attractive because it is:

- already trained for inverse folding
- structurally grounded
- complementary to `ProteinMPNN`
- a stronger candidate source for fair method comparisons

The code integration point is already prepared:

- `scripts/run_esm_if_generator.py`

The current remaining blocker is environment setup, not method design.

## 8. Why held-out family split is the second critical next step

Without a family-aware evaluation, the story risks looking like a collection of protein-specific engineering examples.

A held-out family split is needed to support the claim that the stack is a general framework.

Current family assignments are already explicit for the six-target panel.

Fastest local expansion path already available in inputs:

- `ADK / 1S3G`
- `esterase / 7B4Q`
- `SOD / 1Y67`

Those three would move the panel from six proteins to nine proteins without waiting on new structures.

## 9. Immediate next steps

### 9.1 Method

- activate `ESM-IF` as a third learned generator branch
- compare `baseline_mpnn`, `mars_mpnn`, and `esm_if`
- keep `local_proposal` as chemistry-aware non-learned upper baseline

### 9.2 Benchmark

- expand six-protein panel to nine proteins
- export family-level aggregate tables
- implement leave-one-family-out reporting

### 9.3 Paper logic

Recommended main method claim:

- a multi-generator protein hardening framework with Mars-specific reranking improves design direction across protein families better than raw inverse folding alone

Recommended evidence stack:

- benchmark table with `overall` and `best learned`
- ablation table
- family-level summary
- held-out-family transfer table

## 10. Current risk assessment

Main risks:

- `ESM-IF` environment installation may still require dependency adjustment on the current Windows + Python setup
- the learned branch still underperforms on some targets such as `sfGFP` and `subtilisin`
- family count is still too small for a final top-tier method submission

Main strengths:

- the codebase is already beyond a single-case prototype
- the benchmark and reporting logic are now method-shaped
- the route to a stronger learned branch is clear

## 11. Bottom line

`MarsStack` is now best understood as:

**a general extreme-environment protein engineering stack with a strong reranking objective and an increasingly mature proposal layer, where the next decisive milestone is upgrading learned generation and validating transfer across held-out families.**
