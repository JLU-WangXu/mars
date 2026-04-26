# MarsStack Method v0

## 1. Positioning

`MarsStack` is not framed as a modified inverse-folding model.

The method is:

**a general extreme-environment protein hardening stack that combines a structure-conditioned sequence prior with a Mars-specific objective and optional evolutionary priors.**

## 2. Current algorithm

### Inputs

- protein structure (`PDB`)
- target chain
- wild-type sequence
- protected residues
- designable positions
- optional homolog FASTA or aligned family sequences

### Stage A: structure features

Extract per-residue features from the target chain:

- approximate SASA
- mean `B-factor`
- distance to protected positions
- disulfide membership
- glycosylation motif flag

### Stage B: candidate generators

Current active generator:

- constrained `ProteinMPNN`
- optional `ESM-IF`
- chemistry-aware local proposal branch that enumerates safe substitutions at the design window

Current supporting prior:

- anchored homolog alignment followed by a per-position profile prior
- optional positive/negative family differential prior
- optional ASR-compatible profile prior
- structure-aware position weighting for evolution terms, derived from target-template SASA / flexibility / hotspot context

Planned generators:

- stronger learned generators beyond vanilla inverse folding
- homology-template retrieval for family templates
- true ASR-guided residue recovery proposals with real reconstructed ancestors

### Stage C: Mars objective

Current `MarsScore v0` terms:

- oxidation hotspot hardening reward
- bad oxidation-prone substitutions at hotspot penalty
- surface hydration reward on exposed flexible positions
- sticky hydrophobe penalty on exposed flexible positions
- manual per-position preference bias
- optional evolutionary profile prior
- optional ASR profile prior
- optional family differential evolution prior
- optional structure-aware position weighting over evolution terms
- mutation burden penalty

Important implementation rule:

- only score oxidation and flexibility objectives on positions that are actually open for design in the current run

### Stage D: rerank and shortlist

Combine:

- manual rational controls
- chemistry-aware local proposals
- vanilla inverse-folding outputs
- Mars-biased inverse-folding outputs
- optional `ESM-IF` outputs

Then rerank and export:

- ranked CSV
- shortlist FASTA
- pipeline summary
- per-target profile summary
- benchmark tables with `overall winner` and `best learned winner`

### Implementation note: gap-aware templates

Some templates parsed by `ProteinMPNN` include placeholder gap positions in the chain sequence. `MarsStack` now handles those cases explicitly by:

- mapping real structure residue numbers to parsed-chain indices
- applying fixed-position masks and residue bias in parsed sequence space
- collapsing generated sequences back to the actual residue sequence before scoring

This is required for proteins such as `TEM-1 / 1BTL`, where the parsed chain length is longer than the real residue count.

## 3. Current benchmark state

Validated configs:

- `configs/calb_1lbt.yaml`
- `configs/tem1_1btl.yaml`
- `configs/petase_5xfy.yaml`
- `configs/sfgfp_2b3p.yaml`
- `configs/t4l_171l.yaml`
- `configs/subtilisin_2st1.yaml`
- `configs/benchmark_triplet.yaml`
- `configs/benchmark_sixpack.yaml`
- `configs/benchmark_ninepack.yaml`

### Current method boundary

- family-conditioned prior: already in the executable mainline
- structure-aware evolution weighting: already in the executable mainline
- ASR-compatible input branch: already in the executable interface
- true ASR result: not yet, because the current benchmark panel still has no real loaded ancestor sequences
- publication-grade family generalization: not yet, because the panel is still too small and still dominated by the local rational branch on overall-winner tables

### CALB / 1LBT

- `249`
- `251`
- `298`

Current local result:

- baseline constrained `ProteinMPNN` top outputs remain centered on `M298R`, `M298P`, and `M298K`
- `MarsStack` top outputs are redirected to `M298L`
- top benchmark winner is the rational single mutant `M298L`

Interpretation:

- the inverse-folding prior alone is misaligned with the anti-oxidation objective at position `298`
- the Mars objective is strong enough to rewrite the search direction without retraining the generator

### TEM-1 / 1BTL

Design window:

- `153`
- `155`
- `229`
- `272`

Current local result:

- baseline constrained `ProteinMPNN` top learned candidate is `H153E;M155I;M272D`
- Mars-biased `ProteinMPNN` shifts the top learned candidate to `H153Q;M155I;M272L`
- top overall benchmark winner is the rational combination `H153Q;M155L;W229F;M272L`

Interpretation:

- Mars constraints pull the learned search away from oxidation-prone choices at exposed methionines
- the pipeline now supports structures whose parsed `ProteinMPNN` template sequence contains internal gap placeholders

### PETase / 5XFY

Design window:

- `3`
- `40`
- `41`
- `117`

Current local result:

- the benchmark currently favors the rational aromatic hardening set `Y3F;Y40F;Y41F;Y117F`
- Mars-biased `ProteinMPNN` recovers a partial subset `Y40F;Y41F;Y117F`

Interpretation:

- PETase is currently a useful stress-test target, but not yet the strongest learned-method showcase
- this is a good place to add stronger evolutionary or family-profile guidance next

### Benchmark takeaway

- six-protein batch execution is now working end-to-end
- all six targets now have either legacy or newly added homolog/profile priors
- first ablations (`full`, `no_oxidation`, `no_surface`, `no_evolution`) are exported automatically
- the current stack is past the single-protein toy stage, but still closer to a serious workshop or pre-submission draft than a top-conference final package
- current bottleneck is no longer reranking alone; it is the strength of the learned proposal branch relative to the chemistry-aware local branch

## 4. Recommended paper structure

### Method paper

Target:

- `ICLR 2027` or similar

Core claim:

- a general extreme-environment protein engineering stack improves design direction across families better than raw inverse folding

What to build next:

- expand to `5-10` proteins with a held-out family split
- compare `WT`, rational heuristics, vanilla `ProteinMPNN`, and `MarsStack`
- add ablations for each objective term
- add at least one stronger learned generator (`ESM-IF` or a family-profile proposal branch)

### Biology paper

After wet-lab support:

- prospective validation on `3-5` proteins
- retained expression and activity
- improved oxidative / freeze-thaw / lyophilization tolerance

## 5. Immediate next steps

1. expand the benchmark triplet to `5-10` proteins
2. strengthen the learned proposal branch so it can compete with the chemistry-aware local branch
3. add an `ESM-IF` candidate generator behind the same interface
4. build a held-out family evaluation and aggregate statistics
