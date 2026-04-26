# Mars Stack Project Plan

## 1. Working title

`MarsStack: a general protein engineering framework for extreme-environment hardening`

## 2. What we are trying to publish

Not "a modified ProteinMPNN".

The publishable unit should be:

**a general design framework that improves success rate for engineering proteins toward Mars-relevant stress tolerance across protein families.**

## 3. Immediate paper strategy

### Stage A: ML systems paper

Target:

- `ICLR 2027` or a similar ML venue

What must exist:

- a clear computational method
- a retrospective benchmark over `5-10` proteins
- strong baselines such as:
  - wild-type / rational heuristic
  - vanilla `ProteinMPNN`
  - `ProteinMPNN + simple fixed-position engineering`
- ablations for each Mars objective term

Wet-lab is helpful but not strictly required for the first ML submission if the benchmark is strong.

### Stage B: high-impact biology / methods paper

Target:

- `Nature Communications`
- `Cell Systems`
- later, only if the biology is truly broad, a higher-tier journal

What must exist:

- prospective designs on `3-5` proteins
- expression and activity retained
- stress assays showing improved tolerance
- a generalizable biological principle, not just one good case

## 4. Why CALB is the right seed system

- cheap and mature assay system
- structure is known and stable
- expression and purification are routine
- stress readouts are easy to define
- our current structural analysis already suggests a precise antiradiation hypothesis around `M298`

CALB should be the seed case, not the entire story.

## 5. Mars objective v0

The first version does not need model retraining.

It can be a reranker layered on top of inverse folding outputs.

### 5.1 Terms to include

- oxidation hotspot penalty
  - exposed `Met`, unpaired `Cys`, exposed oxidation-prone aromatics
- surface chemistry term
  - charge density
  - hydration-favoring substitutions
  - acidic/basic balance
- freeze-dry / rehydration proxy
  - loop exposure
  - surface polarity packing
  - avoidance of sticky hydrophobes on highly exposed loops
- flexibility / weak-point term
  - high `B-factor` or high-mobility segments
- protected-site constraint
  - catalytic residues
  - native disulfides
  - glycosylation positions
- evolutionary constraint
  - optional MSA / PSSM / family frequency prior

### 5.2 First implementation choice

`MarsScore v0` should be rule-based and transparent.

That gives us:

- fast iteration
- interpretable ablations
- easier method writing

Later versions can become:

- learned reranker
- adapter / lightweight finetune on top of inverse folding models

## 6. Benchmark design

### 6.1 Retrospective benchmark

Need `5-10` proteins with:

- solved structures
- public mutational stability or stress-tolerance data
- common assay systems

Candidate protein categories:

- lipases
- esterases
- oxidoreductases
- fluorescent proteins
- DNA-binding proteins
- small enzyme scaffolds with abundant engineering literature

### 6.2 Prospective benchmark

Need `3-5` proteins with low assay cost and broad spread.

Recommended first pool:

1. `CALB`
2. one bacterial esterase or cutinase
3. one soluble bacterial enzyme with a simple colorimetric assay
4. one fluorescence readout protein if expression is easy

## 7. Wet-lab package for each prospective protein

- WT construct
- top `2-4` MarsStack variants
- top `1-2` vanilla ProteinMPNN controls
- optional rational-design control

Readouts:

- expression / soluble yield
- baseline activity
- freeze-thaw residual activity
- oxidative stress residual activity
- optional lyophilization / rehydration recovery
- optional irradiation if platform access exists

## 8. Milestones

### Milestone 1

Build a working local design stack for `CALB`.

Deliverables:

- constrained `ProteinMPNN` run
- `MarsScore v0` reranking
- ranked candidate table

### Milestone 2

Generalize the pipeline to another `2-3` proteins.

Deliverables:

- reusable protein config format
- automated protected-site handling
- shared benchmark tables

### Milestone 3

Retrospective benchmark at `5-10` proteins.

Deliverables:

- baseline comparison plots
- ablations
- method draft figures

### Milestone 4

Prospective wet-lab validation.

Deliverables:

- construct list
- assay plan
- first biological figures

## 9. Current status

- `ProteinMPNN` source downloaded locally
- local wrapper scripts being added
- `CALB / 1LBT` selected as seed case
- first constrained design run targeted at positions `249`, `251`, and `298`
