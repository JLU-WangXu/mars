# Evolution Train Schema

This layer is for learning environment-aware priors rather than evaluating final generalization.

## 1. Recommended unit

Prefer one directory per family-level dataset:

- `evolution_train/<dataset_id>/manifest.yaml`
- `evolution_train/<dataset_id>/positive.fasta`
- `evolution_train/<dataset_id>/negative.fasta`
- optional `evolution_train/<dataset_id>/aligned.a3m`
- optional `evolution_train/<dataset_id>/labels.csv`
- optional `evolution_train/<dataset_id>/structures/`

## 2. Meaning of positive and negative

Examples:

- `positive`: psychrophilic, desiccation-tolerant, radiation-tolerant, halophilic, or otherwise adapted homologs
- `negative`: mesophilic or less-adapted homologs from the same broad family

The important point is not absolute perfection of labels.

The important point is:

- consistent collection rules
- same-family comparison
- minimal train/test leakage

## 3. What to learn

This training layer can support three model types later:

### A. Sequence classifier or scorer

Input:

- sequence
- optional MSA profile

Target:

- adaptation class or probability

### B. Pairwise preference model

Input:

- aligned positive/negative homolog sets

Target:

- residue or motif preference at each position

### C. Inverse-folding reranker

Input:

- candidate sequence
- structure-derived features
- evolutionary context

Target:

- adapted vs non-adapted score

## 4. Manifest fields

Each `manifest.yaml` should eventually include:

- `dataset_id`
- `family`
- `adaptation_axis`
- `positive_definition`
- `negative_definition`
- `label_unit`
- `source_strategy`
- `split_group`
- `positive_fasta`
- `negative_fasta`
- `aligned_a3m`
- `labels_csv`
- `notes`

## 5. Split rules

- split by family or clade, not only by rows
- do not let close homologs of benchmark proteins enter the learned prior evaluation pool
- keep `benchmark_pool` families fully auditable and preferably held out from any finetune

## 6. First collection priority

Start with:

1. esterase/lipase family pairs
2. adenylate kinase cold-adaptation pairs
3. superoxide dismutase oxidative-tolerance pairs

These three are the fastest path to a useful evolutionary prior for `MarsStack`.
