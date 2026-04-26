# MarsStack Datasets

This directory is split into two dataset layers.

## 1. `benchmark_pool`

Use this for method evaluation and prospective validation target selection.

Each target should eventually contain:

- `meta.yaml`
- `wt.fasta`
- optional `homologs.fasta`
- optional structure file or a pointer to a shared structure path

This layer answers:

- can `MarsStack` generalize across protein families
- which proteins are best for freeze / oxidation / desiccation validation

## 2. `evolution_train`

Use this for learning from evolutionary adaptation signals.

This layer is meant for future:

- `ESM` or `ESM-IF` light finetuning
- reranker training
- environment-aware residue prior learning

Each training family should eventually contain:

- `manifest.yaml`
- `positive.fasta`
- `negative.fasta`
- optional `aligned.a3m`
- optional `labels.csv`
- optional structures

This layer answers:

- what sequence or surface patterns are enriched in adapted homologs
- whether we can learn a data-driven Mars objective instead of only hand-coded rules

## 3. Split philosophy

- `benchmark_pool` is the held-out evaluation layer
- `evolution_train` is the learning layer
- do not let close homologs leak from training families into benchmark families
- for future model work, split by family, not only by sequence row

## 4. Current state

- `CALB / 1LBT` is registered as the first ready seed target in `benchmark_pool`
- the other benchmark targets are scaffolded as collection candidates
- the evolutionary training layer is currently a collection plan plus schema

## 5. Validation

Use:

`python mars_stack/scripts/validate_dataset_layout.py`

This checks the CSV schema and verifies file paths for ready targets.
