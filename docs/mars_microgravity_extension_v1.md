# MGR01 and Microgravity Topic Extension v1

## Why this exists

`microgravity` should not be promoted to the mandatory MarsStack core panel.

It is better treated as an optional environment extension because the dominant risks are usually process-level:

- reduced convection
- low shear
- altered sedimentation
- aggregation or phase-behavior changes during long incubations

That means the first useful engineering proxy is not "space biology in full".
It is:

**retain function while avoiding sticky surface behavior under reduced-convection, low-shear handling.**

## `MGR01` panel contract

Recommended assay:

- low-shear clinostat
- rotating-wall-vessel exposure
- or another matched reduced-convection setup

Default first-pass condition:

- `24-72 h` exposure
- then return samples to the standard assay
- normalize `variant / WT` under the same treatment batch

Recommended paired readouts:

- residual activity or signal
- matched aggregation, turbidity, or resuspension score

`MGR01` is an optional weighted panel.
It is not part of the mandatory transfer core.

## `microgravity` topic scorer

The topic scorer is a rule-based proxy module that fits directly into the existing `topic_sequence`, `topic_structure`, and `topic_evolution` contract.

### `topic_sequence`

The sequence term rewards:

- moderate global polarity
- moderate absolute net charge

This is meant to bias candidates away from obviously sticky low-solvation compositions without forcing all proteins toward the same chemistry extreme.

### `topic_structure`

The structure term rewards:

- polar or charged substitutions at highly exposed positions
- damping highly exposed flexible patches with solvating residues
- keeping buried positions core-compatible

The structure term penalizes:

- sticky exposed hydrophobes and aromatics
- flexible exposed patches that remain aggregation-prone
- buried core-breaking substitutions

### `topic_evolution`

The evolution term stays conservative:

- use homolog, ASR, and family priors only on the mutated positions
- treat the microgravity extension as a biasing overlay, not a new global training target

## Best first uses

The first places where this extension is most defensible are:

- `AresG`-like protection modules
- `DrwH`-like cargo-cap or compact protection domains
- soluble enzymes or reporters where long low-shear incubation could expose aggregation risk

It is less appropriate as the primary story for the main `Cld` catalytic case.

## Configuration entry point

Use:

- `configs/topic_templates/microgravity_topic_template.yaml`

Set:

- `topic.name: "microgravity"`

This keeps the module in the same interface family as `cld`, `drwh`, and `aresg` without changing the main MarsStack contract.
