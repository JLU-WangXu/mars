# Assay Standard

## Goal

Keep stress claims comparable across proteins and across design rounds.

This file is the human-readable companion to `inputs/benchmark/stress_panel.csv`.

## Minimal prospective package

For any transfer protein, the smallest acceptable package is:

1. `EXP01`
2. `BASE01`
3. `LT04`
4. `FT03`
5. `OXH01`

Add `PER01` for Mars-specific chemistry claims.

Add `LYO01` when storage, shipping, or dry-state survival matters.

Add `MGR01` when the claim is low-shear stability, reduced-convection handling, or microgravity readiness.

Add `UV01` or `RAD01` when the claim is explicitly about radiation resilience.

## Gates

Before claiming a design is successful, apply two gates:

1. soluble yield or usable purified material `>= 0.30 x WT`
2. baseline activity or signal `>= 0.50 x WT`

Below those gates, variants may still be interesting mechanistically, but they should be marked as non-promotable.

## Normalization

Use the same normalization rule in every panel:

- matched WT measured in the same batch equals `1.0`
- each variant is reported as `variant / WT`

Do not compare raw values across proteins without this normalization.

## Recommended panel details

### `LT04`

- use `4 C` as the first cross-protein cold panel
- report either direct assay at `4 C` or residual activity after `4 C` incubation
- keep the format consistent within a protein family

### `FT03`

- use `3` freeze-thaw cycles minimum
- use the same buffer and cycle duration for WT and variants
- assay immediately after the final thaw

### `LYO01`

- use one standardized lyophilization and rehydration cycle
- report both recovery of function and whether the sample visibly aggregates

### `MGR01`

- use a matched low-shear clinostat, rotating-wall-vessel, or equivalent reduced-convection setup
- predefine the exposure window, with `24-72 h` as a practical first-pass default
- report residual function together with a matched aggregation, turbidity, or resuspension readout when possible

### `OXH01`

- pick an `H2O2` dose in pilot work so WT is stressed but still measurable
- do not mix dose-escalation discovery with the benchmark run itself

### `UV01`

- use `UV-C` only if it is reproducible in the local setup
- record dose and distance, not just exposure time

### `RAD01`

- use for gamma or X-ray exposure when facility access exists
- report dose in physical units and exposure conditions

### `PER01`

- prefer a titration in `NaClO4` or `Mg(ClO4)2`
- if only one condition is used, choose a sublethal, informative concentration rather than a maximal shock condition

## Reporting format

For each protein and variant, store:

- assay batch id
- WT control id
- normalized score
- raw value
- unit
- stress condition
- replicate count

The benchmark tables in `inputs/benchmark` define what protein should be tested.
This file defines how the results should be reported.
