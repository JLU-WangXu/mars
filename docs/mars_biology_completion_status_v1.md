# Mars Biology Completion Status v1

## 1. Bottom line

The Mars-oriented biological program is **not fully complete**.

What is already strong:

- the computational platform layer (`MARS-FIELD / MarsStack`)
- the benchmark and paper-asset layer
- the first-pass design logic for `Cld`, `AresG`, and `DrwH`

What is still incomplete:

- `Cld` wet-lab closure under Mars-relevant stress proxies
- `AresG` protection-module functional validation
- `DrwH` cargo-cap validation
- full reintegration of Mars-specific readouts into the platform objective layer

So the current honest framing is:

- the **platform** is already at a shareable research-prototype stage
- the **Mars biology task** is still in an early engineering-validation stage

## 2. Cld line: status and evolutionary interpretation

### 2.1 What is already done

- the target has been narrowed to a realistic `(per)chlorate`-associated `Cld` branch rather than a broad superfamily
- a first executable ASR workflow has already been run
- a recommended ancestor panel has already been produced
- structure mapping and oxidation-aware rational-design suggestions already exist
- `Cld` is already represented inside the platform as both:
  - a benchmark target
  - a topic-aware scoring case

### 2.2 What the current evolutionary outputs imply

From the current `ancestor_candidates.tsv` summary:

- `deep:Node012`
  - `leaf_count = 13`
  - `identity_to_seed = 0.915`
- `mid:Node007`
  - `leaf_count = 7`
  - `identity_to_seed = 0.961`

This already supports a practical engineering interpretation:

- `Node007` is the safer “near-native but consensus-improved” ancestor
- `Node012` is the more aggressive “deeper consensus” ancestor that may improve stability but also carries more functional risk

That means the current panel is not redundant. It gives a meaningful gradient:

- native / near-native stability-preserving option
- deeper consensus-stability option

This is exactly what is needed for a first expression-and-activity panel.

### 2.3 What is not finished

The missing step is not more concept discussion. The missing step is a real readout chain:

- soluble expression
- heme loading
- baseline chlorite activity
- `H2O2` challenge residual activity
- low-temperature activity
- dry-rehydration / freeze-thaw retention

Until those are done, the `Cld` line is still a strong engineering-computational route, but not yet a completed Mars-biology result.

## 3. AresG line: status and design interpretation

### 3.1 What is already done

- the project has already moved away from treating `AresR` as the main direction
- Round 1 AF3 already supported the shift toward `AresG`
- Round 2 already exists with three explicit design variants:
  - `AresG2-01_shortlink_shorttail`
  - `AresG2-02_midlink_balanced`
  - `AresG2-03_longlink_longtail`

### 3.2 What the current sequence-level scores imply

Current construct scores show:

- all three variants remain in a similar hydrophobic-moment regime
- increasing tail/linker length raises total length and net charge
- the long version is not just a scaled copy; it shifts composition and low-complexity balance

This means Round 2 is already a real design experiment, not a cosmetic renaming exercise.

A practical interpretation is:

- `shortlink_shorttail` is the compact conservative control
- `midlink_balanced` is the most plausible deployable baseline
- `longlink_longtail` is the high-risk, high-functionality exploration arm

### 3.3 What is not finished

`AresG` is still missing functional validation as a protection module.

The unresolved question is not “does it look interesting on paper?”
It is:

- does one version remain expressible and soluble?
- does one version preserve the core while leaving the tail functionally exposed?
- does one version improve cargo protection under dehydration / low temperature / osmotic stress?

Until that is measured, `AresG` remains a promising protection-module design line, not a completed biological module.

## 4. DrwH line: status and evolutionary interpretation

### 4.1 What is already done

- `DrwH` has already been reframed correctly as a `cargo-cap` / compact protection-domain route
- a WHy-domain family panel has already been assembled
- three ancestor candidates have already been produced:
  - `DrwHAnc-01_local_near_Node046`
  - `DrwHAnc-02_local_mid_Node054`
  - `DrwHAnc-03_deep_ingroup_Node068`
- `AresW` fusion constructs already exist for cap-on-core testing

### 4.2 What the current ancestor property scores imply

The current `drwh_asr_ancestor_candidates.scores.csv` already shows a meaningful spread:

- `local_near_Node046`
  - lower hydrophobicity
  - more negative charge
- `local_mid_Node054`
  - slightly more compact / balanced physicochemical profile
- `deep_ingroup_Node068`
  - highest hydrophobicity signal
  - more consensus-like but also potentially more behaviorally shifted

That means these three ancestors are not interchangeable.

A practical engineering interpretation is:

- `Node046` is the conservative reference
- `Node054` is the likely best first-pass cargo-cap candidate
- `Node068` is the deeper exploratory stability candidate

### 4.3 What is not finished

The missing step is not another abstract cargo-cap story.
The missing step is:

- AF3-level fold/compactness comparison
- expression-friendliness comparison
- cap-on-core geometry screening in `AresW`
- eventual cargo-protection assay

So `DrwH` is already conceptually clear and computationally seeded, but biologically unfinished.

## 5. What “finished” would actually mean

The Mars-oriented biological task should only be called substantially completed if all three conditions hold:

1. `Cld` has a validated ancestor or engineered variant with better Mars-relevant functional resilience
2. `AresG` or `DrwH` demonstrates real protection-module behavior in an assay that matters
3. those biological readouts are fed back into the platform as explicit evaluation logic

Right now:

- condition 1: not yet
- condition 2: not yet
- condition 3: not yet

So the correct statement is:

- the **computational platform is mature enough to support the biology program**
- the **biology program itself is not yet complete**

## 6. Recommended next biological sequence

If the goal is to complete the Mars-biology side rather than just improve the platform, the clean order is:

1. `Cld` expression + activity panel
2. `Cld` oxidative / low-temperature / dry-rehydration panel
3. `AresG2` AF3 + expression narrowing
4. `DrwH` ancestor AF3 + compactness narrowing
5. one protection assay pairing:
   - `Cld + AresG`
   - or `cargo + DrwH / AresW`

That is the shortest path from current computational status to a biologically defensible Mars-oriented result.
