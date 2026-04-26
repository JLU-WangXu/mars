# MARS-FIELD v1 -> Final Roadmap v1

This document converts the current discussion into an executable upgrade list.

Version anchor:

- current stable public state: `MARS-FIELD engineering approximation v1`
- target state: a substantially more learned `MARS-FIELD` in which the neural branch participates in benchmark-time comparison and the remaining engineered branches are progressively replaced by learned modules

## Item 1. Neural reranker into benchmark main comparison path

Priority: `P0`
Status: `substantially_advanced`

Goal:

- stop treating the neural branch as a side experiment
- make it part of the standard benchmark outputs

Current progress:

- neural holdout reranker exists
- benchmark runner can now invoke it and record neural top candidates
- neural outputs are now present in `benchmark_twelvepack`
- neural rerank summaries and holdout outputs are produced per target

Remaining:

- add neural comparison summaries and figures
- decide whether neural should become a default benchmark branch

## Item 2. Retrieval / ancestry from engineered encoding toward learned branches

Priority: `P0`
Status: `in_progress`

Goal:

- move retrieval and ancestry from hand-built evidence tensors toward jointly learned latent branches

Current progress:

- richer retrieval feature channels added to the neural branch
- richer ancestry feature channels added to the neural branch
- ancestry alignment loss added
- retrieval alignment loss added

## Item 3. Neural branch participates in twelvepack comparison

Priority: `P0`
Status: `completed_for_comparison`

Goal:

- ensure neural outputs appear in twelvepack summaries and can be compared target by target

Current progress:

- neural rerank outputs are now recorded in `benchmark_twelvepack`

Remaining:

- decide whether hybrid/default neural benchmark policy should replace the current default

## Item 4. Score semantics unification

Priority: `P0`
Status: `in_progress`

Goal:

- standardize score language across pipeline, benchmark, and paper bundle

Contract:

- `selection_score`: score used to rank candidates in the current branch
- `engineering_score`: `mars_score`
- `neural_score`: optional neural reranker energy

Current progress:

- benchmark summaries now expose neural energy alongside current ranking and engineering scores
- pipeline candidate export has been updated to carry score-contract fields once target outputs are refreshed
- refreshed twelvepack target pipelines now carry `selection_score` and `engineering_score`

## Item 5. Benchmark protocol automation

Priority: `P0`
Status: `completed`

Goal:

- make split policy and run protocol explicit and exportable

Current progress:

- benchmark protocol manifest is now exported as:
  - `benchmark_protocol_manifest.json`
  - `benchmark_protocol_manifest.md`

## Item 6. Neural interpretability outputs

Priority: `P1`
Status: `in_progress`

Goal:

- expose position-level and pairwise-level neural contributions
- make rerank changes diagnosable

Current progress:

- neural reranker now exports gate-weight summaries
- per-position neural gate outputs are written to `neural_site_gates.json`

## Item 7. Environment branch learning

Priority: `P1`
Status: `in_progress`

Goal:

- stop relying mainly on engineered environment weighting
- move toward a learned environment-conditioned branch

Current progress:

- environment vector expanded in the neural dataset
- neural model now applies environment-conditioned modulation to branch representations

## Item 8. Pairwise head training

Priority: `P1`
Status: `in_progress`

Goal:

- turn pairwise energy from an engineering approximation into a learned head with measurable benchmark effect

Current progress:

- pairwise consistency loss added to the neural training path

## Item 9. GitHub release engineering

Priority: `P1`
Status: `completed`

Goal:

- make the repository publishable and honest

Completed outputs:

- `README.md`
- `.gitignore`
- `docs/github_release_manifest_v1.md`
- `docs/mars_field_release_status_v1.md`

## Item 10. Version / run metadata manifests

Priority: `P1`
Status: `in_progress`

Goal:

- record config hashes, split policy, and run settings with every benchmark

Current progress:

- benchmark runs now record:
  - benchmark config hash
  - target config hashes
  - split policy
  - neural rerank settings

## Item 11. Public demo subset

Priority: `P2`
Status: `pending`

Goal:

- prepare a lighter public demo path for GitHub users

Suggested scope:

- one or two targets
- one benchmark mini-config
- one paper-figure mini bundle

## Suggested Execution Order

1. score semantics unification
2. benchmark protocol automation
3. neural reranker into benchmark main comparison path
4. twelvepack neural comparison summaries
5. neural interpretability outputs
6. retrieval / ancestry learned-branch upgrades
7. pairwise head training
8. environment branch learning
9. version manifests completion
10. public demo subset

## Latest note

Current benchmark policy tracks now include:

- `current`
- `hybrid`
- `neural-default`

The next decision point is not whether neural can be run, but whether its policy behavior is strong enough to replace the current default path.
