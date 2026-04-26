# GitHub Upload Instructions

Use this folder directly as the repository root:

`F:\4-15Marsprotein\release_packages\mars_field_code_snapshot_20260426_release_ready`

## Suggested repository form values

Repository name:

`mars-field`

Description:

`A benchmarked protein engineering research prototype that integrates structural, evolutionary, ancestral, retrieval-based, and environment-conditioned evidence into a shared residue decision field.`

Visibility:

- Public if you want paper / GitHub exposure
- Private if you are still iterating before public release

README:

- Do not add a GitHub-generated README
- This folder already contains the final README

.gitignore:

- Do not add a GitHub-generated .gitignore
- This folder already contains `.gitignore`

License:

- Add only if you have decided the project license
- If undecided, leave blank for now

## Upload sequence

1. Create an empty repository on GitHub
2. Do not initialize it with README / .gitignore / license
3. Upload the contents of this folder as the repository root
4. Confirm that `README.md` renders correctly on GitHub
5. Confirm that `requirements.txt`, `environment.linux-gpu.yml`, `scripts/`, `configs/`, `docs/`, and `inputs/` are present

## What this package already includes

- final root README
- Linux GPU environment spec
- bootstrap script
- runtime checker
- raw `PDB -> analyze/design` entrypoint
- config-driven pipeline
- docs for Mars gap analysis and cross-machine setup

## Honest public framing

Use language like:

- `MARS-FIELD is a benchmarked protein engineering research prototype`
- `engineering approximation v1`
- `unified evidence-to-sequence workflow`

Avoid language like:

- `fully finished production package`
- `fully joint final neural field model`
