# Cross-Machine Setup Guide v1

This repository can already be shared as a research prototype, but it is not a zero-configuration production package.

## 1. What works out of the box

- reading the repository and docs
- browsing existing benchmark outputs and paper assets
- generating the HTML delivery package
- running Python-only utilities that only depend on bundled inputs/outputs

## 2. Base Python environment

Recommended base flow:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Core Python packages:

- `pandas`
- `PyYAML`
- `numpy`
- `matplotlib`
- `python-docx`
- `gemmi`

## 3. Optional / external components

Some branches still depend on local or external tools:

- `ProteinMPNN`
- `ESM-IF`
- `torch` / `esm`
- `iqtree2`
- `mafft`
- AF3 or other external structure prediction services

These are not required for reading the repository or browsing the current outputs, but they are required for selected generation and ASR flows.

## 3.5 Recommended Linux GPU path

If you are targeting a Linux GPU server, prefer:

```bash
bash scripts/bootstrap_linux_gpu.sh
```

This uses:

- `environment.linux-gpu.yml`
- `requirements.txt`
- `scripts/check_mars_runtime.py`

Default CUDA target is `12.1`. Override if needed:

```bash
export MARS_CUDA_VARIANT=12.4
bash scripts/bootstrap_linux_gpu.sh
```

## 4. Important repository behavior

### ESM-IF interpreter

The target configs no longer require a hard-coded local Python path.

If `generation.esm_if.python_executable` is omitted, `run_mars_pipeline.py` now falls back to the current `sys.executable`.

### Relative topic paths

`CLD_3Q09_TOPIC` and `CLD_3Q09_NOTOPIC` now reference the ASR files through relative paths (`../designs/...`) instead of machine-specific `F:/...` paths.

## 5. Recommended first commands

From the repository root:

```bash
cd mars_stack
python scripts/validate_dataset_layout.py
python scripts/run_mars_pipeline.py --config configs/tem1_1btl.yaml --top-k 12
python scripts/run_mars_benchmark.py --benchmark-config configs/benchmark_triplet.yaml --top-k 12
python scripts/run_mars_autodesign.py analyze --pdb inputs/tem1_1btl/1BTL.pdb
```

## 6. Honest public status

Describe the repository as:

- a research prototype
- a benchmarked engineering approximation of `MARS-FIELD`
- a codebase that already implements a field-style engineering workflow

Do not describe it as:

- a finished fully neural end-to-end field model
- a production package
- a final fully joint generator-decoder-field model
