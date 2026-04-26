#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${MARS_ENV_NAME:-mars-field-gpu}"
CUDA_VARIANT="${MARS_CUDA_VARIANT:-12.1}"

pick_manager() {
  if command -v micromamba >/dev/null 2>&1; then
    echo "micromamba"
    return
  fi
  if command -v mamba >/dev/null 2>&1; then
    echo "mamba"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    echo "conda"
    return
  fi
  return 1
}

MANAGER="$(pick_manager || true)"
if [[ -z "${MANAGER}" ]]; then
  echo "No conda-compatible environment manager found. Install conda, mamba, or micromamba first." >&2
  exit 1
fi

echo "[bootstrap] repo root: ${ROOT_DIR}"
echo "[bootstrap] manager: ${MANAGER}"
echo "[bootstrap] env name: ${ENV_NAME}"
echo "[bootstrap] cuda target: ${CUDA_VARIANT}"

if "${MANAGER}" env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  "${MANAGER}" env update -n "${ENV_NAME}" -f "${ROOT_DIR}/environment.linux-gpu.yml" --prune
else
  "${MANAGER}" env create -n "${ENV_NAME}" -f "${ROOT_DIR}/environment.linux-gpu.yml"
fi

if [[ "${CUDA_VARIANT}" != "12.1" ]]; then
  "${MANAGER}" run -n "${ENV_NAME}" conda install -y -c pytorch -c nvidia "pytorch-cuda=${CUDA_VARIANT}"
fi

"${MANAGER}" run -n "${ENV_NAME}" python -m pip install -U pip
"${MANAGER}" run -n "${ENV_NAME}" python -m pip install -r "${ROOT_DIR}/requirements.txt"

for required_path in \
  "${ROOT_DIR}/vendors/ProteinMPNN/protein_mpnn_run.py" \
  "${ROOT_DIR}/vendors/esm-main" \
  "${ROOT_DIR}/scripts/run_mars_pipeline.py" \
  "${ROOT_DIR}/scripts/run_mars_autodesign.py"
do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

echo
echo "[bootstrap] runtime check"
"${MANAGER}" run -n "${ENV_NAME}" python "${ROOT_DIR}/scripts/check_mars_runtime.py"
echo
echo "[bootstrap] next commands"
echo "cd ${ROOT_DIR}"
echo "${MANAGER} run -n ${ENV_NAME} python scripts/run_mars_autodesign.py analyze --pdb inputs/tem1_1btl/1BTL.pdb"
echo "${MANAGER} run -n ${ENV_NAME} python scripts/run_mars_pipeline.py --config configs/tem1_1btl.yaml --top-k 12"
echo "${MANAGER} run -n ${ENV_NAME} python scripts/run_mars_benchmark.py --benchmark-config configs/benchmark_triplet.yaml --top-k 12"
