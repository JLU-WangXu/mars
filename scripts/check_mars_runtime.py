from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def tool_path(name: str) -> str:
    return shutil.which(name) or ""


def file_status(path: Path) -> str:
    return "exists" if path.exists() else "missing"


def branch_status() -> dict[str, object]:
    proteinmpnn_root = ROOT / "vendors" / "ProteinMPNN"
    esm_root = ROOT / "vendors" / "esm-main"
    checkpoint = ROOT / ".cache" / "esm_if1_gvp4_t16_142M_UR50.pt"
    proteinmpnn_available = (
        proteinmpnn_root.exists()
        and (proteinmpnn_root / "protein_mpnn_run.py").exists()
        and (proteinmpnn_root / "helper_scripts" / "parse_multiple_chains.py").exists()
    )
    esm_if_available = esm_root.exists() and has_module("torch") and has_module("gemmi")
    neural_corpus_count = 0
    try:
        from marsstack.field_network.neural_dataset import load_neural_corpus

        neural_corpus_count = len(load_neural_corpus(ROOT / "outputs"))
    except Exception:
        neural_corpus_count = 0
    return {
        "proteinmpnn_available": proteinmpnn_available,
        "esm_root": str(esm_root),
        "esm_checkpoint": file_status(checkpoint),
        "esm_if_available": esm_if_available,
        "neural_available": has_module("torch") and neural_corpus_count > 0,
        "neural_corpus_count": neural_corpus_count,
    }


def main() -> int:
    print("MARS-FIELD runtime check")
    print(f"repo_root: {ROOT}")
    print(f"python: {sys.executable}")
    print()

    print("[python modules]")
    for module in ["pandas", "yaml", "numpy", "matplotlib", "docx", "gemmi"]:
        print(f"- {module}: {'ok' if has_module(module) else 'missing'}")
    print()

    print("[optional tools]")
    for tool in ["mafft", "iqtree2", "python", "powershell"]:
        path = tool_path(tool)
        print(f"- {tool}: {path if path else 'not found in PATH'}")
    print()

    print("[vendor paths]")
    for rel in [
        "vendors/ProteinMPNN",
        "vendors/ProteinMPNN/protein_mpnn_run.py",
        "vendors/esm-main",
        ".cache/esm_if1_gvp4_t16_142M_UR50.pt",
    ]:
        path = (ROOT / rel).resolve()
        print(f"- {path}: {file_status(path)}")
    print()

    print("[important paths]")
    for rel in [
        "configs/benchmark_twelvepack_final.yaml",
        "scripts/run_mars_autodesign.py",
        "scripts/run_mars_pipeline.py",
        "scripts/run_mars_benchmark.py",
        "outputs/paper_bundle_v1",
        "../designs/asr_cld_prb/run3/cld_ancestor_recommended_panel.fasta",
    ]:
        path = (ROOT / rel).resolve()
        print(f"- {path}: {'exists' if path.exists() else 'missing'}")
    print()

    print("[branch usability]")
    status = branch_status()
    for key, value in status.items():
        print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
