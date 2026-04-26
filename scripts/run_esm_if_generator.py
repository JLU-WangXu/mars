from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gemmi
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ESM_ROOT = ROOT / "vendors" / "esm-main"
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def resolve_checkpoint_path(requested_path: Path, torch_home: Path) -> Path | None:
    candidates = [
        requested_path.resolve(),
        (torch_home / "hub" / "checkpoints" / requested_path.name).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_single_chain_coords(pdbfile: Path, chain_id: str) -> tuple[np.ndarray, str]:
    st = gemmi.read_structure(str(pdbfile))
    chain = st[0][chain_id]
    coords: list[list[list[float]]] = []
    seq: list[str] = []
    for res in chain:
        if res.entity_type != gemmi.EntityType.Polymer:
            continue
        if res.name not in AA3_TO_1:
            continue
        atom_map = {atom.name.strip(): atom for atom in res}
        if not {"N", "CA", "C"}.issubset(atom_map):
            continue
        coords.append(
            [
                [atom_map["N"].pos.x, atom_map["N"].pos.y, atom_map["N"].pos.z],
                [atom_map["CA"].pos.x, atom_map["CA"].pos.y, atom_map["CA"].pos.z],
                [atom_map["C"].pos.x, atom_map["C"].pos.y, atom_map["C"].pos.z],
            ]
        )
        seq.append(AA3_TO_1[res.name])
    if not coords:
        raise ValueError(f"No usable polymer backbone coordinates found for chain {chain_id} in {pdbfile}")
    return np.asarray(coords, dtype=np.float32), "".join(seq)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdbfile", type=Path, required=True)
    parser.add_argument("--chain", type=str, required=True)
    parser.add_argument("--outpath", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1e-6)
    parser.add_argument("--esm-root", type=Path, default=ESM_ROOT)
    parser.add_argument("--model-checkpoint", type=Path, default=ROOT / ".cache" / "esm_if1_gvp4_t16_142M_UR50.pt")
    parser.add_argument("--multichain-backbone", action="store_true")
    parser.add_argument("--nogpu", action="store_true")
    args = parser.parse_args()

    esm_root = args.esm_root.resolve()
    if not esm_root.exists():
        raise FileNotFoundError(f"ESM root not found: {esm_root}")

    torch_home = ROOT / ".cache" / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_home))

    sys.path.insert(0, str(esm_root))

    import torch
    import esm

    checkpoint_path = resolve_checkpoint_path(args.model_checkpoint, torch_home)
    if checkpoint_path is not None:
        model, alphabet = esm.pretrained.load_model_and_alphabet(str(checkpoint_path))
    else:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    use_cuda = torch.cuda.is_available() and not args.nogpu
    if use_cuda:
        model = model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.multichain_backbone:
        raise NotImplementedError("Multichain ESM-IF mode is not yet wired in the biotite-free runner.")

    coords, native_seq = load_single_chain_coords(args.pdbfile, args.chain)

    args.outpath.parent.mkdir(parents=True, exist_ok=True)
    with args.outpath.open("w", encoding="utf-8") as fh:
        fh.write(f">native chain={args.chain}\n{native_seq}\n")
        for i in range(args.num_samples):
            sampled_seq = model.sample(coords, temperature=args.temperature, device=device)
            recovery = float(np.mean([a == b for a, b in zip(native_seq, sampled_seq)]))
            fh.write(f">sample={i+1} chain={args.chain} temperature={args.temperature} recovery={recovery:.4f}\n")
            fh.write(sampled_seq + "\n")


if __name__ == "__main__":
    main()
