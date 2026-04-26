from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np


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

VDW = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8}


@dataclass
class ResidueFeature:
    num: int
    name: str
    aa: str
    sasa: float
    mean_b: float
    min_dist_protected: float
    in_disulfide: bool
    glyco_motif: bool


def _fibonacci_sphere(n_points: int = 64) -> np.ndarray:
    pts: list[tuple[float, float, float]] = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_points):
        y = 1 - (i / (n_points - 1)) * 2
        r = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        pts.append((math.cos(theta) * r, y, math.sin(theta) * r))
    return np.array(pts, dtype=float)


def _extract_disulfides(pdb_path: Path, chain: str) -> set[int]:
    nums: set[int] = set()
    for line in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("SSBOND"):
            continue
        c1 = line[15].strip()
        c2 = line[29].strip()
        if c1 == chain:
            nums.add(int(line[17:21].strip()))
        if c2 == chain:
            nums.add(int(line[31:35].strip()))
    return nums


def _sequence_from_chain(chain_obj: gemmi.Chain) -> list[tuple[int, str]]:
    seq = []
    for res in chain_obj:
        if res.entity_type != gemmi.EntityType.Polymer:
            continue
        if res.name not in AA3_TO_1:
            continue
        seq.append((res.seqid.num, AA3_TO_1[res.name]))
    return seq


def analyze_structure(
    pdb_path: Path,
    chain_id: str,
    protected_positions: set[int],
    probe_radius: float = 1.4,
    sasa_points: int = 64,
) -> list[ResidueFeature]:
    st = gemmi.read_structure(str(pdb_path))
    chain = st[0][chain_id]
    seq = _sequence_from_chain(chain)
    seq_map = {num: aa for num, aa in seq}
    glyco_sites = {
        num
        for idx, (num, aa) in enumerate(seq)
        if aa == "N"
        and idx + 2 < len(seq)
        and seq[idx + 1][1] != "P"
        and seq[idx + 2][1] in {"S", "T"}
    }
    disulfides = _extract_disulfides(pdb_path, chain_id)
    sphere = _fibonacci_sphere(sasa_points)

    atoms = []
    residues = []
    for res in chain:
        if res.entity_type != gemmi.EntityType.Polymer:
            continue
        if res.name not in AA3_TO_1:
            continue
        entry = {"num": res.seqid.num, "name": res.name, "atoms": [], "bf": []}
        residues.append(entry)
        for atom in res:
            if atom.element.name == "H":
                continue
            pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
            atoms.append(
                {
                    "res_i": len(residues) - 1,
                    "pos": pos,
                    "rad": VDW.get(atom.element.name, 1.7),
                }
            )
            entry["atoms"].append(len(atoms) - 1)
            entry["bf"].append(atom.b_iso)

    all_pos = np.array([a["pos"] for a in atoms], dtype=float)
    all_rad = np.array([a["rad"] for a in atoms], dtype=float)
    protected_atom_pos = np.array(
        [atoms[i]["pos"] for r in residues if r["num"] in protected_positions for i in r["atoms"]],
        dtype=float,
    )

    per_res_sasa = [0.0] * len(residues)
    for i, atom in enumerate(atoms):
        probe = atom["rad"] + probe_radius
        cutoff2 = (probe + all_rad + probe_radius + 0.01) ** 2
        d2 = np.sum((all_pos - atom["pos"]) ** 2, axis=1)
        neigh = np.where(d2 < cutoff2)[0]
        neigh = neigh[neigh != i]
        exposed = 0
        for unit in sphere:
            sp = atom["pos"] + unit * probe
            occluded = False
            for j in neigh:
                if np.sum((all_pos[j] - sp) ** 2) < (all_rad[j] + probe_radius) ** 2:
                    occluded = True
                    break
            if not occluded:
                exposed += 1
        area = 4 * math.pi * (probe**2) * (exposed / sasa_points)
        per_res_sasa[atom["res_i"]] += area

    result: list[ResidueFeature] = []
    for i, res in enumerate(residues):
        min_dist = 999.0
        if len(protected_atom_pos) > 0:
            for atom_i in res["atoms"]:
                d = np.sqrt(np.min(np.sum((protected_atom_pos - atoms[atom_i]["pos"]) ** 2, axis=1)))
                min_dist = min(min_dist, float(d))
        result.append(
            ResidueFeature(
                num=res["num"],
                name=res["name"],
                aa=AA3_TO_1[res["name"]],
                sasa=round(per_res_sasa[i], 3),
                mean_b=round(sum(res["bf"]) / len(res["bf"]), 3),
                min_dist_protected=round(min_dist, 3),
                in_disulfide=res["num"] in disulfides,
                glyco_motif=res["num"] in glyco_sites,
            )
        )
    return result


def detect_oxidation_hotspots(
    features: list[ResidueFeature],
    min_sasa: float,
    min_dist_protected: float,
) -> list[int]:
    hotspots = []
    for feat in features:
        if feat.min_dist_protected < min_dist_protected:
            continue
        if feat.aa == "C" and feat.in_disulfide:
            continue
        if feat.aa in {"M", "W", "Y", "H", "C"} and feat.sasa >= min_sasa:
            hotspots.append(feat.num)
    return hotspots


def detect_flexible_surface_positions(
    features: list[ResidueFeature],
    min_sasa: float,
    b_percentile: float = 85.0,
) -> list[int]:
    bvals = np.array([f.mean_b for f in features], dtype=float)
    thresh = float(np.percentile(bvals, b_percentile))
    return [f.num for f in features if f.sasa >= min_sasa and f.mean_b >= thresh]
