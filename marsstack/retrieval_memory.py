from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gemmi
import yaml

from .structure_features import analyze_structure


@dataclass
class StructureMemoryEntry:
    target: str
    chain: str
    position: int
    residue: str
    descriptor: list[float]


@dataclass
class StructureMotifPrototype:
    prototype_id: str
    centroid: list[float]
    residue_distribution: dict[str, float]
    support_count: int
    support_targets: list[str]
    member_positions: list[str]


def _preprocess_pdb(
    src_path: Path,
    dst_path: Path,
    residue_renames: list[dict[str, Any]] | None = None,
) -> Path:
    if not residue_renames:
        dst_path.write_bytes(src_path.read_bytes())
        return dst_path

    rename_map: dict[tuple[str, int, str | None], str] = {}
    for item in residue_renames:
        chain = str(item["chain"])
        residue_number = int(item["residue_number"])
        from_name = str(item["from_name"]).upper() if item.get("from_name") else None
        to_name = str(item["to_name"]).upper()
        rename_map[(chain, residue_number, from_name)] = to_name

    out_lines: list[str] = []
    for line in src_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            chain = line[21].strip()
            try:
                residue_number = int(line[22:26].strip())
            except ValueError:
                residue_number = None
            resname = line[17:20].strip().upper()
            replacement = None
            if residue_number is not None:
                replacement = rename_map.get((chain, residue_number, resname))
                if replacement is None:
                    replacement = rename_map.get((chain, residue_number, None))
            if replacement is not None:
                line = "ATOM  " + line[6:]
                line = line[:17] + f"{replacement:>3}" + line[20:]
        out_lines.append(line)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return dst_path


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _angle(v1: tuple[float, float, float], v2: tuple[float, float, float]) -> float:
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    dot = x1 * x2 + y1 * y2 + z1 * z2
    norm1 = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
    norm2 = math.sqrt(x2 * x2 + y2 * y2 + z2 * z2)
    if norm1 <= 1e-8 or norm2 <= 1e-8:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.acos(cosine)


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _load_structure_targets(project_root: Path) -> list[dict[str, Any]]:
    targets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for config_path in sorted((project_root / "configs").glob("*.yaml")):
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        protein = cfg.get("protein", {})
        generation = cfg.get("generation", {})
        pdb_path = protein.get("pdb_path")
        chain = protein.get("chain")
        if not protein or not generation or not pdb_path or not chain:
            continue
        abs_pdb = (project_root / pdb_path).resolve() if not Path(pdb_path).is_absolute() else Path(pdb_path)
        if not abs_pdb.exists():
            continue
        key = (str(protein.get("name", abs_pdb.stem)), str(abs_pdb), str(chain))
        targets[key] = {
            "target": key[0],
            "pdb_path": abs_pdb,
            "chain": str(chain),
            "protected_positions": set(int(x) for x in protein.get("protected_positions", [])),
            "preprocess": protein.get("preprocess", {}) or {},
        }
    return list(targets.values())


def _extract_ca_trace(pdb_path: Path, chain_id: str) -> dict[int, tuple[float, float, float]]:
    st = gemmi.read_structure(str(pdb_path))
    trace: dict[int, tuple[float, float, float]] = {}
    for residue in st[0][chain_id]:
        atom = residue.find_atom("CA", "\0")
        if atom:
            trace[int(residue.seqid.num)] = (atom.pos.x, atom.pos.y, atom.pos.z)
    return trace


def _build_chain_descriptors(
    pdb_path: Path,
    chain_id: str,
    protected_positions: set[int],
) -> list[StructureMemoryEntry]:
    features = analyze_structure(
        pdb_path=pdb_path,
        chain_id=chain_id,
        protected_positions=protected_positions,
    )
    trace = _extract_ca_trace(pdb_path, chain_id)
    feature_positions = [feat.num for feat in features if feat.num in trace]
    contact_cutoffs = (8.0, 12.0)
    entries: list[StructureMemoryEntry] = []

    for idx, feat in enumerate(features):
        if feat.num not in trace:
            continue
        pos = trace[feat.num]
        prev1 = trace.get(feature_positions[idx - 1]) if idx - 1 >= 0 and idx - 1 < len(feature_positions) else None
        next1 = trace.get(feature_positions[idx + 1]) if idx + 1 < len(feature_positions) else None
        prev2 = trace.get(feature_positions[idx - 2]) if idx - 2 >= 0 and idx - 2 < len(feature_positions) else None
        next2 = trace.get(feature_positions[idx + 2]) if idx + 2 < len(feature_positions) else None

        sequential_distances = [
            _distance(pos, prev1) / 10.0 if prev1 is not None else 0.0,
            _distance(pos, next1) / 10.0 if next1 is not None else 0.0,
            _distance(pos, prev2) / 10.0 if prev2 is not None else 0.0,
            _distance(pos, next2) / 10.0 if next2 is not None else 0.0,
        ]

        angle_1 = (
            _angle((prev1[0] - pos[0], prev1[1] - pos[1], prev1[2] - pos[2]), (next1[0] - pos[0], next1[1] - pos[1], next1[2] - pos[2])) / math.pi
            if prev1 is not None and next1 is not None
            else 0.5
        )
        angle_2 = (
            _angle((prev2[0] - pos[0], prev2[1] - pos[1], prev2[2] - pos[2]), (next2[0] - pos[0], next2[1] - pos[1], next2[2] - pos[2])) / math.pi
            if prev2 is not None and next2 is not None
            else 0.5
        )

        contact_counts = []
        for cutoff in contact_cutoffs:
            count = 0
            for other_pos_num in feature_positions:
                if other_pos_num == feat.num:
                    continue
                if _distance(pos, trace[other_pos_num]) <= cutoff:
                    count += 1
            contact_counts.append(count / 20.0)

        descriptor = [
            feat.sasa / 120.0,
            feat.mean_b / 80.0,
            feat.min_dist_protected / 20.0,
            1.0 if feat.in_disulfide else 0.0,
            1.0 if feat.glyco_motif else 0.0,
            *sequential_distances,
            angle_1,
            angle_2,
            *contact_counts,
        ]
        entries.append(
            StructureMemoryEntry(
                target=pdb_path.stem,
                chain=chain_id,
                position=feat.num,
                residue=feat.aa,
                descriptor=descriptor,
            )
        )
    return entries


def _cache_path(project_root: Path) -> Path:
    return project_root / ".cache" / "structure_memory_atlas_v1.json"


def build_structure_memory_bank(project_root: Path) -> list[StructureMemoryEntry]:
    cache_path = _cache_path(project_root)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return [
            StructureMemoryEntry(
                target=str(item["target"]),
                chain=str(item["chain"]),
                position=int(item["position"]),
                residue=str(item["residue"]),
                descriptor=[float(x) for x in item["descriptor"]],
            )
            for item in payload
        ]

    entries: list[StructureMemoryEntry] = []
    for target_info in _load_structure_targets(project_root):
        working_pdb = target_info["pdb_path"]
        preprocess_cfg = dict(target_info.get("preprocess", {}) or {})
        residue_renames = preprocess_cfg.get("residue_renames")
        if residue_renames:
            working_pdb = _preprocess_pdb(
                src_path=target_info["pdb_path"],
                dst_path=project_root / ".cache" / "retrieval_preprocessed" / f"{target_info['target']}_{working_pdb.name}",
                residue_renames=residue_renames,
            )
        entries.extend(
            _build_chain_descriptors(
                pdb_path=working_pdb,
                chain_id=target_info["chain"],
                protected_positions=target_info["protected_positions"],
            )
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps([asdict(entry) for entry in entries], indent=2),
        encoding="utf-8",
    )
    return entries


def _vector_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def build_structure_motif_atlas(
    entries: list[StructureMemoryEntry],
    cluster_radius: float = 0.42,
    min_cluster_size: int = 2,
) -> list[StructureMotifPrototype]:
    clusters: list[dict[str, Any]] = []

    for entry in entries:
        best_idx = None
        best_distance = float("inf")
        for idx, cluster in enumerate(clusters):
            distance = _vector_distance(entry.descriptor, cluster["centroid"])
            if distance < best_distance:
                best_distance = distance
                best_idx = idx

        if best_idx is not None and best_distance <= float(cluster_radius):
            cluster = clusters[best_idx]
            cluster["members"].append(entry)
            member_count = len(cluster["members"])
            cluster["centroid"] = [
                (old * (member_count - 1) + new) / member_count
                for old, new in zip(cluster["centroid"], entry.descriptor)
            ]
        else:
            clusters.append(
                {
                    "centroid": list(entry.descriptor),
                    "members": [entry],
                }
            )

    prototypes: list[StructureMotifPrototype] = []
    for idx, cluster in enumerate(clusters, start=1):
        members: list[StructureMemoryEntry] = cluster["members"]
        if len(members) < int(min_cluster_size):
            continue
        residue_counts: dict[str, int] = {}
        support_targets = sorted({member.target for member in members})
        member_positions = [f"{member.target}:{member.position}:{member.residue}" for member in members[:24]]
        for member in members:
            residue_counts[member.residue] = residue_counts.get(member.residue, 0) + 1
        total = float(sum(residue_counts.values()))
        residue_distribution = {
            aa: round(count / total, 6)
            for aa, count in sorted(residue_counts.items(), key=lambda item: (-item[1], item[0]))
        }
        prototypes.append(
            StructureMotifPrototype(
                prototype_id=f"motif_{idx:03d}",
                centroid=[round(float(x), 6) for x in cluster["centroid"]],
                residue_distribution=residue_distribution,
                support_count=len(members),
                support_targets=support_targets,
                member_positions=member_positions,
            )
        )
    return prototypes


def serialize_motif_atlas(atlas: list[StructureMotifPrototype]) -> list[dict[str, Any]]:
    return [
        {
            "prototype_id": item.prototype_id,
            "centroid": item.centroid,
            "residue_distribution": item.residue_distribution,
            "support_count": item.support_count,
            "support_targets": item.support_targets,
            "member_positions": item.member_positions,
        }
        for item in atlas
    ]


def retrieve_residue_memory(
    target: str,
    pdb_path: Path,
    chain_id: str,
    protected_positions: set[int],
    design_positions: list[int],
    outputs_root: Path,
    top_k: int = 3,
    max_neighbors: int = 24,
) -> tuple[dict[int, dict[str, float]], dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    project_root = outputs_root.parent
    bank = [entry for entry in build_structure_memory_bank(project_root) if entry.target != target]
    atlas = build_structure_motif_atlas(
        bank,
        cluster_radius=0.42,
        min_cluster_size=2,
    )
    query_entries = _build_chain_descriptors(
        pdb_path=pdb_path,
        chain_id=chain_id,
        protected_positions=protected_positions,
    )
    query_map = {entry.position: entry for entry in query_entries}

    recommendations: dict[int, dict[str, float]] = {}
    diagnostics: dict[int, list[dict[str, Any]]] = {}

    for position in design_positions:
        if position not in query_map:
            continue
        query_entry = query_map[position]
        prototypes = sorted(
            atlas,
            key=lambda item: _vector_distance(query_entry.descriptor, item.centroid),
        )[: int(max_neighbors)]

        aa_scores: dict[str, float] = {}
        neighbor_payload: list[dict[str, Any]] = []
        for prototype in prototypes:
            distance = _vector_distance(query_entry.descriptor, prototype.centroid)
            similarity = math.exp(-(distance**2) / 0.30)
            support_weight = min(1.75, 0.55 + math.log1p(prototype.support_count))
            diversity_weight = min(1.5, 0.65 + 0.18 * len(prototype.support_targets))
            proto_weight = similarity * support_weight * diversity_weight
            for residue, prob in prototype.residue_distribution.items():
                aa_scores[residue] = aa_scores.get(residue, 0.0) + proto_weight * float(prob)
            neighbor_payload.append(
                {
                    "prototype_id": prototype.prototype_id,
                    "support_count": prototype.support_count,
                    "support_targets": prototype.support_targets,
                    "residue_distribution": prototype.residue_distribution,
                    "distance": round(distance, 6),
                    "similarity": round(similarity, 6),
                    "support_weight": round(support_weight, 6),
                    "diversity_weight": round(diversity_weight, 6),
                    "weight": round(proto_weight, 6),
                    "source": "motif_atlas",
                }
            )

        if aa_scores:
            ranked = sorted(aa_scores.items(), key=lambda item: (-item[1], item[0]))[: int(top_k)]
            recommendations[int(position)] = {aa: round(score, 6) for aa, score in ranked}
            diagnostics[int(position)] = neighbor_payload

    return recommendations, diagnostics, serialize_motif_atlas(atlas)
