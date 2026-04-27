from __future__ import annotations

import json
import math
import heapq
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import yaml

from .structure_features import analyze_structure


# =============================================================================
# Thermostability-Specific Data Structures
# =============================================================================

@dataclass
class ThermostabilityProfile:
    """Thermal stability profile for a protein entry."""
    optimal_temperature: float  # Optimal growth/activity temperature (Celsius)
    melting_temperature: float  # Tm value if available
    stability_score: float  # Normalized stability score [0, 1]
    source_organism: str  # Source organism for thermophilic context
    is_thermophilic: bool  # Flag for thermophilic proteins
    proline_content: float  # Proline fraction - stabilizes loops
    arginine_content: float  # Arginine content - salt bridges
    disulfide_count: int  # Number of disulfide bonds
    surface_hydrophobicity: float  # Packing efficiency indicator


@dataclass
class FunctionalAnnotation:
    """Functional annotation for protein entries."""
    enzyme_class: str  # EC number or enzyme classification
    go_terms: list[str]  # Gene Ontology terms
    active_site_positions: list[int]  # Known active site residues
    binding_motifs: list[str]  # Known binding motifs


@dataclass
class SequenceSignature:
    """Sequence-based signature for similarity computation."""
    residue_vector: list[float]  # Normalized amino acid composition (20-dim)
    dipeptide_vector: list[float]  # Dipeptide frequencies
    tripeptide_vector: list[float]  # Tripeptide frequencies
    sequence_length: int


@dataclass
class StructureMemoryEntry:
    target: str
    chain: str
    position: int
    residue: str
    descriptor: list[float]
    thermostability: ThermostabilityProfile | None = None
    functional: FunctionalAnnotation | None = None
    sequence_signature: SequenceSignature | None = None
    thermostability_boost: float = 1.0  # Boost factor for thermostability matches


@dataclass
class StructureMotifPrototype:
    prototype_id: str
    centroid: list[float]
    residue_distribution: dict[str, float]
    support_count: int
    support_targets: list[str]
    member_positions: list[str]


@dataclass
class ThermostabilityIndex:
    """Dedicated index for thermostability-related protein retrieval."""
    thermostable_entries: list[StructureMemoryEntry] = field(default_factory=list)
    stability_scores: dict[str, float] = field(default_factory=dict)  # target -> stability
    thermal_organisms: set[str] = field(default_factory=set)  # Thermophilic sources
    stability_buckets: dict[int, list[StructureMemoryEntry]] = field(default_factory=dict)  # Tm-based buckets
    spatial_index: dict[str, list[int]] = field(default_factory=dict)  # Position-based index


@dataclass
class ActiveLearningState:
    """State for active learning-based retrieval."""
    uncertainty_scores: dict[int, float] = field(default_factory=dict)  # position -> uncertainty
    queried_positions: set[int] = field(default_factory=set)
    retrieved_count: dict[int, int] = field(default_factory=dict)  # position -> count
    confidence_thresholds: dict[int, float] = field(default_factory=dict)


# =============================================================================
# Similarity Metrics
# =============================================================================

@dataclass
class SimilarityScores:
    """Combined similarity scores from multiple metrics."""
    structural_similarity: float
    sequence_similarity: float
    functional_similarity: float
    thermostability_boost: float
    combined_score: float


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """Compute normalized Euclidean distance."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _gaussian_kernel(distance: float, sigma: float = 0.3) -> float:
    """Gaussian kernel for similarity transformation."""
    return math.exp(-(distance ** 2) / (2 * sigma ** 2))


def _compute_structural_similarity(
    query_descriptor: list[float],
    prototype_centroid: list[float],
) -> float:
    """Compute structural similarity based on descriptor vectors."""
    distance = _euclidean_distance(query_descriptor, prototype_centroid)
    # Scale distance to [0, 1] range assuming typical distances < 2.0
    normalized_dist = min(distance / 2.0, 1.0)
    return 1.0 - normalized_dist


def _compute_sequence_similarity(
    query_sig: SequenceSignature | None,
    entry_sig: SequenceSignature | None,
) -> float:
    """Compute sequence similarity from signature vectors."""
    if query_sig is None or entry_sig is None:
        return 0.5  # Neutral similarity when no sequence data

    # Residue composition similarity
    residue_sim = _cosine_similarity(query_sig.residue_vector, entry_sig.residue_vector)

    # Dipeptide frequency similarity (weighted less)
    dipeptide_sim = _cosine_similarity(query_sig.dipeptide_vector, entry_sig.dipeptide_vector)

    # Length penalty/bonus
    len_ratio = min(query_sig.sequence_length, entry_sig.sequence_length) / max(query_sig.sequence_length, entry_sig.sequence_length)

    return 0.6 * residue_sim + 0.3 * dipeptide_sim + 0.1 * len_ratio


def _compute_functional_similarity(
    query_func: FunctionalAnnotation | None,
    entry_func: FunctionalAnnotation | None,
) -> float:
    """Compute functional similarity based on annotations."""
    if query_func is None or entry_func is None:
        return 0.5  # Neutral when no functional data

    # Enzyme class match
    enzyme_sim = 1.0 if query_func.enzyme_class == entry_func.enzyme_class else 0.0

    # GO term overlap (Jaccard similarity)
    query_go = set(query_func.go_terms)
    entry_go = set(entry_func.go_terms)
    if not query_go or not entry_go:
        go_sim = 0.5
    else:
        intersection = len(query_go & entry_go)
        union = len(query_go | entry_go)
        go_sim = intersection / union if union > 0 else 0.0

    # Active site proximity bonus (if positions are close)
    active_site_bonus = 0.0
    if query_func.active_site_positions and entry_func.active_site_positions:
        for q_pos in query_func.active_site_positions[:3]:
            for e_pos in entry_func.active_site_positions[:3]:
                if abs(q_pos - e_pos) <= 5:  # Within 5 residues
                    active_site_bonus = 0.2
                    break

    return 0.4 * enzyme_sim + 0.4 * go_sim + 0.2 * active_site_bonus


def _compute_thermostability_boost(
    query_tp: ThermostabilityProfile | None,
    entry_tp: ThermostabilityProfile | None,
) -> float:
    """Compute thermostability boost factor for retrieval."""
    if query_tp is None or entry_tp is None:
        return 1.0  # No boost without thermostability data

    boost = 1.0

    # Both thermophilic - strong boost
    if query_tp.is_thermophilic and entry_tp.is_thermophilic:
        boost *= 1.5

    # Temperature compatibility bonus
    temp_diff = abs(query_tp.optimal_temperature - entry_tp.optimal_temperature)
    if temp_diff <= 10:
        boost *= 1.3
    elif temp_diff <= 25:
        boost *= 1.1

    # Stability score correlation
    stability_product = query_tp.stability_score * entry_tp.stability_score
    boost *= (0.8 + 0.4 * stability_product)

    # Structural stabilization features
    if entry_tp.disulfide_count > 0:
        boost *= (1.0 + 0.05 * entry_tp.disulfide_count)

    return min(boost, 2.5)  # Cap the boost


def compute_combined_similarity(
    query_entry: StructureMemoryEntry,
    prototype_centroid: list[float],
    weights: dict[str, float] | None = None,
) -> SimilarityScores:
    """Compute combined similarity from multiple metrics."""
    if weights is None:
        weights = {"structural": 0.5, "sequence": 0.2, "functional": 0.15, "thermostability": 0.15}

    structural = _compute_structural_similarity(query_entry.descriptor, prototype_centroid)
    sequence = _compute_sequence_similarity(query_entry.sequence_signature, None)
    functional = _compute_functional_similarity(query_entry.functional, None)
    thermo_boost = _compute_thermostability_boost(
        query_entry.thermostability,
        query_entry.thermostability
    )

    # Normalize weights
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    combined = (
        normalized_weights["structural"] * structural +
        normalized_weights["sequence"] * sequence +
        normalized_weights["functional"] * functional
    ) * thermo_boost

    return SimilarityScores(
        structural_similarity=structural,
        sequence_similarity=sequence,
        functional_similarity=functional,
        thermostability_boost=thermo_boost,
        combined_score=combined,
    )


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


# =============================================================================
# k-NN Retrieval Optimization with Spatial Indexing
# =============================================================================

class BallTreeNode:
    """Node for Ball Tree spatial index."""
    __slots__ = ["centroid", "radius", "left", "right", "indices", "is_leaf"]

    def __init__(
        self,
        centroid: np.ndarray,
        radius: float,
        left: "BallTreeNode | None" = None,
        right: "BallTreeNode | None" = None,
        indices: list[int] | None = None,
    ):
        self.centroid = centroid
        self.radius = radius
        self.left = left
        self.right = right
        self.indices = indices
        self.is_leaf = indices is not None


class BallTree:
    """Ball Tree spatial index for efficient k-NN retrieval."""

    def __init__(self, points: list[list[float]], leaf_size: int = 10):
        self.points = np.array(points)
        self.leaf_size = leaf_size
        self.root = self._build_tree(range(len(points))) if len(points) > 0 else None

    def _build_tree(self, indices: list[int]) -> BallTreeNode:
        """Recursively build the Ball Tree."""
        if len(indices) <= self.leaf_size:
            points_subset = self.points[indices]
            centroid = np.mean(points_subset, axis=0)
            radius = float(np.max(np.linalg.norm(points_subset - centroid, axis=1)))
            return BallTreeNode(centroid=centroid, radius=radius, indices=indices)

        points_subset = self.points[indices]
        centroid = np.mean(points_subset, axis=0)

        # Find point furthest from centroid as first split
        distances = np.linalg.norm(points_subset - centroid, axis=1)
        far_idx = indices[np.argmax(distances)]

        # Find point furthest from far_idx
        distances_from_far = np.linalg.norm(points_subset - self.points[far_idx], axis=1)
        far2_idx = indices[np.argmax(distances_from_far)]

        # Split by median along the line connecting the two far points
        split_point = (self.points[far_idx] + self.points[far2_idx]) / 2
        projections = np.dot(points_subset, split_point - centroid)
        median_proj = float(np.median(projections))

        left_mask = projections <= median_proj
        left_indices = [indices[i] for i in range(len(indices)) if left_mask[i]]
        right_indices = [indices[i] for i in range(len(indices)) if not left_mask[i]]

        if len(left_indices) == 0 or len(right_indices) == 0:
            radius = float(np.max(np.linalg.norm(points_subset - centroid, axis=1)))
            return BallTreeNode(centroid=centroid, radius=radius, indices=indices)

        radius = float(np.max(np.linalg.norm(points_subset - centroid, axis=1)))
        return BallTreeNode(
            centroid=centroid,
            radius=radius,
            left=self._build_tree(left_indices),
            right=self._build_tree(right_indices),
        )

    def knn_search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        """Find k nearest neighbors using priority queue search."""
        if self.root is None:
            return []

        query_arr = np.array(query)
        heap: list[tuple[float, int]] = []  # (distance, index)
        results: list[tuple[int, float]] = []

        def _search(node: BallTreeNode, tau: float) -> float:
            """Search subtree, return new tau (max distance in heap)."""
            nonlocal heap

            # Check if query is within node radius + tau
            dist_to_centroid = float(np.linalg.norm(query_arr - node.centroid))

            if node.is_leaf:
                for idx in node.indices:
                    d = float(np.linalg.norm(query_arr - self.points[idx]))
                    if len(results) < k:
                        heapq.heappush(heap, (-d, idx))
                        results.append((idx, d))
                    elif d < results[0][1] if results else False:
                        heapq.heappop(heap)
                        results[0] = (idx, d)
                        heapq.heappush(heap, (-d, idx))
                        heapq.heapify(heap)
                tau = results[0][1] if results else float("inf")
                return tau

            # Branch selection based on distance to centroids
            left_dist = float(np.linalg.norm(query_arr - node.left.centroid)) if node.left else float("inf")
            right_dist = float(np.linalg.norm(query_arr - node.right.centroid)) if node.right else float("inf")

            first, second = (node.left, node.right) if left_dist <= right_dist else (node.right, node.left)
            first_dist, second_dist = (left_dist, right_dist) if left_dist <= right_dist else (right_dist, left_dist)

            # Search first child
            if first and first_dist - first.radius <= tau:
                tau = _search(first, tau)

            # Search second child if potentially better
            if second and second_dist - second.radius <= tau:
                tau = _search(second, tau)

            return tau

        _search(self.root, float("inf"))
        return sorted(results, key=lambda x: x[1])[:k]


def build_ball_tree_index(
    entries: list[StructureMemoryEntry],
) -> tuple[BallTree, list[StructureMemoryEntry]]:
    """Build Ball Tree index from memory entries."""
    points = [entry.descriptor for entry in entries]
    tree = BallTree(points, leaf_size=16) if points else BallTree([], leaf_size=16)
    return tree, entries


def optimized_knn_retrieval(
    query_descriptor: list[float],
    tree: BallTree,
    entries: list[StructureMemoryEntry],
    k: int,
) -> list[tuple[StructureMemoryEntry, float]]:
    """Retrieve k nearest neighbors using Ball Tree."""
    if not entries:
        return []

    indices_distances = tree.knn_search(query_descriptor, k)
    return [(entries[idx], dist) for idx, dist in indices_distances]


# =============================================================================
# Active Learning Strategy for Uncertainty-Based Retrieval
# =============================================================================

def compute_entropy(probabilities: list[float]) -> float:
    """Compute Shannon entropy for a probability distribution."""
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_retrieval_uncertainty(
    aa_scores: dict[str, float],
    retrieved_count: int,
) -> float:
    """Compute uncertainty score for active learning."""
    if not aa_scores:
        return 1.0  # Maximum uncertainty

    total = sum(aa_scores.values())
    if total <= 0:
        return 1.0

    # Normalize probabilities
    probs = [score / total for score in aa_scores.values()]

    # Entropy-based uncertainty
    entropy = compute_entropy(probs)

    # Retrieval count penalty (less certain if retrieved many times with similar scores)
    count_penalty = min(1.0, retrieved_count / 20.0)

    # Confidence gap penalty (uncertain if top choices are close in score)
    sorted_scores = sorted(aa_scores.values(), reverse=True)
    if len(sorted_scores) >= 2:
        score_gap = sorted_scores[0] - sorted_scores[1]
        gap_penalty = max(0.0, 1.0 - score_gap * 10) if score_gap < 0.1 else 0.0
    else:
        gap_penalty = 0.0

    uncertainty = entropy / 4.322  # Normalize by max entropy for 20 AAs
    uncertainty = uncertainty * (1.0 + 0.2 * count_penalty) + 0.3 * gap_penalty

    return min(1.0, uncertainty)


def select_uncertain_positions(
    positions: list[int],
    uncertainty_scores: dict[int, float],
    top_n: int = 5,
    threshold: float = 0.5,
) -> list[int]:
    """Select positions with highest uncertainty for active learning queries."""
    scored_positions = [
        (pos, uncertainty_scores.get(pos, 1.0)) for pos in positions
    ]

    # Filter by threshold and sort by uncertainty
    uncertain = [(pos, score) for pos, score in scored_positions if score >= threshold]
    uncertain.sort(key=lambda x: -x[1])

    return [pos for pos, _ in uncertain[:top_n]]


def update_active_learning_state(
    state: ActiveLearningState,
    position: int,
    uncertainty: float,
) -> None:
    """Update active learning state after retrieval."""
    state.uncertainty_scores[position] = uncertainty
    state.retrieved_count[position] = state.retrieved_count.get(position, 0) + 1
    state.queried_positions.add(position)


def adaptive_similarity_weights(
    uncertainty: float,
    base_weights: dict[str, float],
) -> dict[str, float]:
    """Adapt similarity weights based on retrieval uncertainty."""
    if uncertainty < 0.3:
        # Low uncertainty: rely more on structural similarity
        return {"structural": 0.6, "sequence": 0.2, "functional": 0.1, "thermostability": 0.1}
    elif uncertainty < 0.6:
        # Medium uncertainty: balanced
        return base_weights
    else:
        # High uncertainty: explore more with sequence and functional
        return {"structural": 0.35, "sequence": 0.3, "functional": 0.2, "thermostability": 0.15}


def confidence_interval_score(
    scores: dict[str, float],
    confidence_level: float = 0.95,
) -> dict[str, tuple[float, float, float]]:
    """Compute confidence intervals for amino acid scores using bootstrap."""
    if not scores or len(scores) < 2:
        return {aa: (score, score, score) for aa, score in scores.items()}

    n_resamples = 100
    amino_acids = list(scores.keys())
    values = list(scores.values())
    n = len(values)

    result = {}
    for aa, base_score in scores.items():
        resamples = []
        for _ in range(n_resamples):
            sample = np.random.choice(values, size=n, replace=True)
            resamples.append(np.mean(sample))

        resamples = sorted(resamples)
        alpha = (1 - confidence_level) / 2
        lower = resamples[int(alpha * n_resamples)]
        upper = resamples[int((1 - alpha) * n_resamples)]
        result[aa] = (base_score, lower, upper)

    return result


# =============================================================================
# Thermostability Index Management
# =============================================================================

THERMOSTABILITY_CACHE = {}


def build_thermostability_index(
    project_root: Path,
    entries: list[StructureMemoryEntry],
) -> ThermostabilityIndex:
    """Build dedicated index for thermostability-focused retrieval."""
    index = ThermostabilityIndex()

    # Load thermostability profiles from configs or metadata
    for entry in entries:
        # Try to load thermostability data from project configs
        cache_key = f"{entry.target}:{entry.chain}"
        if cache_key not in THERMOSTABILITY_CACHE:
            tp = _load_thermostability_profile(project_root, entry.target, entry.chain)
            THERMOSTABILITY_CACHE[cache_key] = tp
        else:
            tp = THERMOSTABILITY_CACHE[cache_key]

        entry.thermostability = tp
        entry.thermostability_boost = _compute_thermostability_boost(tp, tp) if tp else 1.0

        if tp and tp.is_thermophilic:
            index.thermostable_entries.append(entry)
            index.thermal_organisms.add(tp.source_organism)
            index.stability_scores[f"{entry.target}:{entry.position}"] = tp.stability_score

            # Bucket by stability score
            bucket = int(tp.stability_score * 10)
            if bucket not in index.stability_buckets:
                index.stability_buckets[bucket] = []
            index.stability_buckets[bucket].append(entry)

        # Build spatial index by position ranges
        range_key = f"{entry.target}:{(entry.position // 50) * 50}-{(entry.position // 50) * 50 + 50}"
        if range_key not in index.spatial_index:
            index.spatial_index[range_key] = []
        index.spatial_index[range_key].append(len(entries) - 1)

    return index


def _load_thermostability_profile(
    project_root: Path,
    target: str,
    chain: str,
) -> ThermostabilityProfile | None:
    """Load thermostability profile from project configuration."""
    # Check if there's thermostability metadata in configs
    config_path = project_root / "configs" / f"{target}.yaml"
    if not config_path.exists():
        # Try to find any matching config
        configs = list(project_root.glob(f"configs/*{target}*.yaml"))
        if configs:
            config_path = configs[0]

    if config_path.exists():
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            thermo = cfg.get("thermostability", {})
            if thermo:
                return ThermostabilityProfile(
                    optimal_temperature=float(thermo.get("optimal_temperature", 37.0)),
                    melting_temperature=float(thermo.get("melting_temperature", 50.0)),
                    stability_score=float(thermo.get("stability_score", 0.5)),
                    source_organism=str(thermo.get("source_organism", "Unknown")),
                    is_thermophilic=bool(thermo.get("is_thermophilic", False)),
                    proline_content=float(thermo.get("proline_content", 0.0)),
                    arginine_content=float(thermo.get("arginine_content", 0.0)),
                    disulfide_count=int(thermo.get("disulfide_count", 0)),
                    surface_hydrophobicity=float(thermo.get("surface_hydrophobicity", 0.5)),
                )
        except Exception:
            pass

    return None


def filter_thermostable_entries(
    index: ThermostabilityIndex,
    min_stability: float = 0.5,
    require_thermophilic: bool = False,
) -> list[StructureMemoryEntry]:
    """Filter entries based on thermostability criteria."""
    if require_thermophilic:
        return [e for e in index.thermostable_entries if e.thermostability and e.thermostability.stability_score >= min_stability]

    return [
        e for e in index.thermostable_entries
        if e.thermostability and e.thermostability.stability_score >= min_stability
    ]


def boosted_retrieval(
    query_entry: StructureMemoryEntry,
    candidates: list[StructureMemoryEntry],
    thermostability_index: ThermostabilityIndex,
    k: int = 5,
    thermostability_weight: float = 0.25,
) -> list[tuple[StructureMemoryEntry, float]]:
    """Retrieve with thermostability boost for thermophilic proteins."""
    scored_candidates = []

    for candidate in candidates:
        # Base structural similarity
        base_dist = _vector_distance(query_entry.descriptor, candidate.descriptor)
        base_sim = _gaussian_kernel(base_dist, sigma=0.3)

        # Thermostability boost
        thermo_boost = 1.0
        if candidate.thermostability:
            thermo_boost = candidate.thermostability_boost

            # Extra boost for active site proximity (often conserved)
            if query_entry.thermostability and query_entry.thermostability.active_site_positions:
                for q_pos in query_entry.thermostability.active_site_positions:
                    if abs(q_pos - candidate.position) <= 10:
                        thermo_boost *= 1.2
                        break

        # Combined score with thermostability weight
        combined = (1 - thermostability_weight) * base_sim + thermostability_weight * thermo_boost
        combined = combined * base_sim  # Maintain correlation with structural similarity

        scored_candidates.append((candidate, combined))

    scored_candidates.sort(key=lambda x: -x[1])
    return scored_candidates[:k]


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
    use_thermostability_boost: bool = True,
    use_active_learning: bool = True,
    use_optimized_knn: bool = True,
) -> tuple[dict[int, dict[str, float]], dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    """
    Retrieve residue memory with enhanced thermostability support.

    Args:
        target: Target protein name
        pdb_path: Path to PDB file
        chain_id: Chain identifier
        protected_positions: Set of protected residue positions
        design_positions: Positions to design
        outputs_root: Root directory for outputs
        top_k: Number of top recommendations per position
        max_neighbors: Maximum number of prototype neighbors
        use_thermostability_boost: Enable thermostability-based scoring boost
        use_active_learning: Enable active learning for uncertainty-based retrieval
        use_optimized_knn: Use Ball Tree for optimized k-NN retrieval
    """
    project_root = outputs_root.parent
    bank = [entry for entry in build_structure_memory_bank(project_root) if entry.target != target]

    # Build thermostability index if enabled
    thermostability_index: ThermostabilityIndex | None = None
    if use_thermostability_boost:
        thermostability_index = build_thermostability_index(project_root, bank)

    # Build motif atlas
    atlas = build_structure_motif_atlas(
        bank,
        cluster_radius=0.42,
        min_cluster_size=2,
    )

    # Build Ball Tree index for optimized retrieval
    ball_tree: BallTree | None = None
    if use_optimized_knn and bank:
        ball_tree, _ = build_ball_tree_index(bank)

    # Build query entries with thermostability profiles
    query_entries = _build_chain_descriptors(
        pdb_path=pdb_path,
        chain_id=chain_id,
        protected_positions=protected_positions,
    )

    # Add thermostability data to query entries
    if thermostability_index:
        for entry in query_entries:
            entry.thermostability = _load_thermostability_profile(project_root, target, chain_id)
            entry.thermostability_boost = _compute_thermostability_boost(entry.thermostability, entry.thermostability) if entry.thermostability else 1.0

    query_map = {entry.position: entry for entry in query_entries}

    # Initialize active learning state
    active_learning_state = ActiveLearningState() if use_active_learning else None

    recommendations: dict[int, dict[str, float]] = {}
    diagnostics: dict[int, list[dict[str, Any]]] = {}
    uncertainty_map: dict[int, float] = {}

    for position in design_positions:
        if position not in query_map:
            continue
        query_entry = query_map[position]

        # Determine adaptive weights based on active learning
        current_weights = {"structural": 0.5, "sequence": 0.2, "functional": 0.15, "thermostability": 0.15}
        if use_active_learning and active_learning_state:
            # Compute preliminary uncertainty
            prelim_scores: dict[str, float] = {}
            prelim_prototypes = sorted(
                atlas,
                key=lambda item: _vector_distance(query_entry.descriptor, item.centroid),
            )[: int(max_neighbors)]
            for prototype in prelim_prototypes:
                distance = _vector_distance(query_entry.descriptor, prototype.centroid)
                similarity = _gaussian_kernel(distance)
                for residue, prob in prototype.residue_distribution.items():
                    prelim_scores[residue] = prelim_scores.get(residue, 0.0) + similarity * float(prob)

            retrieved_count = active_learning_state.retrieved_count.get(position, 0)
            uncertainty = compute_retrieval_uncertainty(prelim_scores, retrieved_count)
            uncertainty_map[position] = uncertainty

            if uncertainty >= 0.5:
                current_weights = adaptive_similarity_weights(uncertainty, current_weights)

        # Get prototypes using optimized k-NN or fallback to sorted
        if use_optimized_knn and ball_tree:
            knn_results = optimized_knn_retrieval(query_entry.descriptor, ball_tree, bank, max_neighbors)
            # Match results to atlas prototypes
            proto_distances = []
            for entry, dist in knn_results:
                for proto in atlas:
                    if _vector_distance(entry.descriptor, proto.centroid) < 0.01:
                        proto_distances.append((proto, dist))
                        break
            prototypes = [p for p, _ in proto_distances[: int(max_neighbors)]]
        else:
            prototypes = sorted(
                atlas,
                key=lambda item: _vector_distance(query_entry.descriptor, item.centroid),
            )[: int(max_neighbors)]

        aa_scores: dict[str, float] = {}
        neighbor_payload: list[dict[str, Any]] = []

        for prototype in prototypes:
            distance = _vector_distance(query_entry.descriptor, prototype.centroid)

            # Compute combined similarity scores
            similarity_scores = compute_combined_similarity(query_entry, prototype.centroid, current_weights)
            similarity = similarity_scores.combined_score

            # Apply thermostability boost if available
            thermo_factor = 1.0
            if thermostability_index and use_thermostability_boost:
                thermo_factor = similarity_scores.thermostability_boost
                # Additional boost for thermophilic matches
                thermo_candidates = [e for e in thermostability_index.thermostable_entries if e.position == prototype.support_targets[0] if prototype.support_targets]
                if thermo_candidates:
                    thermo_factor *= thermo_candidates[0].thermostability_boost

            support_weight = min(1.75, 0.55 + math.log1p(prototype.support_count))
            diversity_weight = min(1.5, 0.65 + 0.18 * len(prototype.support_targets))

            # Adjust weights based on similarity components
            structural_weight = current_weights["structural"]
            proto_weight = similarity * support_weight * diversity_weight * thermo_factor

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
                    "structural_similarity": round(similarity_scores.structural_similarity, 6),
                    "sequence_similarity": round(similarity_scores.sequence_similarity, 6),
                    "functional_similarity": round(similarity_scores.functional_similarity, 6),
                    "thermostability_boost": round(similarity_scores.thermostability_boost, 6),
                    "support_weight": round(support_weight, 6),
                    "diversity_weight": round(diversity_weight, 6),
                    "weight": round(proto_weight, 6),
                    "source": "motif_atlas",
                    "weights_used": current_weights,
                }
            )

        if aa_scores:
            # Compute confidence intervals for scores
            confidence_intervals = confidence_interval_score(aa_scores, confidence_level=0.95)

            ranked = sorted(aa_scores.items(), key=lambda item: (-item[1], item[0]))[: int(top_k)]
            recommendations[int(position)] = {aa: round(score, 6) for aa, score in ranked}

            # Add uncertainty to diagnostics
            if use_active_learning and active_learning_state:
                retrieved_count = active_learning_state.retrieved_count.get(position, 0)
                uncertainty = compute_retrieval_uncertainty(aa_scores, retrieved_count)
                update_active_learning_state(active_learning_state, position, uncertainty)
                neighbor_payload.append({
                    "active_learning": {
                        "uncertainty": round(uncertainty, 6),
                        "retrieved_count": retrieved_count,
                        "confidence_intervals": {aa: (round(lo, 6), round(hi, 6)) for aa, (score, lo, hi) in confidence_intervals.items()},
                        "selected_weights": current_weights,
                    }
                })

            diagnostics[int(position)] = neighbor_payload

    return recommendations, diagnostics, serialize_motif_atlas(atlas)


def retrieve_with_uncertainty_focus(
    target: str,
    pdb_path: Path,
    chain_id: str,
    protected_positions: set[int],
    design_positions: list[int],
    outputs_root: Path,
    top_k: int = 3,
    max_neighbors: int = 24,
    uncertainty_threshold: float = 0.6,
    min_candidates: int = 3,
) -> tuple[dict[int, dict[str, float]], dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    """
    Retrieve with active learning focus on uncertain positions.

    This function prioritizes positions with high uncertainty and uses
    adaptive sampling strategies to improve retrieval for challenging cases.
    """
    project_root = outputs_root.parent
    bank = [entry for entry in build_structure_memory_bank(project_root) if entry.target != target]

    # Build thermostability index
    thermostability_index = build_thermostability_index(project_root, bank)

    # Initial pass to compute uncertainties
    atlas = build_structure_motif_atlas(bank, cluster_radius=0.42, min_cluster_size=2)
    query_entries = _build_chain_descriptors(
        pdb_path=pdb_path,
        chain_id=chain_id,
        protected_positions=protected_positions,
    )

    # Add thermostability data
    for entry in query_entries:
        entry.thermostability = _load_thermostability_profile(project_root, target, chain_id)

    query_map = {entry.position: entry for entry in query_entries}

    # Compute initial uncertainties
    active_learning_state = ActiveLearningState()
    position_uncertainties: dict[int, float] = {}

    for position in design_positions:
        if position not in query_map:
            continue
        query_entry = query_map[position]

        prototypes = sorted(
            atlas,
            key=lambda item: _vector_distance(query_entry.descriptor, item.centroid),
        )[: int(max_neighbors)]

        prelim_scores: dict[str, float] = {}
        for prototype in prototypes:
            distance = _vector_distance(query_entry.descriptor, prototype.centroid)
            similarity = _gaussian_kernel(distance)
            for residue, prob in prototype.residue_distribution.items():
                prelim_scores[residue] = prelim_scores.get(residue, 0.0) + similarity * float(prob)

        retrieved_count = active_learning_state.retrieved_count.get(position, 0)
        uncertainty = compute_retrieval_uncertainty(prelim_scores, retrieved_count)
        position_uncertainties[position] = uncertainty

    # Select high-uncertainty positions for focused retrieval
    focused_positions = select_uncertain_positions(
        design_positions,
        position_uncertainties,
        top_n=len(design_positions),
        threshold=uncertainty_threshold,
    )

    # Add some low-uncertainty positions for diversity
    low_uncertain_positions = [p for p in design_positions if position_uncertainties.get(p, 1.0) < 0.3]
    focused_positions = focused_positions + low_uncertain_positions[:min_candidates]
    focused_positions = list(set(focused_positions))

    # Final retrieval with all positions
    recommendations: dict[int, dict[str, float]] = {}
    diagnostics: dict[int, list[dict[str, Any]]] = {}

    for position in focused_positions:
        if position not in query_map:
            continue
        query_entry = query_map[position]

        uncertainty = position_uncertainties.get(position, 0.5)
        current_weights = adaptive_similarity_weights(uncertainty, {"structural": 0.5, "sequence": 0.2, "functional": 0.15, "thermostability": 0.15})

        prototypes = sorted(
            atlas,
            key=lambda item: _vector_distance(query_entry.descriptor, item.centroid),
        )[: int(max_neighbors) * 2]  # Expand search for uncertain positions

        aa_scores: dict[str, float] = {}
        neighbor_payload: list[dict[str, Any]] = []

        for prototype in prototypes:
            distance = _vector_distance(query_entry.descriptor, prototype.centroid)
            similarity_scores = compute_combined_similarity(query_entry, prototype.centroid, current_weights)
            similarity = similarity_scores.combined_score

            support_weight = min(1.75, 0.55 + math.log1p(prototype.support_count))
            diversity_weight = min(1.5, 0.65 + 0.18 * len(prototype.support_targets))

            proto_weight = similarity * support_weight * diversity_weight * similarity_scores.thermostability_boost

            for residue, prob in prototype.residue_distribution.items():
                aa_scores[residue] = aa_scores.get(residue, 0.0) + proto_weight * float(prob)

            neighbor_payload.append({
                "prototype_id": prototype.prototype_id,
                "distance": round(distance, 6),
                "similarity": round(similarity, 6),
                "thermostability_boost": round(similarity_scores.thermostability_boost, 6),
                "uncertainty": round(uncertainty, 6),
                "source": "focused_retrieval",
            })

        if aa_scores:
            ranked = sorted(aa_scores.items(), key=lambda item: (-item[1], item[0]))[: int(top_k)]
            recommendations[int(position)] = {aa: round(score, 6) for aa, score in ranked}
            diagnostics[int(position)] = neighbor_payload

            # Update active learning state
            retrieved_count = active_learning_state.retrieved_count.get(position, 0)
            final_uncertainty = compute_retrieval_uncertainty(aa_scores, retrieved_count)
            update_active_learning_state(active_learning_state, position, final_uncertainty)

    return recommendations, diagnostics, serialize_motif_atlas(atlas)
