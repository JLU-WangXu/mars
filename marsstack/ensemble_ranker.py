"""Ensemble learning ranker for candidate sequence selection.

This module provides XGBoost/LightGBM-based ranking for MARS-FIELD candidates,
with feature engineering for structure, evolution, energy, and conservation features.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .evolution import profile_log_score, differential_profile_score
from .structure_features import ResidueFeature


FEATURE_NAMES = [
    # Structure features
    "struct__mean_sasa",
    "struct__mean_b_factor",
    "struct__min_dist_protected",
    "struct__disulfide_count",
    "struct__glyco_motif_count",
    # Evolution features
    "evo__profile_score",
    "evo__asr_score",
    "evo__family_diff_score",
    "evo__profile_entropy",
    "evo__conservation",
    # Energy features (mutation-related)
    "energy__mutation_count",
    "energy__mutation_fraction",
    "energy__hotspot_mutated_count",
    "energy__hotspot_mutated_fraction",
    "energy__flexible_mutated_count",
    "energy__flexible_mutated_fraction",
    "energy__buried_mutation_count",
    "energy__surface_mutation_count",
    # Conservation features
    "cons__wt_retention",
    "cons__rare_to_common_ratio",
    "cons__hydrophobic_conservation",
    "cons__charge_conservation",
    # Score features
    "score__decoder_score",
    "score__mpnn_score",
    "score__esm_recovery",
    "score__fusion_score",
    # Support features
    "support__source_count",
    "support__learned_source_count",
    "support__consensus_count",
    "support__manual_count",
]


@dataclass
class CandidateFeatures:
    """Feature vector for a candidate sequence."""

    features: dict[str, float] = field(default_factory=dict)
    candidate_id: str = ""
    sequence: str = ""
    mutations: list[str] = field(default_factory=list)
    source: str = ""
    source_group: str = ""
    supporting_sources: list[str] = field(default_factory=list)
    label: float | None = None  # For training: 1.0 = selected, 0.0 = rejected


@dataclass
class RankerConfig:
    """Configuration for ensemble ranker."""

    model_type: str = "xgboost"  # "xgboost" or "lightgbm"
    learning_rate: float = 0.05
    max_depth: int = 6
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    n_folds: int = 5
    random_state: int = 42
    model_dir: Path = field(default_factory=lambda: Path("outputs/ranker_models"))


def _compute_entropy(probs: dict[str, float]) -> float:
    """Compute Shannon entropy of amino acid probability distribution."""
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log(p + 1e-10)
    return entropy


def _compute_conservation(profile: dict[str, float]) -> float:
    """Compute position-specific conservation score."""
    if not profile:
        return 0.0
    max_prob = max(profile.values())
    return max_prob


def extract_structure_features(
    features: list[ResidueFeature],
    mutations: list[str],
    position_to_index: dict[int, int],
    wt_seq: str,
) -> dict[str, float]:
    """Extract structure-related features from residue features and mutations."""
    feat_map = {f.num: f for f in features}
    mutated_positions = set()
    buried_positions = set()
    surface_positions = set()

    for mut in mutations:
        if len(mut) >= 2:
            digits = "".join(c for c in mut if c.isdigit())
            if digits:
                mutated_positions.add(int(digits))

    for feat in features:
        if feat.num in mutated_positions:
            if feat.sasa < 20:
                buried_positions.add(feat.num)
            elif feat.sasa >= 40:
                surface_positions.add(feat.num)

    sasa_values = [f.sasa for f in features if f.num in mutated_positions]
    b_values = [f.mean_b for f in features if f.num in mutated_positions]
    dist_values = [f.min_dist_protected for f in features if f.num in mutated_positions]

    return {
        "struct__mean_sasa": float(np.mean(sasa_values)) if sasa_values else 0.0,
        "struct__mean_b_factor": float(np.mean(b_values)) if b_values else 0.0,
        "struct__min_dist_protected": float(np.min(dist_values)) if dist_values else 0.0,
        "struct__disulfide_count": float(sum(1 for f in features if f.num in mutated_positions and f.in_disulfide)),
        "struct__glyco_motif_count": float(sum(1 for f in features if f.num in mutated_positions and f.glyco_motif)),
    }


def extract_evolution_features(
    seq: str,
    profile: list[dict[str, float]] | None,
    asr_profile: list[dict[str, float]] | None,
    positive_profile: list[dict[str, float]] | None,
    negative_profile: list[dict[str, float]] | None,
    positions: list[int],
    position_to_index: dict[int, int] | None = None,
) -> dict[str, float]:
    """Extract evolution-related features from sequence profiles."""
    result = {
        "evo__profile_score": 0.0,
        "evo__asr_score": 0.0,
        "evo__family_diff_score": 0.0,
        "evo__profile_entropy": 0.0,
        "evo__conservation": 0.0,
    }

    if profile is not None:
        result["evo__profile_score"] = profile_log_score(seq, profile, positions, position_to_index=position_to_index)
        entropies = []
        conservations = []
        for pos in positions:
            idx = position_to_index[pos] if position_to_index else pos - 1
            if idx < len(profile):
                entropies.append(_compute_entropy(profile[idx]))
                conservations.append(_compute_conservation(profile[idx]))
        if entropies:
            result["evo__profile_entropy"] = float(np.mean(entropies))
        if conservations:
            result["evo__conservation"] = float(np.mean(conservations))

    if asr_profile is not None:
        result["evo__asr_score"] = profile_log_score(seq, asr_profile, positions, position_to_index=position_to_index)

    if positive_profile is not None and negative_profile is not None:
        result["evo__family_diff_score"] = differential_profile_score(
            seq, positive_profile, negative_profile, positions, position_to_index=position_to_index
        )

    return result


def extract_energy_features(
    mutations: list[str],
    design_positions: list[int],
    oxidation_hotspots: list[int],
    flexible_positions: list[int],
    features: list[ResidueFeature],
) -> dict[str, float]:
    """Extract energy/mutation-related features."""
    mutated_positions = set()
    for mut in mutations:
        digits = "".join(c for c in mut if c.isdigit())
        if digits:
            mutated_positions.add(int(digits))

    hotspot_set = set(oxidation_hotspots)
    flexible_set = set(flexible_positions)
    num_design = max(1, len(design_positions))

    hotspot_hits = sum(1 for pos in mutated_positions if pos in hotspot_set)
    flex_hits = sum(1 for pos in mutated_positions if pos in flexible_set)

    feat_map = {f.num: f for f in features}
    buried_count = 0
    surface_count = 0
    for pos in mutated_positions:
        feat = feat_map.get(pos)
        if feat:
            if feat.sasa < 20:
                buried_count += 1
            elif feat.sasa >= 40:
                surface_count += 1

    return {
        "energy__mutation_count": float(len(mutations)),
        "energy__mutation_fraction": float(len(mutations) / num_design),
        "energy__hotspot_mutated_count": float(hotspot_hits),
        "energy__hotspot_mutated_fraction": float(hotspot_hits / num_design),
        "energy__flexible_mutated_count": float(flex_hits),
        "energy__flexible_mutated_fraction": float(flex_hits / num_design),
        "energy__buried_mutation_count": float(buried_count),
        "energy__surface_mutation_count": float(surface_count),
    }


def extract_conservation_features(
    seq: str,
    wt_seq: str,
    profile: list[dict[str, float]] | None,
    positions: list[int],
    position_to_index: dict[int, int] | None = None,
) -> dict[str, float]:
    """Extract conservation-related features."""
    wt_retention = 0
    rare_to_common = 0.0
    hydrophobic_cons = 0
    charge_cons = 0

    HYDROPHOBIC = {"A", "V", "I", "L", "M", "F", "W", "P"}
    POSITIVE = {"K", "R", "H"}
    NEGATIVE = {"D", "E"}

    for pos in positions:
        idx = position_to_index[pos] if position_to_index else pos - 1
        if idx >= len(seq) or idx >= len(wt_seq):
            continue

        aa = seq[idx]
        wt = wt_seq[idx]

        if aa == wt:
            wt_retention += 1

        if profile is not None and idx < len(profile):
            prof = profile[idx]
            if prof:
                aa_prob = prof.get(aa, 0.0)
                wt_prob = prof.get(wt, 0.0)
                max_prob = max(prof.values()) if prof else 1.0
                if wt_prob > 0 and max_prob > 0:
                    rare_to_common += (aa_prob / max_prob) / (wt_prob / max_prob + 1e-6)

        if aa in HYDROPHOBIC and wt in HYDROPHOBIC:
            hydrophobic_cons += 1
        if (aa in POSITIVE and wt in POSITIVE) or (aa in NEGATIVE and wt in NEGATIVE):
            charge_cons += 1

    n_positions = max(1, len(positions))

    return {
        "cons__wt_retention": float(wt_retention / n_positions),
        "cons__rare_to_common_ratio": float(rare_to_common / n_positions),
        "cons__hydrophobic_conservation": float(hydrophobic_cons / n_positions),
        "cons__charge_conservation": float(charge_cons / n_positions),
    }


def extract_score_features(
    decoder_score: float,
    header_metrics: dict[str, float],
    fusion_score: float | None = None,
) -> dict[str, float]:
    """Extract score-related features."""
    return {
        "score__decoder_score": float(decoder_score),
        "score__mpnn_score": float(header_metrics.get("mpnn_score", header_metrics.get("header__mpnn_score", 0.0))),
        "score__esm_recovery": float(header_metrics.get("esm_recovery", header_metrics.get("header__esm_recovery", 0.0))),
        "score__fusion_score": float(fusion_score) if fusion_score is not None else 0.0,
    }


def extract_support_features(supporting_sources: list[str]) -> dict[str, float]:
    """Extract support/source-related features."""
    unique_sources = list(dict.fromkeys(supporting_sources))
    learned = {"baseline_mpnn", "mars_mpnn", "esm_if"}
    manual = {"manual"}

    return {
        "support__source_count": float(len(unique_sources)),
        "support__learned_source_count": float(sum(1 for s in unique_sources if s in learned)),
        "support__consensus_count": float(sum(1 for s in unique_sources if s not in learned and s not in manual)),
        "support__manual_count": float(sum(1 for s in unique_sources if s in manual)),
    }


def build_candidate_features(
    sequence: str,
    wt_sequence: str,
    mutations: list[str],
    features: list[ResidueFeature],
    profile: list[dict[str, float]] | None,
    asr_profile: list[dict[str, float]] | None,
    positive_profile: list[dict[str, float]] | None,
    negative_profile: list[dict[str, float]] | None,
    design_positions: list[int],
    oxidation_hotspots: list[int],
    flexible_positions: list[int],
    position_to_index: dict[int, int],
    decoder_score: float = 0.0,
    header_metrics: dict[str, float] | None = None,
    supporting_sources: list[str] | None = None,
    source: str = "",
    source_group: str = "",
    candidate_id: str = "",
    fusion_score: float | None = None,
) -> CandidateFeatures:
    """Build complete feature vector for a candidate sequence."""
    header_metrics = header_metrics or {}

    result = CandidateFeatures(
        candidate_id=candidate_id,
        sequence=sequence,
        mutations=mutations,
        source=source,
        source_group=source_group,
        supporting_sources=supporting_sources or [],
    )

    # Extract all feature groups
    result.features.update(extract_structure_features(features, mutations, position_to_index, wt_sequence))
    result.features.update(
        extract_evolution_features(
            sequence,
            profile,
            asr_profile,
            positive_profile,
            negative_profile,
            design_positions,
            position_to_index,
        )
    )
    result.features.update(
        extract_energy_features(mutations, design_positions, oxidation_hotspots, flexible_positions, features)
    )
    result.features.update(
        extract_conservation_features(sequence, wt_sequence, profile, design_positions, position_to_index)
    )
    result.features.update(extract_score_features(decoder_score, header_metrics, fusion_score))
    result.features.update(extract_support_features(supporting_sources or []))

    return result


def features_to_array(candidate_features: list[CandidateFeatures], feature_names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Convert list of CandidateFeatures to numpy array."""
    feature_names = feature_names or FEATURE_NAMES
    X = np.array([[cf.features.get(name, 0.0) for name in feature_names] for cf in candidate_features], dtype=float)
    return X, feature_names


def labels_to_array(candidate_features: list[CandidateFeatures]) -> np.ndarray:
    """Extract labels from candidate features."""
    return np.array([cf.label if cf.label is not None else 0.0 for cf in candidate_features], dtype=float)


class EnsembleRanker:
    """Ensemble learning ranker using XGBoost/LightGBM for candidate ranking."""

    def __init__(self, config: RankerConfig | None = None):
        self.config = config or RankerConfig()
        self.model: Any = None
        self.feature_names: list[str] = FEATURE_NAMES
        self.feature_importance_: dict[str, float] = {}
        self.is_fitted_: bool = False
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the ranking model."""
        if self.config.model_type == "lightgbm":
            try:
                import lightgbm as lgb

                self.model = lgb.LGBMRanker(
                    objective="lambdarank",
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    max_depth=self.config.max_depth,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    min_child_samples=self.config.min_child_weight,
                    reg_alpha=self.config.reg_alpha,
                    reg_lambda=self.config.reg_lambda,
                    random_state=self.config.random_state,
                    verbose=-1,
                    importance_type="gain",
                )
            except ImportError:
                self.config.model_type = "xgboost"
                self._init_model()
        else:
            try:
                import xgboost as xgb

                self.model = xgb.XGBRanker(
                    objective="rank:pairwise",
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    max_depth=self.config.max_depth,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    min_child_weight=self.config.min_child_weight,
                    reg_alpha=self.config.reg_alpha,
                    reg_lambda=self.config.reg_lambda,
                    random_state=self.config.random_state,
                    verbosity=0,
                    importance_type="gain",
                )
            except ImportError:
                raise ImportError("Neither xgboost nor lightgbm is available. Please install one: pip install xgboost lightgbm")

    def fit(
        self,
        candidate_features: list[CandidateFeatures],
        groups: list[int] | None = None,
    ) -> "EnsembleRanker":
        """Fit the ranker model."""
        X, feature_names = features_to_array(candidate_features, self.feature_names)
        y = labels_to_array(candidate_features)

        if groups is None:
            groups = [len(candidate_features)]

        self.feature_names = feature_names
        self.model.fit(X, y)

        self.is_fitted_ = True
        self._compute_feature_importance(X)

        return self

    def _compute_feature_importance(self, X: np.ndarray) -> None:
        """Compute and store feature importance scores."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            self.feature_importance_ = {
                name: float(imp) for name, imp in zip(self.feature_names, importances) if imp > 0
            }

    def predict(self, candidate_features: list[CandidateFeatures]) -> np.ndarray:
        """Predict ranking scores for candidates."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        X, _ = features_to_array(candidate_features, self.feature_names)
        return self.model.predict(X)

    def cross_validate(
        self,
        candidate_features: list[CandidateFeatures],
        n_folds: int | None = None,
    ) -> dict[str, Any]:
        """Perform cross-validation evaluation."""
        n_folds = n_folds or self.config.n_folds
        X, feature_names = features_to_array(candidate_features, self.feature_names)
        y = labels_to_array(candidate_features)

        n_samples = len(candidate_features)
        fold_size = n_samples // n_folds

        metrics = {
            "ndcg@5": [],
            "ndcg@10": [],
            "ndcg@all": [],
            "precision@5": [],
            "precision@10": [],
            "mrr": [],
        }

        indices = np.arange(n_samples)
        np.random.seed(self.config.random_state)
        np.random.shuffle(indices)

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n_samples

            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            fold_model = self._create_fold_model()
            fold_model.fit(X_train, y_train)

            y_pred = fold_model.predict(X_val)

            fold_metrics = self._compute_ranking_metrics(y_val, y_pred, k=10)
            for key, value in fold_metrics.items():
                if key in metrics:
                    metrics[key].append(value)

        return {
            "mean_ndcg@5": float(np.mean(metrics["ndcg@5"])),
            "mean_ndcg@10": float(np.mean(metrics["ndcg@10"])),
            "mean_ndcg@all": float(np.mean(metrics["ndcg@all"])),
            "mean_precision@5": float(np.mean(metrics["precision@5"])),
            "mean_precision@10": float(np.mean(metrics["precision@10"])),
            "mean_mrr": float(np.mean(metrics["mrr"])),
            "std_ndcg@10": float(np.std(metrics["ndcg@10"])),
            "fold_metrics": {k: [float(v) for v in vals] for k, vals in metrics.items()},
        }

    def _create_fold_model(self) -> Any:
        """Create a new model instance for cross-validation fold."""
        if self.config.model_type == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMRanker(
                objective="lambdarank",
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                min_child_samples=self.config.min_child_weight,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=self.config.random_state,
                verbose=-1,
            )
        else:
            import xgboost as xgb

            return xgb.XGBRanker(
                objective="rank:pairwise",
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                min_child_weight=self.config.min_child_weight,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=self.config.random_state,
                verbosity=0,
            )

    def _compute_ranking_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> dict[str, float]:
        """Compute ranking metrics (NDCG, Precision, MRR)."""
        n = len(y_true)
        if n == 0:
            return {"ndcg@5": 0.0, "ndcg@10": 0.0, "ndcg@all": 0.0, "precision@5": 0.0, "precision@10": 0.0, "mrr": 0.0}

        order = np.argsort(-y_pred)
        y_sorted = y_true[order]

        def dcg(scores: np.ndarray) -> float:
            return sum((2**s - 1) / math.log2(i + 2) for i, s in enumerate(scores))

        ideal_scores = np.sort(y_true)[::-1]
        k_actual = min(k, n)
        dcg_k = dcg(y_sorted[:k_actual])
        idcg_k = dcg(ideal_scores[:k_actual])
        ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0.0

        dcg_all = dcg(y_sorted)
        idcg_all = dcg(ideal_scores)
        ndcg_all = dcg_all / idcg_all if idcg_all > 0 else 0.0

        k_actual_5 = min(5, n)
        precision_5 = np.sum(y_sorted[:k_actual_5]) / k_actual_5 if k_actual_5 > 0 else 0.0

        k_actual_10 = min(10, n)
        precision_10 = np.sum(y_sorted[:k_actual_10]) / k_actual_10 if k_actual_10 > 0 else 0.0

        mrr = 0.0
        for i, score in enumerate(y_sorted):
            if score > 0:
                mrr = 1.0 / (i + 1)
                break

        return {
            "ndcg@5": float(ndcg_k) if k == 5 else float(dcg_k / (dcg(ideal_scores[:min(5, n)]) if dcg(ideal_scores[:min(5, n)]) > 0 else 1.0)),
            "ndcg@10": float(ndcg_k) if k == 10 else float(dcg_k / (dcg(ideal_scores[:min(10, n)]) if dcg(ideal_scores[:min(10, n)]) > 0 else 1.0)),
            "ndcg@all": float(ndcg_all),
            "precision@5": float(precision_5),
            "precision@10": float(precision_10),
            "mrr": float(mrr),
        }

    def get_feature_importance(self, top_k: int = 20) -> list[tuple[str, float]]:
        """Get top-k most important features."""
        if not self.feature_importance_:
            return []
        sorted_features = sorted(self.feature_importance_.items(), key=lambda x: -x[1])
        return sorted_features[:top_k]

    def save(self, path: Path | str | None = None) -> Path:
        """Save the trained model to disk."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Cannot save.")

        path = Path(path) if path else self.config.model_dir / f"ranker_{self.config.model_type}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance_,
            "config": {
                "model_type": self.config.model_type,
                "learning_rate": self.config.learning_rate,
                "max_depth": self.config.max_depth,
                "n_estimators": self.config.n_estimators,
                "subsample": self.config.subsample,
                "colsample_bytree": self.config.colsample_bytree,
                "min_child_weight": self.config.min_child_weight,
                "reg_alpha": self.config.reg_alpha,
                "reg_lambda": self.config.reg_lambda,
                "random_state": self.config.random_state,
            },
        }

        with path.open("wb") as f:
            pickle.dump(model_data, f)

        importance_path = path.with_suffix(".importance.json")
        with importance_path.open("w", encoding="utf-8") as f:
            json.dump(self.feature_importance_, f, indent=2)

        return path

    @classmethod
    def load(cls, path: Path | str) -> "EnsembleRanker":
        """Load a trained model from disk."""
        path = Path(path)

        with path.open("rb") as f:
            model_data = pickle.load(f)

        config = RankerConfig(
            model_type=model_data["config"]["model_type"],
            learning_rate=model_data["config"]["learning_rate"],
            max_depth=model_data["config"]["max_depth"],
            n_estimators=model_data["config"]["n_estimators"],
            subsample=model_data["config"]["subsample"],
            colsample_bytree=model_data["config"]["colsample_bytree"],
            min_child_weight=model_data["config"]["min_child_weight"],
            reg_alpha=model_data["config"]["reg_alpha"],
            reg_lambda=model_data["config"]["reg_lambda"],
            random_state=model_data["config"]["random_state"],
        )

        ranker = cls(config)
        ranker.model = model_data["model"]
        ranker.feature_names = model_data["feature_names"]
        ranker.feature_importance_ = model_data["feature_importance"]
        ranker.is_fitted_ = True

        return ranker


def rank_candidates(
    candidates: list[CandidateFeatures],
    model: EnsembleRanker | None = None,
    config: RankerConfig | None = None,
    use_fallback: bool = True,
) -> list[tuple[CandidateFeatures, float]]:
    """Rank candidates using the ensemble ranker.

    Args:
        candidates: List of candidate features to rank
        model: Pre-trained EnsembleRanker (if None, uses fallback scoring)
        config: Ranker configuration (used if model is None)
        use_fallback: If True, use weighted sum fallback when model unavailable

    Returns:
        List of (candidate, score) tuples sorted by score descending
    """
    if model is not None and model.is_fitted_:
        scores = model.predict(candidates)
        return sorted(zip(candidates, scores), key=lambda x: -x[1])

    if use_fallback:
        fallback_scores = []
        for cf in candidates:
            score = _compute_fallback_score(cf)
            fallback_scores.append((cf, score))
        return sorted(fallback_scores, key=lambda x: -x[1])

    return [(cf, 0.0) for cf in candidates]


def _compute_fallback_score(candidate: CandidateFeatures) -> float:
    """Compute fallback score when no trained model is available."""
    features = candidate.features
    score = 0.0

    score += features.get("evo__profile_score", 0.0) * 0.3
    score += features.get("evo__asr_score", 0.0) * 0.25
    score += features.get("evo__family_diff_score", 0.0) * 0.2
    score += features.get("cons__wt_retention", 0.0) * 0.1
    score += features.get("support__learned_source_count", 0.0) * 0.15

    score -= features.get("energy__mutation_fraction", 0.0) * 0.1
    score -= features.get("energy__hotspot_mutated_fraction", 0.0) * 0.2

    score += features.get("score__decoder_score", 0.0) * 0.1

    return score


def analyze_feature_importance(
    candidate_features: list[CandidateFeatures],
    config: RankerConfig | None = None,
) -> dict[str, Any]:
    """Analyze feature importance from candidate features.

    This trains a temporary model to compute feature importance
    without saving the model.
    """
    config = config or RankerConfig()
    ranker = EnsembleRanker(config)

    y = labels_to_array(candidate_features)
    if len(np.unique(y)) < 2:
        return {"error": "Need at least 2 distinct labels for importance analysis"}

    ranker.fit(candidate_features)

    importance = ranker.get_feature_importance(top_k=len(FEATURE_NAMES))

    grouped_importance: dict[str, float] = {
        "structure": 0.0,
        "evolution": 0.0,
        "energy": 0.0,
        "conservation": 0.0,
        "score": 0.0,
        "support": 0.0,
        "other": 0.0,
    }

    for name, imp in importance:
        if name.startswith("struct__"):
            grouped_importance["structure"] += imp
        elif name.startswith("evo__"):
            grouped_importance["evolution"] += imp
        elif name.startswith("energy__"):
            grouped_importance["energy"] += imp
        elif name.startswith("cons__"):
            grouped_importance["conservation"] += imp
        elif name.startswith("score__"):
            grouped_importance["score"] += imp
        elif name.startswith("support__"):
            grouped_importance["support"] += imp
        else:
            grouped_importance["other"] += imp

    total = sum(grouped_importance.values())
    if total > 0:
        grouped_importance = {k: v / total for k, v in grouped_importance.items()}

    return {
        "feature_importance": dict(importance),
        "grouped_importance": grouped_importance,
        "model_type": config.model_type,
    }
