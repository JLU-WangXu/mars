from __future__ import annotations

from pathlib import Path

import torch

from ..decoder import PositionField, ResidueOption
from .neural_dataset import AA_ORDER, NeuralTargetBatch, load_neural_corpus
from .neural_model import MarsFieldNeuralModel
from .neural_training import tensorize_pair_inputs, train_model


def train_holdout_neural_model(
    outputs_root: Path,
    holdout_batch: NeuralTargetBatch,
    epochs: int,
    lr: float,
) -> tuple[MarsFieldNeuralModel, list[dict[str, float]], list[str]]:
    corpus = load_neural_corpus(outputs_root=outputs_root)
    train_batches = [batch for batch in corpus if batch.pipeline_dir != holdout_batch.pipeline_dir]
    if not train_batches:
        raise ValueError("Need at least one non-holdout target to train the neural field generator.")
    model, history = train_model(train_batches=train_batches, epochs=int(epochs), lr=float(lr))
    return model, history, [batch.target for batch in train_batches]


def _position_field_lookup(position_fields: list[PositionField] | None) -> dict[int, dict[str, dict[str, object]]]:
    lookup: dict[int, dict[str, dict[str, object]]] = {}
    for field in position_fields or []:
        bucket: dict[str, dict[str, object]] = {}
        for option in field.options:
            bucket[str(option.residue)] = {
                "score": float(option.score),
                "supporting_sources": list(option.supporting_sources or []),
            }
        lookup[int(field.position)] = bucket
    return lookup


def _zscore_tensor(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / (values.std() + 1e-6)


def _prior_pair_matrix(
    pair_key: tuple[int, int],
    prior_pairwise: dict[tuple[int, int], dict[tuple[str, str], float]] | None,
) -> torch.Tensor:
    matrix = torch.zeros((len(AA_ORDER), len(AA_ORDER)), dtype=torch.float32)
    if not prior_pairwise:
        return matrix
    bucket = prior_pairwise.get(pair_key, prior_pairwise.get((pair_key[1], pair_key[0]), {}))
    for (aa_i, aa_j), score in bucket.items():
        if aa_i in AA_ORDER and aa_j in AA_ORDER:
            matrix[AA_ORDER.index(aa_i), AA_ORDER.index(aa_j)] = float(score)
    return matrix


def build_neural_residue_field(
    model: MarsFieldNeuralModel,
    batch: NeuralTargetBatch,
    top_k_per_position: int = 4,
    pair_top_k: int = 32,
    prior_position_fields: list[PositionField] | None = None,
    prior_pairwise: dict[tuple[int, int], dict[tuple[str, str], float]] | None = None,
    prior_field_weight: float = 0.55,
    prior_pair_weight: float = 0.35,
) -> tuple[list[PositionField], dict[tuple[int, int], dict[tuple[str, str], float]], dict[str, object]]:
    model.eval()
    with torch.no_grad():
        output = model(
            geom_inputs=torch.tensor(batch.geom_inputs),
            evo_inputs=torch.tensor(batch.evo_inputs),
            asr_inputs=torch.tensor(batch.asr_inputs),
            retrieval_inputs=torch.tensor(batch.retrieval_inputs),
            env_inputs=torch.tensor(batch.env_inputs),
            pair_features=tensorize_pair_inputs(batch.pair_inputs),
        )

    unary_logits = output.unary.detach().cpu()
    prior_lookup = _position_field_lookup(prior_position_fields)
    fields: list[PositionField] = []
    for pos_idx, position in enumerate(batch.positions):
        wt_residue = AA_ORDER[int(batch.wt_indices[pos_idx])]
        neural_scores = _zscore_tensor(unary_logits[pos_idx])
        prior_bucket = prior_lookup.get(int(position), {})
        prior_scores = torch.zeros(len(AA_ORDER), dtype=torch.float32)
        for aa_idx, aa in enumerate(AA_ORDER):
            if aa in prior_bucket:
                prior_scores[aa_idx] = float(prior_bucket[aa]["score"])
        if float(prior_scores.abs().sum().item()) > 0:
            prior_scores = _zscore_tensor(prior_scores)
        combined_scores = (1.0 - float(prior_field_weight)) * neural_scores + float(prior_field_weight) * prior_scores
        unary_probs = torch.softmax(combined_scores, dim=-1)
        ranked_indices = torch.argsort(combined_scores, descending=True).tolist()
        option_indices = ranked_indices[: int(top_k_per_position)]
        wt_idx = int(batch.wt_indices[pos_idx])
        if wt_idx not in option_indices:
            option_indices.append(wt_idx)
        option_indices = sorted(
            set(option_indices),
            key=lambda idx: (-float(combined_scores[idx]), AA_ORDER[idx]),
        )[: int(top_k_per_position)]
        options: list[ResidueOption] = []
        for aa_idx in option_indices:
            aa = AA_ORDER[int(aa_idx)]
            score = float(combined_scores[aa_idx])
            prob = float(unary_probs[aa_idx])
            supporting_sources = ["neural_field"]
            if aa in prior_bucket:
                supporting_sources.extend([src for src in prior_bucket[aa]["supporting_sources"] if src not in supporting_sources])
            options.append(
                ResidueOption(
                    residue=aa,
                    score=round(score, 6),
                    supporting_sources=supporting_sources,
                    support_strength=round(prob, 6),
                    evidence_breakdown={
                        "neural_unary": round(float(neural_scores[aa_idx]), 6),
                        "evidence_prior": round(float(prior_scores[aa_idx]), 6),
                        "combined_score": round(score, 6),
                        "neural_probability": round(prob, 6),
                    },
                )
            )
        fields.append(
            PositionField(
                position=int(position),
                wt_residue=wt_residue,
                options=options,
            )
        )

    pairwise_tensor: dict[tuple[int, int], dict[tuple[str, str], float]] = {}
    pair_diagnostics: dict[str, dict[str, float]] = {}
    for (i, j), matrix in output.pairwise.items():
        centered = matrix.detach().cpu() - matrix.detach().cpu().mean()
        centered = _zscore_tensor(centered.reshape(-1)).reshape(centered.shape)
        prior_matrix = _prior_pair_matrix((int(batch.positions[i]), int(batch.positions[j])), prior_pairwise)
        if float(prior_matrix.abs().sum().item()) > 0:
            prior_matrix = _zscore_tensor(prior_matrix.reshape(-1)).reshape(prior_matrix.shape)
        combined_pair = (1.0 - float(prior_pair_weight)) * centered + float(prior_pair_weight) * prior_matrix
        flat = combined_pair.reshape(-1)
        top_indices = torch.argsort(flat, descending=True).tolist()[: int(pair_top_k)]
        bucket: dict[tuple[str, str], float] = {}
        for flat_idx in top_indices:
            aa_i = flat_idx // len(AA_ORDER)
            aa_j = flat_idx % len(AA_ORDER)
            score = float(combined_pair[aa_i, aa_j])
            if score <= 0:
                continue
            bucket[(AA_ORDER[int(aa_i)], AA_ORDER[int(aa_j)])] = round(score, 6)
        if bucket:
            pair_key = (int(batch.positions[i]), int(batch.positions[j]))
            pairwise_tensor[pair_key] = bucket
            pair_diagnostics[f"{pair_key[0]}-{pair_key[1]}"] = {
                "max_score": round(max(bucket.values()), 6),
                "mean_score": round(sum(bucket.values()) / max(1, len(bucket)), 6),
                "support_count": float(len(bucket)),
            }

    gate_means = output.gates.mean(dim=0).detach().cpu().numpy().tolist()
    diagnostics = {
        "gate_means": {
            "geom": round(float(gate_means[0]), 6),
            "phylo": round(float(gate_means[1]), 6),
            "asr": round(float(gate_means[2]), 6),
            "retrieval": round(float(gate_means[3]), 6),
            "environment": round(float(gate_means[4]), 6),
        },
        "pairwise_edges": int(len(pairwise_tensor)),
        "pairwise_diagnostics": pair_diagnostics,
    }
    return fields, pairwise_tensor, diagnostics
