from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .neural_model import MarsFieldNeuralModel


def tensorize_pair_inputs(pair_inputs: dict[tuple[int, int], Any]) -> dict[tuple[int, int], torch.Tensor]:
    return {
        pair: torch.tensor(values, dtype=torch.float32)
        for pair, values in pair_inputs.items()
    }


def _empirical_pair_targets(batch) -> dict[tuple[int, int], torch.Tensor]:
    targets: dict[tuple[int, int], torch.Tensor] = {}
    if not batch.pair_inputs:
        return targets
    scores = torch.tensor(batch.candidate_scores, dtype=torch.float32)
    weights = torch.softmax((scores - scores.mean()) / (scores.std() + 1e-6), dim=0)
    residues = torch.tensor(batch.candidate_indices, dtype=torch.long)
    for (i, j) in batch.pair_inputs.keys():
        matrix = torch.zeros((20, 20), dtype=torch.float32)
        for row_idx in range(residues.size(0)):
            aa_i = int(residues[row_idx, i])
            aa_j = int(residues[row_idx, j])
            matrix[aa_i, aa_j] += float(weights[row_idx])
        if matrix.sum() > 0:
            matrix = matrix / matrix.sum()
        targets[(i, j)] = matrix
    return targets


def _alignment_loss(
    unary_logits: torch.Tensor,
    target_distributions: torch.Tensor,
    site_weights: torch.Tensor,
) -> torch.Tensor:
    if target_distributions.numel() == 0:
        return torch.tensor(0.0)
    valid = site_weights > 1e-8
    if int(valid.sum().item()) == 0:
        return torch.tensor(0.0)
    log_probs = torch.log_softmax(unary_logits, dim=-1)
    per_site = -(target_distributions * log_probs).sum(dim=-1)
    weighted = per_site * site_weights
    return weighted[valid].sum() / site_weights[valid].sum()


def _candidate_site_targets(batch) -> torch.Tensor:
    residues = torch.tensor(batch.candidate_indices, dtype=torch.long)
    selection_targets = torch.tensor(batch.candidate_scores, dtype=torch.float32)
    selection_targets = (selection_targets - selection_targets.mean()) / (selection_targets.std() + 1e-6)
    engineering_targets = torch.tensor(batch.candidate_mars_scores, dtype=torch.float32)
    engineering_targets = (engineering_targets - engineering_targets.mean()) / (engineering_targets.std() + 1e-6)
    policy_targets = 0.30 * selection_targets + 0.70 * engineering_targets
    weights = torch.softmax(policy_targets * 1.5, dim=0)
    targets = torch.zeros((residues.size(1), 20), dtype=torch.float32)
    for row_idx in range(residues.size(0)):
        for pos_idx in range(residues.size(1)):
            aa = int(residues[row_idx, pos_idx])
            targets[pos_idx, aa] += float(weights[row_idx])
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return targets


def _pairwise_rank_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    diff_targets = targets.unsqueeze(1) - targets.unsqueeze(0)
    mask = diff_targets > float(margin)
    if int(mask.sum().item()) == 0:
        return predictions.new_tensor(0.0)
    diff_preds = predictions.unsqueeze(1) - predictions.unsqueeze(0)
    return F.softplus(-diff_preds[mask]).mean()


def _winner_guard_loss(
    predictions: torch.Tensor,
    engineering_targets: torch.Tensor,
    policy_targets: torch.Tensor,
    margin: float = 0.12,
    engineering_gap: float = 0.45,
    policy_gap: float = 0.20,
) -> torch.Tensor:
    anchor_idx = int(torch.argmax(policy_targets).item())
    bad_mask = (
        (engineering_targets < engineering_targets[anchor_idx] - float(engineering_gap))
        & (policy_targets < policy_targets[anchor_idx] - float(policy_gap))
    )
    bad_mask[anchor_idx] = False
    if int(bad_mask.sum().item()) == 0:
        return predictions.new_tensor(0.0)
    anchor_pred = predictions[anchor_idx]
    return F.softplus(predictions[bad_mask] - anchor_pred + float(margin)).mean()


def _non_decoder_guard_loss(
    predictions: torch.Tensor,
    engineering_targets: torch.Tensor,
    candidate_sources: list[str],
    margin: float = 0.10,
    engineering_gap: float = 0.35,
) -> torch.Tensor:
    if not candidate_sources:
        return predictions.new_tensor(0.0)
    non_decoder_mask = torch.tensor(
        [str(source) != "fusion_decoder" for source in candidate_sources],
        dtype=torch.bool,
        device=predictions.device,
    )
    decoder_mask = ~non_decoder_mask
    if int(non_decoder_mask.sum().item()) == 0 or int(decoder_mask.sum().item()) == 0:
        return predictions.new_tensor(0.0)
    non_decoder_indices = torch.nonzero(non_decoder_mask, as_tuple=False).flatten()
    anchor_local = int(torch.argmax(engineering_targets[non_decoder_mask]).item())
    anchor_idx = int(non_decoder_indices[anchor_local].item())
    risky_decoder_mask = decoder_mask & (engineering_targets < engineering_targets[anchor_idx] - float(engineering_gap))
    if int(risky_decoder_mask.sum().item()) == 0:
        return predictions.new_tensor(0.0)
    anchor_pred = predictions[anchor_idx]
    return F.softplus(predictions[risky_decoder_mask] - anchor_pred + float(margin)).mean()


def _simplicity_guard_loss(
    predictions: torch.Tensor,
    engineering_targets: torch.Tensor,
    candidate_mutations: list[str],
    margin: float = 0.08,
    engineering_slack: float = 0.15,
) -> torch.Tensor:
    if not candidate_mutations:
        return predictions.new_tensor(0.0)
    mutation_counts = torch.tensor(
        [
            0.0 if not str(mutations).strip() or str(mutations).strip().upper() == "WT" else float(len([token for token in str(mutations).split(";") if token.strip()]))
            for mutations in candidate_mutations
        ],
        dtype=torch.float32,
        device=predictions.device,
    )
    anchor_idx = int(torch.argmax(engineering_targets - 0.10 * mutation_counts).item())
    risky_mask = (
        (mutation_counts > mutation_counts[anchor_idx])
        & (engineering_targets < engineering_targets[anchor_idx] + float(engineering_slack))
    )
    risky_mask[anchor_idx] = False
    if int(risky_mask.sum().item()) == 0:
        return predictions.new_tensor(0.0)
    anchor_pred = predictions[anchor_idx]
    return F.softplus(predictions[risky_mask] - anchor_pred + float(margin)).mean()


def _selector_anchor_loss(
    predictions: torch.Tensor,
    engineering_targets: torch.Tensor,
    candidate_mutations: list[str],
    margin: float = 0.12,
    engineering_gap: float = 0.10,
) -> torch.Tensor:
    if predictions.numel() == 0:
        return predictions.new_tensor(0.0)
    mutation_counts = torch.tensor(
        [
            0.0 if not str(mutations).strip() or str(mutations).strip().upper() == "WT" else float(len([token for token in str(mutations).split(";") if token.strip()]))
            for mutations in candidate_mutations
        ],
        dtype=torch.float32,
        device=predictions.device,
    )
    anchor_idx = 0
    risky_mask = (
        (engineering_targets < engineering_targets[anchor_idx] - float(engineering_gap))
        | (
            (mutation_counts > mutation_counts[anchor_idx])
            & (engineering_targets < engineering_targets[anchor_idx] + float(engineering_gap))
        )
    )
    risky_mask[anchor_idx] = False
    if int(risky_mask.sum().item()) == 0:
        return predictions.new_tensor(0.0)
    anchor_pred = predictions[anchor_idx]
    return F.softplus(predictions[risky_mask] - anchor_pred + float(margin)).mean()


def _gate_prior_from_batch(batch) -> torch.Tensor:
    env = torch.tensor(batch.env_inputs, dtype=torch.float32)
    base = torch.tensor([0.26, 0.22, 0.10, 0.22, 0.20], dtype=torch.float32)
    homolog_signal = float(env[3].item()) if env.numel() > 3 else 0.0
    asr_signal = float(env[4].item()) if env.numel() > 4 else 0.0
    family_signal = float(env[5].item()) if env.numel() > 5 else 0.0
    template_signal = float(env[6].item()) if env.numel() > 6 else 0.0
    asr_flag = float(env[7].item()) if env.numel() > 7 else 0.0

    base[1] += 0.18 * min(1.0, homolog_signal + family_signal)
    base[2] += 0.22 * min(1.0, asr_signal + asr_flag)
    base[3] += 0.08 * min(1.0, 0.5 + homolog_signal)
    base[4] += 0.10 * template_signal

    # Keep geometry and retrieval from collapsing when priors are weak/strong.
    base[0] = max(0.10, float(base[0]))
    base[3] = min(0.32, float(base[3]))
    prior = torch.clamp(base, min=1e-6)
    return prior / prior.sum()


def train_model(
    train_batches,
    epochs: int,
    lr: float,
    regression_weight: float = 1.0,
    selection_head_weight: float = 0.55,
    engineering_head_weight: float = 0.35,
    policy_head_weight: float = 0.80,
    policy_pair_weight: float = 0.35,
    decoder_field_weight: float = 0.22,
    winner_guard_weight: float = 0.30,
    non_decoder_guard_weight: float = 0.20,
    simplicity_guard_weight: float = 0.16,
    selector_anchor_weight: float = 0.24,
    recovery_weight: float = 0.20,
    pairwise_weight: float = 0.12,
    ancestry_weight: float = 0.10,
    retrieval_weight: float = 0.08,
    environment_weight: float = 0.05,
    gate_entropy_weight: float = 0.03,
    gate_prior_weight: float = 0.10,
):
    model = MarsFieldNeuralModel.from_batch(train_batches[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    history: list[dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        total_loss = 0.0
        total_reg = 0.0
        total_sel = 0.0
        total_eng = 0.0
        total_pol = 0.0
        total_pol_rank = 0.0
        total_decoder = 0.0
        total_guard = 0.0
        total_non_decoder_guard = 0.0
        total_simplicity_guard = 0.0
        total_selector_anchor = 0.0
        total_rec = 0.0
        total_pair = 0.0
        total_asr = 0.0
        total_retr = 0.0
        total_env = 0.0
        total_gate = 0.0
        total_gate_prior = 0.0
        total_items = 0
        model.train()
        for batch in train_batches:
            output = model(
                geom_inputs=torch.tensor(batch.geom_inputs),
                evo_inputs=torch.tensor(batch.evo_inputs),
                asr_inputs=torch.tensor(batch.asr_inputs),
                retrieval_inputs=torch.tensor(batch.retrieval_inputs),
                env_inputs=torch.tensor(batch.env_inputs),
                pair_features=tensorize_pair_inputs(batch.pair_inputs),
            )
            sequence_indices = torch.tensor(batch.candidate_indices)
            candidate_inputs = torch.tensor(batch.candidate_features, dtype=torch.float32)
            energies = model.sequence_energy(
                output=output,
                sequence_indices=sequence_indices,
            )
            selection_pred = model.candidate_selection_scores(output, sequence_indices, candidate_inputs=candidate_inputs)
            engineering_pred = model.candidate_engineering_scores(output, sequence_indices, candidate_inputs=candidate_inputs)
            policy_pred = model.candidate_policy_scores(output, sequence_indices, candidate_inputs=candidate_inputs)
            selection_targets = torch.tensor(batch.candidate_scores, dtype=torch.float32)
            selection_targets = (selection_targets - selection_targets.mean()) / (selection_targets.std() + 1e-6)
            engineering_targets = torch.tensor(batch.candidate_mars_scores, dtype=torch.float32)
            engineering_targets = (engineering_targets - engineering_targets.mean()) / (engineering_targets.std() + 1e-6)
            policy_targets = 0.30 * selection_targets + 0.70 * engineering_targets
            regression_loss = F.mse_loss(energies, selection_targets)
            selection_head_loss = F.mse_loss(selection_pred, selection_targets)
            engineering_head_loss = F.mse_loss(engineering_pred, engineering_targets)
            policy_head_loss = F.mse_loss(policy_pred, policy_targets)
            policy_rank_loss = _pairwise_rank_loss(policy_pred, policy_targets)
            decoder_targets = _candidate_site_targets(batch)
            decoder_field_loss = _alignment_loss(
                unary_logits=output.unary,
                target_distributions=decoder_targets,
                site_weights=torch.ones(decoder_targets.size(0), dtype=torch.float32),
            )
            winner_guard_loss = _winner_guard_loss(policy_pred, engineering_targets, policy_targets)
            non_decoder_guard_loss = _non_decoder_guard_loss(policy_pred, engineering_targets, batch.candidate_sources)
            simplicity_guard_loss = _simplicity_guard_loss(policy_pred, engineering_targets, batch.candidate_mutations)
            selector_anchor_loss = _selector_anchor_loss(policy_pred, engineering_targets, batch.candidate_mutations)

            recovery_loss = F.cross_entropy(output.unary, torch.tensor(batch.wt_indices, dtype=torch.long))

            asr_targets = torch.tensor(batch.asr_inputs[:, :20], dtype=torch.float32)
            asr_weights = torch.tensor(batch.asr_inputs[:, 20], dtype=torch.float32)
            ancestry_loss = _alignment_loss(
                unary_logits=output.asr_logits,
                target_distributions=asr_targets,
                site_weights=asr_weights,
            )

            retrieval_targets = torch.tensor(batch.retrieval_inputs[:, :20], dtype=torch.float32)
            retrieval_weights = torch.clamp(torch.tensor(batch.retrieval_inputs[:, 20], dtype=torch.float32), 0.0, 1.0)
            retrieval_loss = _alignment_loss(
                unary_logits=output.retrieval_logits,
                target_distributions=retrieval_targets,
                site_weights=retrieval_weights,
            )

            env_target = torch.tensor(batch.env_inputs, dtype=torch.float32)
            environment_loss = F.mse_loss(output.env_reconstruction, env_target)

            gate_mean = output.gates.mean(dim=0)
            gate_entropy = -(gate_mean * torch.log(gate_mean + 1e-8)).sum() / torch.log(torch.tensor(float(output.gates.size(1))))
            gate_regularization = 1.0 - gate_entropy
            gate_prior = _gate_prior_from_batch(batch)
            gate_prior_loss = F.mse_loss(gate_mean, gate_prior)

            pair_targets = _empirical_pair_targets(batch)
            if pair_targets:
                pair_losses = []
                for pair_key, target_matrix in pair_targets.items():
                    pred = torch.softmax(output.pairwise[pair_key].reshape(-1), dim=0).reshape(20, 20)
                    pair_losses.append(F.mse_loss(pred, target_matrix))
                pairwise_loss = torch.stack(pair_losses).mean()
            else:
                pairwise_loss = torch.tensor(0.0)

            loss = (
                regression_weight * regression_loss
                + selection_head_weight * selection_head_loss
                + engineering_head_weight * engineering_head_loss
                + policy_head_weight * policy_head_loss
                + policy_pair_weight * policy_rank_loss
                + decoder_field_weight * decoder_field_loss
                + winner_guard_weight * winner_guard_loss
                + non_decoder_guard_weight * non_decoder_guard_loss
                + simplicity_guard_weight * simplicity_guard_loss
                + selector_anchor_weight * selector_anchor_loss
                + recovery_weight * recovery_loss
                + pairwise_weight * pairwise_loss
                + ancestry_weight * ancestry_loss
                + retrieval_weight * retrieval_loss
                + environment_weight * environment_loss
                + gate_entropy_weight * gate_regularization
                + gate_prior_weight * gate_prior_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_items = len(batch.candidate_scores)
            total_loss += float(loss.item()) * batch_items
            total_reg += float(regression_loss.item()) * batch_items
            total_sel += float(selection_head_loss.item()) * batch_items
            total_eng += float(engineering_head_loss.item()) * batch_items
            total_pol += float(policy_head_loss.item()) * batch_items
            total_pol_rank += float(policy_rank_loss.item()) * batch_items
            total_decoder += float(decoder_field_loss.item()) * batch_items
            total_guard += float(winner_guard_loss.item()) * batch_items
            total_non_decoder_guard += float(non_decoder_guard_loss.item()) * batch_items
            total_simplicity_guard += float(simplicity_guard_loss.item()) * batch_items
            total_selector_anchor += float(selector_anchor_loss.item()) * batch_items
            total_rec += float(recovery_loss.item()) * batch_items
            total_pair += float(pairwise_loss.item()) * batch_items
            total_asr += float(ancestry_loss.item()) * batch_items
            total_retr += float(retrieval_loss.item()) * batch_items
            total_env += float(environment_loss.item()) * batch_items
            total_gate += float(gate_regularization.item()) * batch_items
            total_gate_prior += float(gate_prior_loss.item()) * batch_items
            total_items += batch_items

        epoch_loss = total_loss / max(1, total_items)
        epoch_reg = total_reg / max(1, total_items)
        epoch_sel = total_sel / max(1, total_items)
        epoch_eng = total_eng / max(1, total_items)
        epoch_pol = total_pol / max(1, total_items)
        epoch_pol_rank = total_pol_rank / max(1, total_items)
        epoch_decoder = total_decoder / max(1, total_items)
        epoch_guard = total_guard / max(1, total_items)
        epoch_non_decoder_guard = total_non_decoder_guard / max(1, total_items)
        epoch_simplicity_guard = total_simplicity_guard / max(1, total_items)
        epoch_selector_anchor = total_selector_anchor / max(1, total_items)
        epoch_rec = total_rec / max(1, total_items)
        epoch_pair = total_pair / max(1, total_items)
        epoch_asr = total_asr / max(1, total_items)
        epoch_retr = total_retr / max(1, total_items)
        epoch_env = total_env / max(1, total_items)
        epoch_gate = total_gate / max(1, total_items)
        epoch_gate_prior = total_gate_prior / max(1, total_items)
        history.append(
            {
                "epoch": float(epoch),
                "loss": round(epoch_loss, 6),
                "regression_loss": round(epoch_reg, 6),
                "selection_head_loss": round(epoch_sel, 6),
                "engineering_head_loss": round(epoch_eng, 6),
                "policy_head_loss": round(epoch_pol, 6),
                "policy_rank_loss": round(epoch_pol_rank, 6),
                "decoder_field_loss": round(epoch_decoder, 6),
                "winner_guard_loss": round(epoch_guard, 6),
                "non_decoder_guard_loss": round(epoch_non_decoder_guard, 6),
                "simplicity_guard_loss": round(epoch_simplicity_guard, 6),
                "selector_anchor_loss": round(epoch_selector_anchor, 6),
                "recovery_loss": round(epoch_rec, 6),
                "pairwise_loss": round(epoch_pair, 6),
                "ancestry_loss": round(epoch_asr, 6),
                "retrieval_loss": round(epoch_retr, 6),
                "environment_loss": round(epoch_env, 6),
                "gate_regularization": round(epoch_gate, 6),
                "gate_prior_loss": round(epoch_gate_prior, 6),
            }
        )
        print(
            f"epoch={epoch} loss={epoch_loss:.6f} "
            f"reg={epoch_reg:.6f} sel={epoch_sel:.6f} eng={epoch_eng:.6f} pol={epoch_pol:.6f} prank={epoch_pol_rank:.6f} dec={epoch_decoder:.6f} "
            f"guard={epoch_guard:.6f} nondec={epoch_non_decoder_guard:.6f} simple={epoch_simplicity_guard:.6f} anchor={epoch_selector_anchor:.6f} "
            f"rec={epoch_rec:.6f} pair={epoch_pair:.6f} asr={epoch_asr:.6f} retr={epoch_retr:.6f} env={epoch_env:.6f} "
            f"gate={epoch_gate:.6f} gprior={epoch_gate_prior:.6f}"
        )

    return model, history


def score_batch(model: MarsFieldNeuralModel, batch):
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
        energies = model.sequence_energy(
            output=output,
            sequence_indices=torch.tensor(batch.candidate_indices),
        )
        selection_pred = model.candidate_selection_scores(
            output=output,
            sequence_indices=torch.tensor(batch.candidate_indices),
            candidate_inputs=torch.tensor(batch.candidate_features, dtype=torch.float32),
        )
        engineering_pred = model.candidate_engineering_scores(
            output=output,
            sequence_indices=torch.tensor(batch.candidate_indices),
            candidate_inputs=torch.tensor(batch.candidate_features, dtype=torch.float32),
        )
        policy_pred = model.candidate_policy_scores(
            output=output,
            sequence_indices=torch.tensor(batch.candidate_indices),
            candidate_inputs=torch.tensor(batch.candidate_features, dtype=torch.float32),
        )
    return {
        "energies": energies.detach().cpu().numpy(),
        "selection_pred": selection_pred.detach().cpu().numpy(),
        "engineering_pred": engineering_pred.detach().cpu().numpy(),
        "policy_pred": policy_pred.detach().cpu().numpy(),
        "gates": output.gates.detach().cpu(),
    }
