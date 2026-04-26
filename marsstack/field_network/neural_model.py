from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


AA_COUNT = 20


@dataclass
class NeuralFieldOutput:
    unary: torch.Tensor
    pairwise: dict[tuple[int, int], torch.Tensor]
    site_hidden: torch.Tensor
    gates: torch.Tensor
    asr_logits: torch.Tensor
    retrieval_logits: torch.Tensor
    env_reconstruction: torch.Tensor


class MarsFieldNeuralModel(nn.Module):
    def __init__(
        self,
        geom_dim: int = 6,
        evo_dim: int = AA_COUNT,
        asr_dim: int = AA_COUNT + 4,
        retrieval_dim: int = AA_COUNT + 5,
        env_dim: int = 3,
        candidate_dim: int = 40,
        hidden_dim: int = 64,
        pair_rank: int = 16,
        pair_feature_dim: int = 3,
        memory_slots: int = 12,
        branch_dropout_p: float = 0.10,
    ) -> None:
        super().__init__()
        self.geom_dim = geom_dim
        self.evo_dim = evo_dim
        self.asr_dim = asr_dim
        self.retrieval_dim = retrieval_dim
        self.env_dim = env_dim
        self.candidate_dim = candidate_dim
        self.hidden_dim = hidden_dim
        self.pair_rank = pair_rank
        self.pair_feature_dim = pair_feature_dim
        self.memory_slots = memory_slots
        self.branch_dropout_p = branch_dropout_p
        self.geom_proj = nn.Sequential(nn.Linear(geom_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.evo_proj = nn.Sequential(nn.Linear(evo_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.asr_proj = nn.Sequential(nn.Linear(asr_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.retr_proj = nn.Sequential(nn.Linear(retrieval_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.env_proj = nn.Sequential(nn.Linear(env_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.lineage_memory = nn.Parameter(torch.randn(memory_slots, hidden_dim) * 0.05)
        self.retrieval_memory = nn.Parameter(torch.randn(memory_slots, hidden_dim) * 0.05)
        self.lineage_memory_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.retrieval_memory_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.env_scale = nn.Linear(hidden_dim, hidden_dim * 4)
        self.env_bias = nn.Linear(hidden_dim, hidden_dim * 4)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )
        self.site_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residue_prototypes = nn.Parameter(torch.randn(AA_COUNT, hidden_dim) * 0.05)
        self.unary_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.candidate_token_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.candidate_proj = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.pair_summary_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.candidate_selection_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.candidate_engineering_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.candidate_policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.asr_aux_head = nn.Linear(hidden_dim, AA_COUNT)
        self.retrieval_aux_head = nn.Linear(hidden_dim, AA_COUNT)
        self.env_recon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, env_dim),
        )
        self.pair_ctx = nn.Sequential(
            nn.Linear(hidden_dim * 3 + pair_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pair_rank),
        )
        self.pair_proto = nn.Linear(hidden_dim, pair_rank, bias=False)

    @classmethod
    def from_batch(cls, batch, hidden_dim: int = 64, pair_rank: int = 16) -> "MarsFieldNeuralModel":
        pair_feature_dim = 3
        if batch.pair_inputs:
            sample = next(iter(batch.pair_inputs.values()))
            pair_feature_dim = int(len(sample))
        return cls(
            geom_dim=int(batch.geom_inputs.shape[1]),
            evo_dim=int(batch.evo_inputs.shape[1]),
            asr_dim=int(batch.asr_inputs.shape[1]),
            retrieval_dim=int(batch.retrieval_inputs.shape[1]),
            env_dim=int(batch.env_inputs.shape[0]),
            candidate_dim=int(batch.candidate_features.shape[1]),
            hidden_dim=hidden_dim,
            pair_rank=pair_rank,
            pair_feature_dim=pair_feature_dim,
        )

    def export_config(self) -> dict[str, int]:
        return {
            "geom_dim": int(self.geom_dim),
            "evo_dim": int(self.evo_dim),
            "asr_dim": int(self.asr_dim),
            "retrieval_dim": int(self.retrieval_dim),
            "env_dim": int(self.env_dim),
            "candidate_dim": int(self.candidate_dim),
            "hidden_dim": int(self.hidden_dim),
            "pair_rank": int(self.pair_rank),
            "pair_feature_dim": int(self.pair_feature_dim),
            "memory_slots": int(self.memory_slots),
            "branch_dropout_p": float(self.branch_dropout_p),
        }

    def _memory_fuse(
        self,
        branch_hidden: torch.Tensor,
        memory_bank: torch.Tensor,
        fuse_layer: nn.Module,
    ) -> torch.Tensor:
        attention = torch.softmax(branch_hidden @ memory_bank.t() / (self.hidden_dim ** 0.5), dim=-1)
        memory_context = attention @ memory_bank
        return fuse_layer(torch.cat([branch_hidden, memory_context, branch_hidden * memory_context], dim=-1))

    def encode_sites(
        self,
        geom_inputs: torch.Tensor,
        evo_inputs: torch.Tensor,
        asr_inputs: torch.Tensor,
        retrieval_inputs: torch.Tensor,
        env_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_geom = self.geom_proj(geom_inputs)
        h_evo = self.evo_proj(evo_inputs)
        h_asr = self.asr_proj(asr_inputs)
        h_retr = self.retr_proj(retrieval_inputs)
        h_asr = self._memory_fuse(h_asr, self.lineage_memory, self.lineage_memory_fuse)
        h_retr = self._memory_fuse(h_retr, self.retrieval_memory, self.retrieval_memory_fuse)
        env_base = self.env_proj(env_inputs)
        env_token = env_base.unsqueeze(0).expand_as(h_geom)

        env_scale = torch.tanh(self.env_scale(env_base)).reshape(4, self.hidden_dim)
        env_bias = self.env_bias(env_base).reshape(4, self.hidden_dim)
        h_geom = h_geom * (1.0 + env_scale[0]) + env_bias[0]
        h_evo = h_evo * (1.0 + env_scale[1]) + env_bias[1]
        h_asr = h_asr * (1.0 + env_scale[2]) + env_bias[2]
        h_retr = h_retr * (1.0 + env_scale[3]) + env_bias[3]

        if self.training and self.branch_dropout_p > 0:
            keep = torch.bernoulli(
                torch.full((h_geom.size(0), 4), 1.0 - self.branch_dropout_p, device=h_geom.device)
            )
            dead_rows = keep.sum(dim=1) == 0
            if dead_rows.any():
                keep[dead_rows, 0] = 1.0
            h_geom = h_geom * keep[:, 0:1]
            h_evo = h_evo * keep[:, 1:2]
            h_asr = h_asr * keep[:, 2:3]
            h_retr = h_retr * keep[:, 3:4]
        stacked = torch.cat([h_geom, h_evo, h_asr, h_retr, env_token], dim=-1)
        alpha = torch.softmax(self.gate(stacked), dim=-1)
        mixed = torch.cat(
            [
                alpha[:, 0:1] * h_geom,
                alpha[:, 1:2] * h_evo,
                alpha[:, 2:3] * h_asr,
                alpha[:, 3:4] * h_retr,
                alpha[:, 4:5] * env_token,
            ],
            dim=-1,
        )
        return self.site_mlp(mixed), alpha

    def forward(
        self,
        geom_inputs: torch.Tensor,
        evo_inputs: torch.Tensor,
        asr_inputs: torch.Tensor,
        retrieval_inputs: torch.Tensor,
        env_inputs: torch.Tensor,
        pair_features: dict[tuple[int, int], torch.Tensor],
    ) -> NeuralFieldOutput:
        site_hidden, alpha = self.encode_sites(
            geom_inputs=geom_inputs,
            evo_inputs=evo_inputs,
            asr_inputs=asr_inputs,
            retrieval_inputs=retrieval_inputs,
            env_inputs=env_inputs,
        )

        proto = self.residue_prototypes.unsqueeze(0).expand(site_hidden.size(0), -1, -1)
        site_expand = site_hidden.unsqueeze(1).expand(-1, AA_COUNT, -1)
        unary_in = torch.cat([site_expand, proto, site_expand * proto], dim=-1)
        unary = self.unary_mlp(unary_in).squeeze(-1)
        asr_logits = self.asr_aux_head(site_hidden)
        retrieval_logits = self.retrieval_aux_head(site_hidden)
        env_reconstruction = self.env_recon_head(site_hidden.mean(dim=0))

        pairwise: dict[tuple[int, int], torch.Tensor] = {}
        proto_pair = self.pair_proto(self.residue_prototypes)
        for (i, j), pair_feat in pair_features.items():
            pair_vec = self.pair_ctx(
                torch.cat(
                    [
                        site_hidden[i],
                        site_hidden[j],
                        torch.abs(site_hidden[i] - site_hidden[j]),
                        pair_feat,
                    ],
                    dim=-1,
                )
            )
            pair_matrix = torch.einsum("ar,r,br->ab", proto_pair, pair_vec, proto_pair)
            pairwise[(i, j)] = pair_matrix

        return NeuralFieldOutput(
            unary=unary,
            pairwise=pairwise,
            site_hidden=site_hidden,
            gates=alpha,
            asr_logits=asr_logits,
            retrieval_logits=retrieval_logits,
            env_reconstruction=env_reconstruction,
        )

    def sequence_energy(
        self,
        output: NeuralFieldOutput,
        sequence_indices: torch.Tensor,
    ) -> torch.Tensor:
        unary = output.unary.gather(1, sequence_indices.t()).t().sum(dim=1)
        pair_energy = torch.zeros_like(unary)
        for (i, j), matrix in output.pairwise.items():
            pair_energy = pair_energy + matrix[sequence_indices[:, i], sequence_indices[:, j]]
        return unary + pair_energy

    def candidate_embedding(
        self,
        output: NeuralFieldOutput,
        sequence_indices: torch.Tensor,
        candidate_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selected_proto = self.residue_prototypes[sequence_indices]
        site_hidden = output.site_hidden.unsqueeze(0).expand(sequence_indices.size(0), -1, -1)
        fused = torch.cat([site_hidden, selected_proto, site_hidden * selected_proto], dim=-1)
        sequence_context = self.candidate_token_proj(fused.mean(dim=1))
        if candidate_inputs is None:
            candidate_context = torch.zeros_like(sequence_context)
        else:
            candidate_context = self.candidate_proj(candidate_inputs)

        if output.pairwise:
            pair_values = []
            for (i, j), matrix in output.pairwise.items():
                pair_values.append(matrix[sequence_indices[:, i], sequence_indices[:, j]])
            pair_tensor = torch.stack(pair_values, dim=1)
            pair_summary = torch.stack(
                [
                    pair_tensor.mean(dim=1),
                    pair_tensor.max(dim=1).values,
                    pair_tensor.min(dim=1).values,
                    torch.full_like(pair_tensor.mean(dim=1), float(pair_tensor.size(1)) / 10.0),
                ],
                dim=-1,
            )
        else:
            pair_summary = torch.zeros((sequence_indices.size(0), 4), dtype=sequence_context.dtype, device=sequence_context.device)
        pair_context = self.pair_summary_proj(pair_summary)
        return torch.cat([sequence_context, candidate_context, pair_context], dim=-1)

    def candidate_selection_scores(
        self,
        output: NeuralFieldOutput,
        sequence_indices: torch.Tensor,
        candidate_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.candidate_embedding(output, sequence_indices, candidate_inputs=candidate_inputs)
        return self.candidate_selection_head(embedding).squeeze(-1)

    def candidate_engineering_scores(
        self,
        output: NeuralFieldOutput,
        sequence_indices: torch.Tensor,
        candidate_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.candidate_embedding(output, sequence_indices, candidate_inputs=candidate_inputs)
        return self.candidate_engineering_head(embedding).squeeze(-1)

    def candidate_policy_scores(
        self,
        output: NeuralFieldOutput,
        sequence_indices: torch.Tensor,
        candidate_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.candidate_embedding(output, sequence_indices, candidate_inputs=candidate_inputs)
        return self.candidate_policy_head(embedding).squeeze(-1)
