"""Attention Pooling Module for Enhanced Feature Aggregation.

This module provides attention-based pooling mechanisms for aggregating
features from multiple sources in protein design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionPoolConfig:
    """Configuration for attention pooling layers."""

    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.1
    use_bias: bool = True
    temperature: float = 1.0
    routing_iterations: int = 3
    epsilon: float = 1e-8


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling for aggregating feature sequences.

    This layer applies multi-head self-attention followed by pooling to create
    a fixed-size representation from variable-length feature sequences.

    Args:
        config: Attention pooling configuration
        input_dim: Dimension of input features
    """

    def __init__(
        self,
        input_dim: int,
        config: AttentionPoolConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or AttentionPoolConfig()
        self.input_dim = input_dim
        self.hidden_dim = self.config.hidden_dim
        self.num_heads = self.config.num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        # Linear projections
        self.query_proj = nn.Linear(input_dim, self.hidden_dim, bias=self.config.use_bias)
        self.key_proj = nn.Linear(input_dim, self.hidden_dim, bias=self.config.use_bias)
        self.value_proj = nn.Linear(input_dim, self.hidden_dim, bias=self.config.use_bias)
        self.output_proj = nn.Linear(self.hidden_dim, input_dim, bias=self.config.use_bias)

        self.dropout = nn.Dropout(self.config.dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply multi-head attention pooling.

        Args:
            features: Input features of shape (batch, seq_len, input_dim)
            mask: Optional attention mask of shape (batch, seq_len)

        Returns:
            Pooled features of shape (batch, input_dim)
        """
        batch_size, seq_len, _ = features.shape

        # Project to hidden dimension
        q = self.query_proj(features)  # (batch, seq_len, hidden_dim)
        k = self.key_proj(features)
        v = self.value_proj(features)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scale = math.sqrt(self.head_dim) * self.config.temperature
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Apply mask if provided
        if mask is not None:
            mask = mask.view(batch_size, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        # Output projection and residual connection
        output = self.output_proj(attn_output)
        output = self.dropout(output)
        output = self.layer_norm(features + output)

        # Mean pooling over sequence dimension
        if mask is not None:
            mask_expanded = mask.view(batch_size, seq_len, 1).float()
            pooled = (output * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + self.config.epsilon)
        else:
            pooled = output.mean(dim=1)

        return pooled

    def get_attention_weights(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get attention weights for visualization or analysis.

        Args:
            features: Input features of shape (batch, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Attention weights of shape (batch, num_heads, seq_len, seq_len)
        """
        q = self.query_proj(features)
        k = self.key_proj(features)

        q = q.view(features.size(0), features.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(features.size(0), features.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim) * self.config.temperature
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            mask = mask.view(features.size(0), 1, 1, features.size(1))
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        return F.softmax(attn_scores, dim=-1)


class WeightedSumPooling(nn.Module):
    """Learnable weighted sum pooling for feature aggregation.

    This layer learns attention weights to aggregate features from
    multiple sources with learned importance scores.

    Args:
        input_dim: Dimension of input features
        num_sources: Number of feature sources to aggregate
        config: Optional pooling configuration
    """

    def __init__(
        self,
        input_dim: int,
        num_sources: int,
        config: AttentionPoolConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or AttentionPoolConfig()
        self.input_dim = input_dim
        self.num_sources = num_sources

        # Source embeddings for computing importance
        self.source_embeddings = nn.Parameter(
            torch.randn(num_sources, input_dim) * 0.02
        )

        # Learnable query for weighted aggregation
        self.query = nn.Parameter(torch.randn(1, input_dim) * 0.02)

        # Optional feature transformation
        self.feature_proj = nn.Linear(input_dim, input_dim, bias=self.config.use_bias)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        features: torch.Tensor,
        source_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply weighted sum pooling.

        Args:
            features: Input features of shape (batch, num_sources, input_dim)
                     or (batch, seq_len, input_dim)
            source_ids: Optional source indices for named sources

        Returns:
            Pooled features of shape (batch, input_dim)
        """
        batch_size = features.size(0)

        # Compute attention scores
        if source_ids is not None and self.num_sources <= features.size(1):
            source_emb = self.source_embeddings[:features.size(1)]
        else:
            source_emb = self.source_embeddings[:self.num_sources]

        # Compute similarity with query
        query = self.query.expand(batch_size, -1)
        scores = torch.matmul(query, source_emb.t())  # (batch, num_sources)

        # Apply temperature scaling
        scores = scores / self.config.temperature

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_sources)

        # Aggregate features
        aggregated = torch.matmul(attn_weights, source_emb)  # (batch, input_dim)

        # Optional feature transformation
        transformed = self.feature_proj(aggregated)
        output = self.layer_norm(transformed)

        return output

    def get_weights(self) -> torch.Tensor:
        """Get current source weights.

        Returns:
            Source weights of shape (num_sources,)
        """
        return F.softmax(self.query @ self.source_embeddings.t(), dim=-1).squeeze()


class DynamicRoutingPooler(nn.Module):
    """Dynamic routing pooling for capsule-style feature aggregation.

    This layer uses iterative routing-by-agreement to compute
    pose-invariant feature representations.

    Args:
        input_dim: Dimension of input features
        output_dim: Dimension of output representation
        num_capsules: Number of output capsules
        config: Optional pooling configuration
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        num_capsules: int = 8,
        config: AttentionPoolConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or AttentionPoolConfig()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.num_capsules = num_capsules
        self.routing_iterations = self.config.routing_iterations

        # Transformation matrix for capsules
        self.capsule_transform = nn.Parameter(
            torch.randn(input_dim, self.output_dim, num_capsules) * 0.02
        )

        # Bias terms for capsules
        self.capsule_bias = nn.Parameter(torch.zeros(1, 1, num_capsules))

        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply dynamic routing pooling.

        Args:
            features: Input features of shape (batch, seq_len, input_dim)
            mask: Optional mask of shape (batch, seq_len)

        Returns:
            Capsule representations of shape (batch, num_capsules, output_dim)
        """
        batch_size, seq_len, _ = features.shape

        # Transform features to capsule space
        # (batch, seq_len, input_dim) -> (batch, seq_len, output_dim, num_capsules)
        transformed = torch.einsum("bij,jkl->bikl", features, self.capsule_transform)
        transformed = transformed + self.capsule_bias

        # Initialize routing weights uniformly
        routing_weights = torch.ones(
            batch_size, seq_len, self.num_capsules,
            device=features.device, dtype=features.dtype
        ) / seq_len

        # Iterative routing
        for _ in range(self.routing_iterations):
            # Weighted sum of transformed features
            capsule_input = torch.einsum("bsc,bsnc->bnc", routing_weights, transformed)

            # Apply squashing activation
            capsule_input_norm = torch.norm(capsule_input, dim=-2, keepdim=True)
            squash_factor = (capsule_input_norm ** 2) / (
                capsule_input_norm ** 2 + 1
            )
            capsule_output = squash_factor * capsule_input / (
                capsule_input_norm + self.config.epsilon
            )

            # Update routing weights
            if _ < self.routing_iterations - 1:
                agreement = torch.einsum("bnc,bikl->bcni", capsule_output, transformed)
                routing_weights = F.softmax(
                    torch.log(routing_weights + self.config.epsilon) + agreement,
                    dim=1
                )

        # Apply layer norm to outputs
        capsule_output = self.layer_norm(capsule_output)

        return capsule_output

    def get_routing_weights(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get final routing weights for analysis.

        Args:
            features: Input features
            mask: Optional mask

        Returns:
            Routing weights of shape (batch, seq_len, num_capsules)
        """
        batch_size, seq_len, _ = features.shape

        transformed = torch.einsum("bij,jkl->bikl", features, self.capsule_transform)
        routing_weights = torch.ones(
            batch_size, seq_len, self.num_capsules,
            device=features.device, dtype=features.dtype
        ) / seq_len

        for _ in range(self.routing_iterations):
            capsule_input = torch.einsum("bsc,bsnc->bnc", routing_weights, transformed)
            capsule_input_norm = torch.norm(capsule_input, dim=-2, keepdim=True)
            squash_factor = (capsule_input_norm ** 2) / (
                capsule_input_norm ** 2 + 1
            )
            capsule_output = squash_factor * capsule_input / (
                capsule_input_norm + self.config.epsilon
            )

            if _ < self.routing_iterations - 1:
                agreement = torch.einsum("bnc,bikl->bcni", capsule_output, transformed)
                routing_weights = F.softmax(
                    torch.log(routing_weights + self.config.epsilon) + agreement,
                    dim=1
                )

        return routing_weights


class AttentionPoolingEnsemble(nn.Module):
    """Ensemble of attention pooling methods for robust feature aggregation.

    This module combines multiple pooling strategies with learned combination weights.

    Args:
        input_dim: Dimension of input features
        num_heads: Number of attention heads for multi-head pooling
        num_capsules: Number of capsules for dynamic routing
        config: Optional pooling configuration
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        num_capsules: int = 8,
        config: AttentionPoolConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or AttentionPoolConfig()

        # Initialize pooling modules
        self.multihead_pool = MultiHeadAttentionPooling(
            input_dim=input_dim,
            config=self.config,
        )
        # Override num_heads if specified
        self.multihead_pool.num_heads = num_heads
        self.multihead_pool.head_dim = self.config.hidden_dim // num_heads
        self.multihead_pool.query_proj = nn.Linear(
            input_dim, self.config.hidden_dim, bias=self.config.use_bias
        )
        self.multihead_pool.key_proj = nn.Linear(
            input_dim, self.config.hidden_dim, bias=self.config.use_bias
        )
        self.multihead_pool.value_proj = nn.Linear(
            input_dim, self.config.hidden_dim, bias=self.config.use_bias
        )
        self.multihead_pool.output_proj = nn.Linear(
            self.config.hidden_dim, input_dim, bias=self.config.use_bias
        )

        self.weighted_pool = WeightedSumPooling(
            input_dim=input_dim,
            num_sources=num_capsules,
            config=self.config,
        )

        self.dynamic_pool = DynamicRoutingPooler(
            input_dim=input_dim,
            num_capsules=num_capsules,
            config=self.config,
        )

        # Combination weights (learnable)
        self.combination_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        pool_type: Literal["multihead", "weighted", "dynamic", "ensemble"] = "ensemble",
    ) -> torch.Tensor:
        """Apply attention pooling with specified strategy.

        Args:
            features: Input features of shape (batch, seq_len, input_dim)
            mask: Optional attention mask
            pool_type: Pooling strategy to use

        Returns:
            Pooled features
        """
        if pool_type == "multihead":
            return self.multihead_pool(features, mask)

        elif pool_type == "weighted":
            return self.weighted_pool(features)

        elif pool_type == "dynamic":
            capsules = self.dynamic_pool(features, mask)
            return capsules.mean(dim=1)  # Average over capsules

        else:  # ensemble
            pooled_multihead = self.multihead_pool(features, mask)
            pooled_weighted = self.weighted_pool(features)
            capsules = self.dynamic_pool(features, mask)
            pooled_dynamic = capsules.mean(dim=1)

            # Combine with learned weights
            weights = F.softmax(self.combination_weights, dim=0)
            combined = (
                weights[0] * pooled_multihead
                + weights[1] * pooled_weighted
                + weights[2] * pooled_dynamic
            )

            return combined

    def get_pooling_weights(self) -> dict[str, float]:
        """Get current combination weights.

        Returns:
            Dictionary of pooling method weights
        """
        weights = F.softmax(self.combination_weights, dim=0).detach().cpu()
        return {
            "multihead": weights[0].item(),
            "weighted": weights[1].item(),
            "dynamic": weights[2].item(),
        }


def aggregate_decoder_features(
    features_list: list[torch.Tensor],
    pooling_type: Literal["multihead", "weighted", "dynamic", "ensemble"] = "ensemble",
    config: AttentionPoolConfig | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate features from decoder sources using attention pooling.

    This is a utility function for integrating attention pooling
    with the existing decoder pipeline.

    Args:
        features_list: List of feature tensors to aggregate
        pooling_type: Type of pooling to use
        config: Pooling configuration
        mask: Optional mask tensor

    Returns:
        Aggregated feature tensor
    """
    if not features_list:
        raise ValueError("features_list cannot be empty")

    if len(features_list) == 1:
        return features_list[0]

    # Stack features along sequence dimension
    stacked = torch.stack(features_list, dim=1)  # (batch, num_sources, seq_len, dim)

    batch_size, num_sources, seq_len, feature_dim = stacked.shape

    # Flatten sources into sequence
    flattened = stacked.view(batch_size, num_sources * seq_len, feature_dim)

    # Create appropriate mask
    if mask is None:
        mask = torch.ones(batch_size, num_sources * seq_len, device=stacked.device)
    else:
        mask = mask.unsqueeze(1).expand(-1, num_sources, -1).reshape(batch_size, -1)

    # Apply pooling
    config = config or AttentionPoolConfig()
    ensemble = AttentionPoolingEnsemble(
        input_dim=feature_dim,
        config=config,
    )

    return ensemble(flattened, mask=mask, pool_type=pooling_type)


# ============================================================================
# Integration with Decoder
# ============================================================================

def apply_attention_pooling_to_decoder(
    position_features: torch.Tensor,
    residue_scores: torch.Tensor,
    pooling_config: AttentionPoolConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply attention pooling to enhance decoder feature aggregation.

    This function can be used to post-process decoder outputs with
    attention-based feature aggregation.

    Args:
        position_features: Position-level features (batch, num_positions, feature_dim)
        residue_scores: Residue-level scores (batch, num_positions, num_residues)
        pooling_config: Attention pooling configuration

    Returns:
        Tuple of (pooled_features, attention_weights)
    """
    config = pooling_config or AttentionPoolConfig()

    # Apply multi-head attention pooling
    pooler = MultiHeadAttentionPooling(
        input_dim=position_features.size(-1),
        config=config,
    )

    pooled = pooler(position_features)  # (batch, feature_dim)

    # Get attention weights for analysis
    attn_weights = pooler.get_attention_weights(position_features)

    # Enhance residue scores with pooled context
    # (batch, num_positions, feature_dim) @ (batch, feature_dim, 1)
    pooled_expanded = pooled.unsqueeze(-1)
    score_enhancement = torch.matmul(position_features, pooled_expanded).squeeze(-1)

    enhanced_scores = residue_scores + score_enhancement.unsqueeze(-1) * 0.1

    return enhanced_scores, attn_weights


def create_attention_pooled_fields(
    features: torch.Tensor,
    top_k: int = 4,
) -> torch.Tensor:
    """Create top-k pooled representation for field generation.

    Args:
        features: Input features (batch, seq_len, feature_dim)
        top_k: Number of top features to pool

    Returns:
        Pooled features (batch, top_k, feature_dim)
    """
    # Compute feature magnitudes
    magnitudes = torch.norm(features, dim=-1, keepdim=True)  # (batch, seq_len, 1)

    # Get top-k indices
    _, top_indices = magnitudes.squeeze(-1).topk(min(top_k, features.size(1)), dim=-1)

    # Gather top-k features
    batch_size = features.size(0)
    top_features = torch.zeros(
        batch_size, top_k, features.size(-1),
        device=features.device, dtype=features.dtype
    )

    for b in range(batch_size):
        top_features[b] = features[b, top_indices[b]]

    return top_features
