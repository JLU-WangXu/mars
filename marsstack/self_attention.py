from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: float = 0.0,
    scale: Optional[float] = None,
) -> Tensor:
    """Compute scaled dot-product attention.

    Args:
        query: Query tensor of shape (batch, heads, seq_len, d_k)
        key: Key tensor of shape (batch, heads, seq_len, d_k)
        value: Value tensor of shape (batch, heads, seq_len, d_v)
        mask: Optional attention mask of shape (batch, 1, 1, seq_len) or
              (batch, 1, seq_len, seq_len) for causal masking
        dropout: Dropout probability applied to attention weights
        scale: Scaling factor. Defaults to sqrt(d_k) if not provided

    Returns:
        Attention output tensor of shape (batch, heads, seq_len, d_v)
    """
    d_k = query.size(-1)
    if scale is None:
        scale = math.sqrt(d_k)

    # Compute attention scores: (batch, heads, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / scale

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Apply dropout during training
    if dropout > 0.0 and torch.is_grad_enabled():
        attention_weights = F.dropout(attention_weights, p=dropout)

    # Compute weighted sum of values
    output = torch.matmul(attention_weights, value)
    return output


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for long-range dependency modeling.

    Implements the self-attention mechanism where queries, keys, and values
    are all derived from the same input sequence, enabling the model to
    capture relationships between all positions in the sequence.

    Args:
        d_model: Model dimension (input and output feature size)
        num_heads: Number of attention heads
        dropout: Dropout probability for attention weights
        bias: Whether to include bias in linear projections
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.dropout = dropout

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass of multi-head self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention: If True, return attention weights as well

        Returns:
            If return_attention is False: output tensor (batch, seq_len, d_model)
            If return_attention is True: tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections and reshape to (batch, seq_len, heads, d_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_v)

        # Transpose to (batch, heads, seq_len, d_k) for attention computation
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute scaled dot-product attention
        attn_output = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )

        # Concatenate heads: (batch, seq_len, num_heads * d_v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Final output projection
        output = self.W_o(attn_output)

        if return_attention:
            # Compute attention weights for visualization/analysis
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            attention_weights = F.softmax(scores, dim=-1)
            return output, attention_weights

        return output


class CausalSelfAttention(nn.Module):
    """Causal self-attention that prevents attending to future positions.

    Uses a triangular mask to ensure each position can only attend to
    positions at or before itself in the sequence. Essential for
    autoregressive sequence modeling.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length (for pre-computing causal mask)
        dropout: Dropout probability
        bias: Whether to include bias in linear projections
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=dropout)

        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            ),
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with causal masking.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional additional mask to apply

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose for attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention with causal mask
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Apply causal mask (lower triangular)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0.0:
            attention_weights = self.dropout_layer(attention_weights)

        # Compute output
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)

        return self.W_o(output)


class LongRangeAttention(nn.Module):
    """Efficient attention mechanism for modeling long-range dependencies.

    Implements attention with relative position encoding to capture
    positional relationships between elements. Uses chunked computation
    for improved efficiency on long sequences.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        chunk_size: Size of chunks for efficient computation
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        chunk_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.chunk_size = chunk_size

        # Projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Relative position bias
        self.max_rel_pos = chunk_size * 2
        self.relative_bias = nn.Parameter(torch.zeros(2 * self.max_rel_pos + 1, num_heads))

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=dropout)

    def _compute_relative_positions(self, seq_len: int) -> Tensor:
        """Compute relative position indices for all pairs in sequence."""
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(1) - range_vec.unsqueeze(0)
        # Clamp to valid range
        range_mat = torch.clamp(range_mat, -self.max_rel_pos, self.max_rel_pos)
        # Offset to make all indices non-negative
        return range_mat + self.max_rel_pos

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with relative position encoding.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention per chunk for efficiency
        if seq_len > self.chunk_size:
            output = self._chunked_attention(Q, K, V, mask)
        else:
            output = self._full_attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        return self.W_o(output)

    def _full_attention(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        """Standard full attention computation."""
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Add relative position bias
        rel_positions = self._compute_relative_positions(Q.size(2)).to(Q.device)
        rel_bias = self.relative_bias[rel_positions]  # (seq_len, seq_len, num_heads)
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, seq_len, seq_len)
        scores = scores + rel_bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0.0:
            attn_weights = self.dropout_layer(attn_weights)

        return torch.matmul(attn_weights, V)

    def _chunked_attention(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        """Chunked attention for long sequences."""
        batch_size, num_heads, seq_len, d_k = Q.shape
        chunk_size = self.chunk_size

        # Reshape into chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        # Pad sequence to be divisible by chunk size
        if seq_len % chunk_size != 0:
            pad_len = chunk_size - (seq_len % chunk_size)
            Q = F.pad(Q, (0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, pad_len))
            if mask is not None:
                mask = F.pad(mask, (0, pad_len, 0, pad_len) if mask.dim() == 4 else (0, pad_len))

        # Reshape for chunked processing
        Q = Q.view(batch_size, num_heads, -1, chunk_size, d_k)
        K = K.view(batch_size, num_heads, -1, chunk_size, d_k)
        V = V.view(batch_size, num_heads, -1, chunk_size, d_k)

        outputs = []
        for i in range(num_chunks):
            chunk_q = Q[:, :, i]  # (batch, heads, chunk, d_k)

            # Attend to all previous chunks and current chunk
            key_chunks = K[:, :, : i + 1]  # (batch, heads, i+1, chunk, d_k)
            val_chunks = V[:, :, : i + 1]

            # Flatten for matmul
            key_chunks = key_chunks.reshape(batch_size, num_heads, -1, d_k)
            val_chunks = val_chunks.reshape(batch_size, num_heads, -1, d_k)

            # Compute scores with relative position
            scale = math.sqrt(self.d_k)
            scores = torch.matmul(chunk_q, key_chunks.transpose(-2, -1)) / scale

            # Relative position within context
            rel_pos_offset = i * chunk_size
            chunk_positions = torch.arange(chunk_size, device=Q.device)
            key_positions = torch.arange((i + 1) * chunk_size, device=Q.device) - rel_pos_offset
            rel_mat = chunk_positions.unsqueeze(1) - key_positions.unsqueeze(0)
            rel_mat = torch.clamp(rel_mat, -self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
            rel_bias = self.relative_bias[rel_mat]  # (chunk, num_chunks*chunk, heads)
            rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)  # (1, heads, chunk, context)
            scores = scores + rel_bias

            if mask is not None:
                chunk_mask = mask[:, :, i * chunk_size : (i + 1) * chunk_size, : (i + 1) * chunk_size]
                scores = scores.masked_fill(chunk_mask == 0, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            if self.dropout > 0.0:
                attn_weights = self.dropout_layer(attn_weights)

            chunk_out = torch.matmul(attn_weights, val_chunks)
            outputs.append(chunk_out)

        output = torch.cat(outputs, dim=2)
        # Trim to original sequence length
        return output[:, :, :seq_len, :]
