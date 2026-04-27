"""Knowledge Distillation Module for Model Compression.

This module implements knowledge distillation techniques to compress large neural
network models (teachers) into smaller, faster models (students) while preserving
prediction quality.

Features:
- Teacher-Student architecture with configurable model sizes
- Soft label distillation (Hinton et al., 2015)
- Feature map alignment for intermediate representations
- Compression ratio control via layer pruning and width scaling
- Multi-task distillation support

Example usage:
    from marsstack.knowledge_distillation import (
        TeacherModel, StudentModel, DistillationTrainer,
        DistillationConfig, create_compressed_model
    )

    # Create teacher and student
    teacher = TeacherModel(embedding_dim=1280, num_layers=33)
    student = StudentModel(embedding_dim=320, num_layers=6)

    # Configure distillation
    config = DistillationConfig(
        temperature=4.0,
        alpha=0.7,  # Weight for soft targets
        feature_weight=0.2,
        compression_ratio=0.25,
    )

    # Train student from teacher
    trainer = DistillationTrainer(teacher, student, config)
    trainer.train(train_loader, epochs=100)
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================


class DistillationStrategy(Enum):
    """Available distillation strategies."""
    SOFT_LABELS = "soft_labels"           # Hinton-style temperature scaling
    FEATURE_MATCHING = "feature_matching" # Intermediate layer alignment
    SELF_DISTILLATION = "self_distillation"  # Deep self-distillation
    CONTRASTIVE = "contrastive"          # Contrastive distillation


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation.

    Attributes:
        temperature: Softmax temperature for soft label generation.
            Higher values produce softer probability distributions.
        alpha: Weight balance between soft targets (alpha) and hard targets (1-alpha).
        beta: Weight for feature matching loss.
        gamma: Weight for contrastive loss.
        feature_weight: Weight for intermediate feature alignment.
        contrastive_weight: Weight for contrastive distillation.
        compression_ratio: Target compression ratio (student_size / teacher_size).
        use_hard_labels: Whether to include hard label supervision.
        feature_layers: List of layer indices for feature matching.
        temperature_schedule: Whether to anneal temperature during training.
        initial_temperature: Starting temperature for scheduling.
        final_temperature: Ending temperature for scheduling.
    """
    temperature: float = 4.0
    alpha: float = 0.7
    beta: float = 0.1
    gamma: float = 0.1
    feature_weight: float = 0.2
    contrastive_weight: float = 0.1
    compression_ratio: float = 0.25
    use_hard_labels: bool = True
    feature_layers: list[int] = field(default_factory=lambda: [0, -1])
    temperature_schedule: bool = False
    initial_temperature: float = 4.0
    final_temperature: float = 2.0
    strategy: DistillationStrategy = DistillationStrategy.SOFT_LABELS


@dataclass
class CompressionConfig:
    """Configuration for model compression.

    Attributes:
        depth_ratio: Ratio of student depth to teacher depth.
        width_ratio: Ratio of student width to teacher width.
        attention_heads_ratio: Ratio of attention heads.
        intermediate_ratio: Ratio of FFN intermediate dimension.
        prune_layers: Whether to prune entire layers.
        prune_attention_heads: Whether to prune attention heads.
        residual_pruning: Layer dropout during training for residual connections.
    """
    depth_ratio: float = 0.2
    width_ratio: float = 0.25
    attention_heads_ratio: float = 0.25
    intermediate_ratio: float = 0.25
    prune_layers: bool = True
    prune_attention_heads: bool = True
    residual_pruning: float = 0.1


# =============================================================================
# Model Architecture Classes
# =============================================================================


class AttentionLayer(nn.Module):
    """Multi-head attention layer with configurable size."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output), attn_probs


class FeedForwardLayer(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        embed_dim: int,
        intermediate_dim: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        intermediate_dim = intermediate_dim or embed_dim * 4
        self.linear1 = nn.Linear(embed_dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        intermediate_dim: int | None = None,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.attention = AttentionLayer(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForwardLayer(embed_dim, intermediate_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        attn_output, attn_probs = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_output)
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_output)
        return x, attn_probs


# =============================================================================
# Teacher and Student Models
# =============================================================================


class EmbeddingModel(nn.Module):
    """Base embedding model with extractable features."""

    def __init__(
        self,
        vocab_size: int = 21,
        embed_dim: int = 320,
        max_positions: int = 2048,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(max_positions, embed_dim)

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        return self.token_embedding(input_ids) + self.position_embedding(positions)


class TeacherModel(nn.Module):
    """Large teacher model for knowledge distillation.

    This model represents the larger, pre-trained model that will be
    compressed into a smaller student model.
    """

    def __init__(
        self,
        vocab_size: int = 21,
        embed_dim: int = 1280,
        num_layers: int = 33,
        num_heads: int = 20,
        intermediate_dim: int = 5120,
        dropout: float = 0.1,
        max_positions: int = 2048,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embedding = EmbeddingModel(vocab_size, embed_dim, max_positions)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, intermediate_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: Tensor,
        output_features: bool = False,
    ) -> dict[str, Tensor]:
        """Forward pass with optional intermediate features."""
        hidden_states: list[Tensor] = []
        x = self.embedding(input_ids)

        for i, layer in enumerate(self.layers):
            x, _ = layer(x)
            if output_features and i in [0, len(self.layers) // 2, len(self.layers) - 1]:
                hidden_states.append(x)

        logits = self.norm(x)

        if output_features:
            return {"logits": logits, "hidden_states": hidden_states}
        return {"logits": logits}

    def get_soft_labels(
        self,
        input_ids: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """Generate soft labels using temperature scaling."""
        outputs = self.forward(input_ids)
        logits = outputs["logits"]
        return F.softmax(logits / temperature, dim=-1)


class StudentModel(nn.Module):
    """Compact student model for knowledge distillation.

    This model learns from the teacher model through distillation.
    """

    def __init__(
        self,
        vocab_size: int = 21,
        embed_dim: int = 320,
        num_layers: int = 6,
        num_heads: int = 5,
        intermediate_dim: int = 1280,
        dropout: float = 0.1,
        max_positions: int = 2048,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embedding = EmbeddingModel(vocab_size, embed_dim, max_positions)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, intermediate_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: Tensor,
        output_features: bool = False,
    ) -> dict[str, Tensor]:
        """Forward pass with optional intermediate features."""
        hidden_states: list[Tensor] = []
        x = self.embedding(input_ids)

        for i, layer in enumerate(self.layers):
            x, _ = layer(x)
            if output_features and i in [0, len(self.layers) // 2, len(self.layers) - 1]:
                hidden_states.append(x)

        logits = self.norm(x)

        if output_features:
            return {"logits": logits, "hidden_states": hidden_states}
        return {"logits": logits}

    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Distillation Loss Functions
# =============================================================================


class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation."""

    def __init__(self, config: DistillationConfig) -> None:
        super().__init__()
        self.config = config
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def soft_target_loss(
        self,
        student_logits: Tensor,
        teacher_probs: Tensor,
        temperature: float,
    ) -> Tensor:
        """Compute soft target loss (KL divergence)."""
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        return self.kl_div(student_log_probs, teacher_probs) * (temperature ** 2)

    def hard_target_loss(
        self,
        student_logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Compute hard target loss (cross-entropy)."""
        return self.ce(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))

    def feature_matching_loss(
        self,
        student_features: list[Tensor],
        teacher_features: list[Tensor],
    ) -> Tensor:
        """Compute feature matching loss using MSE."""
        if len(student_features) != len(teacher_features):
            teacher_features = self._align_features(student_features, teacher_features)

        loss = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Align dimensions if needed
            if s_feat.shape != t_feat.shape:
                s_feat = F.adaptive_avg_pool1d(s_feat.transpose(1, 2), t_feat.shape[1]).transpose(1, 2)
            loss += self.mse(s_feat, t_feat)
        return loss / max(len(student_features), 1)

    def _align_features(
        self,
        student_features: list[Tensor],
        teacher_features: list[Tensor],
    ) -> list[Tensor]:
        """Align feature dimensions between student and teacher."""
        aligned = []
        for i, s_feat in enumerate(student_features):
            if i < len(teacher_features):
                t_feat = teacher_features[i]
            else:
                t_feat = teacher_features[-1]
            aligned.append(t_feat)
        return aligned

    def forward(
        self,
        student_logits: Tensor,
        teacher_probs: Tensor | None,
        targets: Tensor | None,
        student_features: list[Tensor] | None = None,
        teacher_features: list[Tensor] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Tensor]:
        """Compute total distillation loss."""
        temp = temperature or self.config.temperature
        total_loss = torch.tensor(0.0, device=student_logits.device)
        loss_dict: dict[str, Tensor] = {}

        # Soft target loss
        if teacher_probs is not None:
            soft_loss = self.soft_target_loss(student_logits, teacher_probs, temp)
            total_loss = total_loss + self.config.alpha * soft_loss
            loss_dict["soft_loss"] = soft_loss

        # Hard target loss
        if targets is not None and self.config.use_hard_labels:
            hard_loss = self.hard_target_loss(student_logits, targets)
            total_loss = total_loss + (1 - self.config.alpha) * hard_loss
            loss_dict["hard_loss"] = hard_loss

        # Feature matching loss
        if student_features is not None and teacher_features is not None:
            feat_loss = self.feature_matching_loss(student_features, teacher_features)
            total_loss = total_loss + self.config.feature_weight * feat_loss
            loss_dict["feature_loss"] = feat_loss

        loss_dict["total_loss"] = total_loss
        return loss_dict


# =============================================================================
# Distillation Trainer
# =============================================================================


class DistillationTrainer:
    """Trainer for knowledge distillation."""

    def __init__(
        self,
        teacher: TeacherModel,
        student: StudentModel,
        config: DistillationConfig,
        optimizer_config: dict[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Move models to device
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Loss function
        self.criterion = DistillationLoss(config)

        # Optimizer
        optimizer_config = optimizer_config or {"lr": 1e-4, "weight_decay": 1e-5}
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            **optimizer_config
        )

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.history: list[dict[str, float]] = []

    def compute_teacher_outputs(
        self,
        input_ids: Tensor,
        temperature: float,
    ) -> tuple[Tensor, list[Tensor] | None]:
        """Compute teacher outputs (soft labels and features)."""
        with torch.no_grad():
            use_features = self.config.feature_weight > 0
            outputs = self.teacher(input_ids, output_features=use_features)
            soft_labels = self.teacher.get_soft_labels(input_ids, temperature)
            features = outputs.get("hidden_states") if use_features else None
        return soft_labels, features

    def train_step(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Single training step."""
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device) if targets is not None else None

        # Compute temperature (with optional scheduling)
        if self.config.temperature_schedule:
            progress = self.current_epoch / max(self.config.epochs - 1, 1)
            temperature = self.config.initial_temperature + progress * (
                self.config.final_temperature - self.config.initial_temperature
            )
        else:
            temperature = self.config.temperature

        # Get teacher outputs
        soft_labels, teacher_features = self.compute_teacher_outputs(input_ids, temperature)

        # Student forward pass
        self.student.train()
        student_outputs = self.student(input_ids, output_features=self.config.feature_weight > 0)
        student_logits = student_outputs["logits"]
        student_features = student_outputs.get("hidden_states")

        # Compute loss
        loss_dict = self.criterion(
            student_logits=student_logits,
            teacher_probs=soft_labels,
            targets=targets,
            student_features=student_features,
            teacher_features=teacher_features,
            temperature=temperature,
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {k: v.item() for k, v in loss_dict.items()}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 100,
        callbacks: list[callable] | None = None,
    ) -> dict[str, Any]:
        """Train the student model."""
        self.config.epochs = epochs
        callbacks = callbacks or []

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_losses: list[dict[str, float]] = []

            # Training phase
            self.student.train()
            for batch in train_loader:
                input_ids = batch["input_ids"]
                targets = batch.get("targets")

                loss_dict = self.train_step(input_ids, targets)
                epoch_losses.append(loss_dict)

            # Aggregate training losses
            avg_losses = self._aggregate_losses(epoch_losses)

            # Validation phase
            val_losses = None
            if val_loader is not None:
                val_losses = self.evaluate(val_loader)
                avg_losses.update({f"val_{k}": v for k, v in val_losses.items()})

            self.history.append(avg_losses)

            # Log progress
            logger.info(f"Epoch {epoch + 1}/{epochs}: {avg_losses}")

            # Checkpoint best model
            if avg_losses.get("total_loss", float("inf")) < self.best_loss:
                self.best_loss = avg_losses["total_loss"]
                self.best_state = copy.deepcopy(self.student.state_dict())

            # Run callbacks
            for callback in callbacks:
                callback(epoch, avg_losses, self)

        return self._summarize_training()

    def evaluate(self, data_loader: DataLoader) -> dict[str, float]:
        """Evaluate the student model."""
        self.student.eval()
        all_losses: list[dict[str, float]] = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                targets = batch.get("targets")
                targets = targets.to(self.device) if targets is not None else None

                soft_labels, teacher_features = self.compute_teacher_outputs(
                    input_ids, self.config.temperature
                )
                student_outputs = self.student(input_ids, output_features=self.config.feature_weight > 0)

                loss_dict = self.criterion(
                    student_logits=student_outputs["logits"],
                    teacher_probs=soft_labels,
                    targets=targets,
                    student_features=student_outputs.get("hidden_states"),
                    teacher_features=teacher_features,
                )
                all_losses.append({k: v.item() for k, v in loss_dict.items()})

        return self._aggregate_losses(all_losses)

    def _aggregate_losses(self, losses: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate loss values across batches."""
        if not losses:
            return {}
        keys = losses[0].keys()
        return {k: sum(l.get(k, 0) for l in losses) / len(losses) for k in keys}

    def _summarize_training(self) -> dict[str, Any]:
        """Summarize training results."""
        # Restore best model
        if hasattr(self, "best_state"):
            self.student.load_state_dict(self.best_state)

        return {
            "best_loss": self.best_loss,
            "epochs_trained": len(self.history),
            "history": self.history,
            "compression_ratio": self.get_compression_ratio(),
            "student_params": self.student.num_parameters(),
        }

    def get_compression_ratio(self) -> float:
        """Calculate actual compression ratio achieved."""
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = self.student.num_parameters()
        return student_params / teacher_params

    def save(self, path: Path) -> None:
        """Save student model and training state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.student.state_dict(),
            "config": self.config,
            "history": self.history,
            "best_loss": self.best_loss,
        }, path)

    @classmethod
    def load(cls, path: Path, teacher: TeacherModel) -> "DistillationTrainer":
        """Load a trained distillation trainer."""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        student = StudentModel()  # Would need proper initialization
        student.load_state_dict(checkpoint["model_state_dict"])
        trainer = cls(teacher, student, config)
        trainer.history = checkpoint["history"]
        trainer.best_loss = checkpoint["best_loss"]
        return trainer


# =============================================================================
# Compression Utilities
# =============================================================================


def compute_compression_config(
    teacher: TeacherModel,
    target_ratio: float,
) -> CompressionConfig:
    """Compute compression configuration to achieve target ratio.

    Args:
        teacher: The teacher model to compress.
        target_ratio: Target compression ratio (0.0 to 1.0).

    Returns:
        CompressionConfig with appropriate settings.
    """
    # Estimate parameter counts
    teacher_params = sum(p.numel() for p in teacher.parameters())

    # Solve for width/depth ratio that achieves target
    # Assuming params ~ depth * width^2 (simplified)
    # target_params = depth_ratio * width_ratio^2 * teacher_params
    # target_ratio = depth_ratio * width_ratio^2

    # Start with depth reduction
    depth_ratio = max(0.1, min(1.0, target_ratio ** 0.5))
    width_ratio = (target_ratio / depth_ratio) ** 0.5

    return CompressionConfig(
        depth_ratio=depth_ratio,
        width_ratio=width_ratio,
        attention_heads_ratio=width_ratio,
        intermediate_ratio=width_ratio,
    )


def create_compressed_model(
    teacher: TeacherModel,
    compression_config: CompressionConfig,
) -> StudentModel:
    """Create a compressed student model from teacher.

    Args:
        teacher: The teacher model to compress.
        compression_config: Configuration for compression.

    Returns:
        A new StudentModel with compressed architecture.
    """
    # Compute compressed dimensions
    embed_dim = max(64, int(teacher.embed_dim * compression_config.width_ratio))
    embed_dim = (embed_dim // 64) * 64  # Ensure divisible by 64

    num_layers = max(1, int(teacher.num_layers * compression_config.depth_ratio))
    num_heads = max(1, int(teacher.num_heads * compression_config.attention_heads_ratio))
    intermediate_dim = max(256, int(teacher.num_layers * compression_config.intermediate_ratio))

    # Align heads
    num_heads = min(num_heads, embed_dim // 64) * 64 // 64

    return StudentModel(
        vocab_size=teacher.embedding.token_embedding.num_embeddings,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
        dropout=teacher.layers[0].dropout.pdrop if hasattr(teacher.layers[0], 'dropout') else 0.1,
        max_positions=teacher.embedding.position_embedding.num_embeddings,
    )


def progressive_distillation(
    teacher: TeacherModel,
    base_student: StudentModel,
    config: DistillationConfig,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    intermediate_ratios: list[float] | None = None,
) -> StudentModel:
    """Progressive knowledge distillation with intermediate models.

    Args:
        teacher: The teacher model.
        base_student: The final target student model.
        config: Distillation configuration.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        intermediate_ratios: Intermediate compression ratios.

    Returns:
        The trained student model.
    """
    intermediate_ratios = intermediate_ratios or [0.5, 0.3, 0.15]

    current_student = base_student

    for i, ratio in enumerate(intermediate_ratios + [config.compression_ratio]):
        logger.info(f"Progressive step {i + 1}/{len(intermediate_ratios) + 1}: ratio={ratio}")

        # Create intermediate model
        compression_config = compute_compression_config(teacher, ratio)
        intermediate_student = create_compressed_model(teacher, compression_config)

        # Initialize from current student if available
        if current_student is not base_student:
            intermediate_student = _initialize_from_larger_student(
                intermediate_student, current_student
            )

        # Train intermediate model
        step_config = DistillationConfig(
            temperature=config.temperature,
            alpha=config.alpha,
            feature_weight=config.feature_weight * (1 - i / len(intermediate_ratios)),
            compression_ratio=ratio,
        )

        trainer = DistillationTrainer(teacher, intermediate_student, step_config)
        trainer.train(train_loader, val_loader, epochs=config.epochs // 2)

        current_student = intermediate_student

    return current_student


def _initialize_from_larger_student(
    smaller: StudentModel,
    larger: StudentModel,
) -> StudentModel:
    """Initialize smaller student from larger student weights."""
    smaller_dict = smaller.state_dict()
    larger_dict = larger.state_dict()

    # Match layers by index
    for key in smaller_dict:
        if key in larger_dict and smaller_dict[key].shape == larger_dict[key].shape:
            smaller_dict[key] = larger_dict[key]

    smaller.load_state_dict(smaller_dict)
    return smaller


# =============================================================================
# Model Conversion Utilities
# =============================================================================


def convert_pretrained_to_student(
    pretrained_path: Path,
    compression_config: CompressionConfig,
    output_path: Path | None = None,
) -> StudentModel:
    """Convert a pretrained teacher model to a compressed student.

    Args:
        pretrained_path: Path to pretrained teacher weights.
        compression_config: Compression configuration.
        output_path: Optional path to save the student model.

    Returns:
        The compressed student model.
    """
    # Load teacher
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    teacher = TeacherModel()
    if "model_state_dict" in checkpoint:
        teacher.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        teacher.load_state_dict(checkpoint["state_dict"])

    # Create student
    student = create_compressed_model(teacher, compression_config)

    # Save if output path provided
    if output_path:
        torch.save({"student_state_dict": student.state_dict()}, output_path)

    return student


def export_for_inference(
    model: StudentModel,
    output_path: Path,
    quantize: bool = True,
) -> None:
    """Export student model for inference deployment.

    Args:
        model: The student model to export.
        output_path: Path to save the exported model.
        quantize: Whether to apply quantization.
    """
    model.eval()

    if quantize:
        # Dynamic quantization for smaller model size
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Embedding}, dtype=torch.qint8
        )
        torch.save({
            "model_state_dict": quantized_model.state_dict(),
            "quantized": True,
            "num_params": model.num_parameters(),
        }, output_path)
    else:
        torch.save({
            "model_state_dict": model.state_dict(),
            "quantized": False,
            "num_params": model.num_parameters(),
        }, output_path)


# =============================================================================
# Integration Helpers
# =============================================================================


class DistillationWrapper(nn.Module):
    """Wrapper to add distillation capabilities to existing models.

    This wrapper can be used to add knowledge distillation to models
    that don't natively support it.
    """

    def __init__(
        self,
        model: nn.Module,
        is_teacher: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.is_teacher = is_teacher
        self._features: dict[str, Tensor] = {}

        # Register hooks to capture intermediate features
        self._hooks: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks to capture intermediate features."""
        def hook_fn(name: str):
            def fn(module, input, output):
                self._features[name] = output
            return fn

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, TransformerBlock)):
                self._hooks.append(module.register_forward_hook(hook_fn(name)))

    def get_features(self) -> dict[str, Tensor]:
        """Get captured intermediate features."""
        return self._features.copy()

    def clear_features(self) -> None:
        """Clear captured features."""
        self._features.clear()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with feature capture."""
        self.clear_features()
        return self.model(x)


# =============================================================================
# Dataset Classes
# =============================================================================


class DistillationDataset(Dataset):
    """Dataset for knowledge distillation training."""

    def __init__(
        self,
        sequences: list[str],
        labels: list[int] | None = None,
        vocab: dict[str, int] | None = None,
        max_length: int = 512,
    ) -> None:
        self.sequences = sequences
        self.labels = labels
        self.vocab = vocab or self._default_vocab()
        self.max_length = max_length

    @staticmethod
    def _default_vocab() -> dict[str, int]:
        """Default amino acid vocabulary."""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        return {aa: i + 1 for i, aa in enumerate(amino_acids)}

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        seq = self.sequences[idx]
        input_ids = self._encode_sequence(seq)

        item: dict[str, Tensor] = {"input_ids": input_ids}

        if self.labels is not None:
            item["targets"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

    def _encode_sequence(self, seq: str) -> Tensor:
        """Encode sequence to token IDs."""
        seq = seq.upper().replace("*", "").replace(" ", "")[:self.max_length]
        encoded = [self.vocab.get(aa, 0) for aa in seq]

        # Pad to max_length
        if len(encoded) < self.max_length:
            encoded += [0] * (self.max_length - len(encoded))

        return torch.tensor(encoded, dtype=torch.long)


def create_distillation_loader(
    sequences: list[str],
    labels: list[int] | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for distillation training."""
    dataset = DistillationDataset(sequences, labels, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# =============================================================================
# API Functions
# =============================================================================


def create_distilled_model(
    teacher: TeacherModel,
    target_compression: float = 0.25,
    train_sequences: list[str] | None = None,
    train_labels: list[int] | None = None,
    val_sequences: list[str] | None = None,
    val_labels: list[int] | None = None,
    epochs: int = 100,
    device: str | None = None,
) -> tuple[StudentModel, dict[str, Any]]:
    """Create a distilled model from teacher.

    This is a high-level API function that handles the complete
    distillation pipeline.

    Args:
        teacher: The teacher model.
        target_compression: Target compression ratio.
        train_sequences: Training sequences.
        train_labels: Training labels (optional).
        val_sequences: Validation sequences (optional).
        val_labels: Validation labels (optional).
        epochs: Number of training epochs.
        device: Device to use for training.

    Returns:
        Tuple of (trained_student_model, training_info).
    """
    # Compute compression config
    compression_config = compute_compression_config(teacher, target_compression)
    student = create_compressed_model(teacher, compression_config)

    # Default config
    config = DistillationConfig(
        compression_ratio=target_compression,
        temperature=4.0,
        alpha=0.7,
        feature_weight=0.2,
    )

    # Create trainer
    trainer = DistillationTrainer(teacher, student, config, device=device)

    # Train if data provided
    training_info: dict[str, Any] = {
        "compression_ratio": trainer.get_compression_ratio(),
        "student_params": student.num_parameters(),
    }

    if train_sequences:
        train_loader = create_distillation_loader(train_sequences, train_labels)
        val_loader = create_distillation_loader(val_sequences, val_labels) if val_sequences else None

        result = trainer.train(train_loader, val_loader, epochs=epochs)
        training_info.update(result)

    return student, training_info


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "DistillationConfig",
    "DistillationStrategy",
    "CompressionConfig",
    # Models
    "TeacherModel",
    "StudentModel",
    "TransformerBlock",
    "AttentionLayer",
    "FeedForwardLayer",
    "EmbeddingModel",
    # Training
    "DistillationTrainer",
    "DistillationLoss",
    "DistillationWrapper",
    # Utilities
    "DistillationDataset",
    "create_distillation_loader",
    "create_compressed_model",
    "compute_compression_config",
    "progressive_distillation",
    "convert_pretrained_to_student",
    "export_for_inference",
    "create_distilled_model",
]
