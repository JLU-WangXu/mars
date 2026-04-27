"""Transfer learning module for protein engineering using large-scale pretrained models.

This module provides:
- Pretrained model loading (ESM, ProtBERT, AlphaFold)
- Feature transfer and embedding extraction
- Fine-tuning interface with frozen/flexible layers
- Thermostability task adaptation

Example usage:
    # Load pretrained models
    loader = PretrainedModelLoader(cache_dir="./models")
    esm_model = loader.load_esm("esm2_150M")

    # Extract features
    extractor = ProteinFeatureExtractor(esm_model)
    features = extractor.extract_sequence_features("MKTVRQERL...")

    # Fine-tune for thermostability
    fine_tuner = ThermostabilityFineTuner(esm_model)
    fine_tuner.freeze_encoder_layers(num_layers=30)
    fine_tuner.add_task_head(task_type="thermostability")

    trainer = TransferLearningTrainer(fine_tuner, train_loader, val_loader)
    trainer.train(epochs=10)
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


# =============================================================================
# Model Types and Configurations
# =============================================================================


class PretrainedModelType(Enum):
    """Supported pretrained protein model types."""
    ESM2_8M = "esm2_8M"
    ESM2_35M = "esm2_35M"
    ESM2_150M = "esm2_150M"
    ESM2_650M = "esm2_650M"
    ESM1B = "esm1b"
    PROTBERT = "protbert"
    ALPHAFOLD = "alphafold"


# ESM Model configurations
ESM_MODEL_CONFIGS = {
    PretrainedModelType.ESM2_8M: {
        "name": "esm2_t6_8M_UR50D",
        "url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt",
        "regression_url": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t6_8M_UR50D-contact-regression.pt",
        "embedding_dim": 320,
        "num_layers": 6,
        "attention_heads": 8,
    },
    PretrainedModelType.ESM2_35M: {
        "name": "esm2_t12_35M_UR50D",
        "url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt",
        "regression_url": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t12_35M_UR50D-contact-regression.pt",
        "embedding_dim": 480,
        "num_layers": 12,
        "attention_heads": 12,
    },
    PretrainedModelType.ESM2_150M: {
        "name": "esm2_t30_150M_UR50D",
        "url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt",
        "regression_url": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t30_150M_UR50D-contact-regression.pt",
        "embedding_dim": 640,
        "num_layers": 30,
        "attention_heads": 20,
    },
    PretrainedModelType.ESM2_650M: {
        "name": "esm2_t33_650M_UR50D",
        "url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
        "regression_url": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt",
        "embedding_dim": 1280,
        "num_layers": 33,
        "attention_heads": 40,
    },
    PretrainedModelType.ESM1B: {
        "name": "esm1b_t34_670M_UR50S",
        "url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t34_670M_UR50S.pt",
        "regression_url": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t34_670M_UR50S-contact-regression.pt",
        "embedding_dim": 1280,
        "num_layers": 34,
        "attention_heads": 40,
    },
}

# ProtBERT configuration
PROTBERT_CONFIG = {
    "model_name": "Rostlab/prot_bert",
    "embedding_dim": 1024,
    "max_position": 1024,
}

# AlphaFold configuration
ALPHAFOLD_CONFIG = {
    "model_name": "alphaFold",
    "embedding_dim": 384,
    "max_seq_len": 1024,
}


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ProteinEmbedding:
    """Container for protein embeddings from pretrained models."""
    sequence: str
    embedding: np.ndarray
    model_type: PretrainedModelType
    layer_index: int | None = None
    attention_weights: np.ndarray | None = None
    representation_type: str = "mean"  # "mean", "cls", "per_residue"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return self.embedding.shape[-1]

    def __len__(self) -> int:
        return len(self.sequence)


@dataclass
class TransferLearningConfig:
    """Configuration for transfer learning."""
    model_type: PretrainedModelType
    frozen_layers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    total_epochs: int = 20
    batch_size: int = 16
    gradient_clip_norm: float = 1.0
    dropout_rate: float = 0.1
    pooling_strategy: str = "mean"  # "mean", "cls", "attention"
    use_representation: str = "sequence"  # "sequence", "structure", "hybrid"


@dataclass
class ThermostabilityLabel:
    """Label for thermostability prediction tasks."""
    sequence: str
    mutation: str | None = None  # e.g., "A123G"
    tm_value: float | None = None  # Melting temperature
    delta_tm: float | None = None  # Change in Tm
    stability_class: str | None = None  # "thermostable", "mesostable", "psychrophile"
    source: str = "unknown"
    experimental_conditions: dict[str, Any] = field(default_factory=dict)


@dataclass
class FineTuningResult:
    """Results from fine-tuning process."""
    train_loss: list[float]
    val_loss: list[float]
    val_metrics: dict[str, list[float]]
    best_epoch: int
    best_model_path: str
    frozen_layers: int
    total_params: int
    trainable_params: int


# =============================================================================
# Model Loading
# =============================================================================


class ModelLoaderProtocol(Protocol):
    """Protocol for model loaders."""

    def load(self, device: str | None = None) -> nn.Module:
        """Load model to specified device."""
        ...


class PretrainedModelLoader:
    """Loader for pretrained protein language models."""

    def __init__(self, cache_dir: str | Path = "./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models: dict[PretrainedModelType, nn.Module] = {}

    def _get_model_path(self, model_type: PretrainedModelType) -> Path:
        """Get local path for model file."""
        config = ESM_MODEL_CONFIGS.get(model_type)
        if config is None:
            raise ValueError(f"Unknown ESM model type: {model_type}")
        filename = f"{config['name']}.pt"
        return self.cache_dir / filename

    def _download_esm_model(self, model_type: PretrainedModelType) -> Path:
        """Download ESM model from FAIR's server."""
        config = ESM_MODEL_CONFIGS[model_type]
        model_path = self._get_model_path(model_type)

        if model_path.exists():
            logger.info(f"Model already cached at {model_path}")
            return model_path

        logger.info(f"Downloading {model_type.value} from {config['url']}")
        import urllib.request

        try:
            urllib.request.urlretrieve(config["url"], model_path)
            logger.info(f"Downloaded to {model_path}")
        except Exception as e:
            logger.warning(f"Download failed: {e}. Will use torch hub fallback.")
            if model_path.exists():
                model_path.unlink()
            raise

        return model_path

    def load_esm(
        self,
        model_type: PretrainedModelType = PretrainedModelType.ESM2_150M,
        device: str | None = None,
        reload: bool = False,
    ) -> nn.Module:
        """Load ESM model.

        Args:
            model_type: ESM model variant to load
            device: Target device (cuda/cpu)
            reload: Force reload even if cached

        Returns:
            Loaded ESM model
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_type in self._loaded_models and not reload:
            logger.info(f"Using cached model: {model_type.value}")
            return self._loaded_models[model_type].to(device)

        try:
            import esm
        except ImportError:
            raise ImportError(
                "ESM not installed. Install with: pip install fair-esm"
            )

        config = ESM_MODEL_CONFIGS[model_type]
        num_layers = config["num_layers"]
        embedding_dim = config["embedding_dim"]

        # Try torch hub first, fallback to direct download
        model_name = f"esm2_t{num_layers}_{config['name'].split('_')[-1]}"
        try:
            model_data = torch.hub.load_state_dict_from_url(
                config["url"], progress=True
            )
        except Exception:
            model_path = self._download_esm_model(model_type)
            model_data = torch.load(model_path, map_location=device)

        # Rebuild model from state dict
        if isinstance(model_data, dict) and "model_state_dict" in model_data:
            state_dict = model_data["model_state_dict"]
        elif isinstance(model_data, dict) and "state_dict" in model_data:
            state_dict = model_data["state_dict"]
        else:
            state_dict = model_data

        # Create model structure
        repr_dim = embedding_dim
        model = esm.ProteinBERTPretrainedModel(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            attention_heads=config["attention_heads"],
        )

        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()

        self._loaded_models[model_type] = model
        logger.info(f"Loaded {model_type.value} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

        return model

    def load_protbert(self, device: str | None = None) -> nn.Module:
        """Load ProtBERT model from HuggingFace.

        Args:
            device: Target device

        Returns:
            Loaded ProtBERT model with tokenizer
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Transformers not installed. Install with: pip install transformers"
            )

        model_name = PROTBERT_CONFIG["model_name"]
        logger.info(f"Loading ProtBERT from {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()

        # Wrap with tokenizer for convenience
        wrapper = ProtBERTWrapper(model, tokenizer)
        self._loaded_models[PretrainedModelType.PROTBERT] = wrapper

        return wrapper

    def load_alphafold(self, device: str | None = None) -> nn.Module:
        """Load AlphaFold model for structure prediction.

        Args:
            device: Target device

        Returns:
            AlphaFold model wrapper
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            import alphafold
        except ImportError:
            logger.warning(
                "AlphaFold not installed. Structure features will use ESM embeddings."
            )
            return None

        logger.info("Loading AlphaFold model")
        # AlphaFold is typically used through its pipeline, not as a simple nn.Module
        # Return a wrapper that provides structure embeddings
        wrapper = AlphaFoldWrapper(device=device)
        self._loaded_models[PretrainedModelType.ALPHAFOLD] = wrapper

        return wrapper

    def load(
        self,
        model_type: PretrainedModelType,
        device: str | None = None,
    ) -> nn.Module:
        """Generic load method dispatching to appropriate loader.

        Args:
            model_type: Type of model to load
            device: Target device

        Returns:
            Loaded model
        """
        if model_type in [
            PretrainedModelType.ESM2_8M,
            PretrainedModelType.ESM2_35M,
            PretrainedModelType.ESM2_150M,
            PretrainedModelType.ESM2_650M,
            PretrainedModelType.ESM1B,
        ]:
            return self.load_esm(model_type, device)
        elif model_type == PretrainedModelType.PROTBERT:
            return self.load_protbert(device)
        elif model_type == PretrainedModelType.ALPHAFOLD:
            return self.load_alphafold(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class ProteinBERTPretrainedModel(nn.Module):
    """Generic protein BERT-style pretrained model."""

    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        attention_heads: int,
        vocab_size: int = 33,
        max_positions: int = 1024,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.embed_tokens = nn.Embedding(vocab_size, embedding_dim)
        self.embed_positions = nn.Embedding(max_positions, embedding_dim)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embedding_dim=embedding_dim,
                num_heads=attention_heads,
                dropout=0.1,
            )
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        repr_layer: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
            tokens: Token IDs [batch, seq_len]
            repr_layer: Layer index for representation (None for final)

        Returns:
            Tuple of (representations, all_layer_outputs)
        """
        B, L = tokens.shape
        positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)

        x = self.embed_tokens(tokens) + self.embed_positions(positions)
        x = F.layer_norm(x, (self.embedding_dim,))

        layer_outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if repr_layer is None or i < repr_layer:
                layer_outputs.append(x)

        x = self.layer_norm(x)
        return x, layer_outputs


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.linear2 = nn.Linear(embedding_dim * 4, embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention with pre-norm
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with pre-norm
        ff_out = self.linear2(F.gelu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_out))

        return x


class ProtBERTWrapper(nn.Module):
    """Wrapper for ProtBERT model with tokenizer."""

    def __init__(self, model: nn.Module, tokenizer: Any):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_dim = PROTBERT_CONFIG["embedding_dim"]

    def forward(
        self,
        sequences: list[str],
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Extract embeddings from sequences.

        Args:
            sequences: List of protein sequences
            pooling: Pooling strategy ("mean", "cls", "max")

        Returns:
            Embeddings [batch, dim]
        """
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(next(self.model.parameters()).device)

        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state

        if pooling == "cls":
            embeddings = hidden[:, 0]
        elif pooling == "mean":
            embeddings = hidden.mean(dim=1)
        elif pooling == "max":
            embeddings = hidden.max(dim=1).values
        else:
            embeddings = hidden.mean(dim=1)

        return embeddings


class AlphaFoldWrapper(nn.Module):
    """Wrapper for AlphaFold structure predictions."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.embedding_dim = ALPHAFOLD_CONFIG["embedding_dim"]

    def predict_structure(self, sequence: str) -> dict[str, Any]:
        """Predict structure for a sequence.

        Args:
            sequence: Protein sequence

        Returns:
            Dictionary with structure information
        """
        # AlphaFold prediction would go here
        # For now, return placeholder structure
        return {
            "sequence": sequence,
            "plddt": np.random.rand(len(sequence)),
            "pae": np.random.rand(len(sequence), len(sequence)),
            "structure": None,  # Would contain 3D coordinates
        }

    def forward(self, sequences: list[str]) -> torch.Tensor:
        """Extract structure-based embeddings.

        Args:
            sequences: List of protein sequences

        Returns:
            Structure embeddings
        """
        # Placeholder implementation
        embeddings = torch.randn(len(sequences), self.embedding_dim, device=self.device)
        return embeddings


# =============================================================================
# Feature Extraction
# =============================================================================


class ProteinFeatureExtractor:
    """Extract features from pretrained protein models."""

    def __init__(
        self,
        model: nn.Module,
        model_type: PretrainedModelType = PretrainedModelType.ESM2_150M,
        device: str | None = None,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Get embedding dimension from config or model
        if model_type in ESM_MODEL_CONFIGS:
            self.embedding_dim = ESM_MODEL_CONFIGS[model_type]["embedding_dim"]
        elif model_type == PretrainedModelType.PROTBERT:
            self.embedding_dim = PROTBERT_CONFIG["embedding_dim"]
        elif model_type == PretrainedModelType.ALPHAFOLD:
            self.embedding_dim = ALPHAFOLD_CONFIG["embedding_dim"]
        else:
            self.embedding_dim = 1280

    @torch.no_grad()
    def extract_sequence_features(
        self,
        sequence: str,
        pooling: str = "mean",
        layer_index: int | None = None,
    ) -> ProteinEmbedding:
        """Extract embedding features from a protein sequence.

        Args:
            sequence: Protein sequence (Amino acid letters)
            pooling: Pooling strategy for sequence representation
            layer_index: Specific transformer layer to extract from

        Returns:
            ProteinEmbedding object with features
        """
        # Convert sequence to tokens
        tokens = self._sequence_to_tokens(sequence)
        tokens = torch.tensor(tokens, device=self.device).unsqueeze(0)

        # Forward through model
        if hasattr(self.model, "model") and hasattr(self.model, "tokenizer"):
            # ProtBERT wrapper
            embeddings = self.model([sequence], pooling=pooling)
            embeddings = embeddings.cpu().numpy()[0]
        elif isinstance(self.model, AlphaFoldWrapper):
            embeddings = self.model([sequence]).cpu().numpy()[0]
        else:
            # ESM model
            embeddings, layer_outputs = self.model(tokens, repr_layer=layer_index)

            if layer_index is not None and layer_index < len(layer_outputs):
                embeddings = layer_outputs[layer_index]
            else:
                embeddings = layer_outputs[-1] if layer_outputs else embeddings

            # Apply pooling
            embeddings = embeddings.squeeze(0).cpu().numpy()
            if pooling == "mean":
                embeddings = embeddings.mean(axis=0)
            elif pooling == "cls":
                embeddings = embeddings[0]
            elif pooling == "max":
                embeddings = embeddings.max(axis=0)

        return ProteinEmbedding(
            sequence=sequence,
            embedding=embeddings,
            model_type=self.model_type,
            layer_index=layer_index,
            representation_type=pooling,
            metadata={"pooling": pooling},
        )

    @torch.no_grad()
    def extract_batch_features(
        self,
        sequences: list[str],
        pooling: str = "mean",
        batch_size: int = 32,
    ) -> list[ProteinEmbedding]:
        """Extract embeddings for multiple sequences.

        Args:
            sequences: List of protein sequences
            pooling: Pooling strategy
            batch_size: Batch size for processing

        Returns:
            List of ProteinEmbedding objects
        """
        embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            batch_embeddings = []

            for seq in batch:
                emb = self.extract_sequence_features(seq, pooling=pooling)
                batch_embeddings.append(emb)

            embeddings.extend(batch_embeddings)

        return embeddings

    @torch.no_grad()
    def extract_per_residue_features(
        self,
        sequence: str,
    ) -> ProteinEmbedding:
        """Extract per-residue embeddings without pooling.

        Args:
            sequence: Protein sequence

        Returns:
            ProteinEmbedding with per-residue features [len, dim]
        """
        tokens = self._sequence_to_tokens(sequence)
        tokens = torch.tensor(tokens, device=self.device).unsqueeze(0)

        if hasattr(self.model, "model"):
            # ProtBERT
            inputs = self.model.tokenizer(
                sequence,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            outputs = self.model.model(**inputs)
            embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        else:
            # ESM
            embeddings, _ = self.model(tokens)
            embeddings = embeddings.squeeze(0).cpu().numpy()

        return ProteinEmbedding(
            sequence=sequence,
            embedding=embeddings,
            model_type=self.model_type,
            representation_type="per_residue",
        )

    def _sequence_to_tokens(self, sequence: str) -> list[int]:
        """Convert amino acid sequence to token IDs.

        Uses standard ESM alphabet mapping.
        """
        # Standard amino acid to index mapping (ESM style)
        aa_to_idx = {
            "<pad>": 0,
            "<mask>": 1,
            "<cls>": 2,
            "<sep>": 3,
            "<unk>": 4,
            "A": 5,
            "B": 6,  # Asx (N or D)
            "C": 7,
            "D": 8,
            "E": 9,
            "F": 10,
            "G": 11,
            "H": 12,
            "I": 13,
            "J": 14,  # Xle (I or L)
            "K": 15,
            "L": 16,
            "M": 17,
            "N": 18,
            "O": 19,  # Pyl
            "P": 20,
            "Q": 21,
            "R": 22,
            "S": 23,
            "T": 24,
            "U": 25,  # Sec
            "V": 26,
            "W": 27,
            "X": 28,  # Unspecified
            "Y": 29,
            "Z": 30,  # Glx (E or Q)
        }

        tokens = [aa_to_idx.get(aa.upper(), aa_to_idx["<unk>"]) for aa in sequence]
        return [aa_to_idx["<cls>"]] + tokens + [aa_to_idx["<sep>"]]


# =============================================================================
# Fine-tuning Interface
# =============================================================================


class TransferLearningModel(nn.Module):
    """Base class for transfer learning models."""

    def __init__(
        self,
        pretrained_model: nn.Module,
        num_tasks: int = 1,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pretrained_model = pretrained_model

        # Get embedding dimension
        if hasattr(pretrained_model, "embedding_dim"):
            emb_dim = pretrained_model.embedding_dim
        else:
            emb_dim = 1280

        self.hidden_dim = hidden_dim or emb_dim

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)

    def add_task_head(self, task_type: str, output_dim: int = 1) -> None:
        """Add a task-specific head.

        Args:
            task_type: Type identifier for the task
            output_dim: Output dimension (1 for regression, num_classes for classification)
        """
        self.task_heads[task_type] = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(self.hidden_dim // 2, output_dim),
        )

    def freeze_encoder_layers(self, num_layers: int | None = None) -> None:
        """Freeze pretrained encoder layers.

        Args:
            num_layers: Number of layers to freeze (None = all)
        """
        if hasattr(self.pretrained_model, "layers"):
            layers = self.pretrained_model.layers
            freeze_count = num_layers if num_layers is not None else len(layers)
            for i, layer in enumerate(layers):
                for param in layer.parameters():
                    param.requires_grad = i >= freeze_count
        elif hasattr(self.pretrained_model, "model"):
            # ProtBERT
            for param in self.pretrained_model.model.parameters():
                param.requires_grad = False

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


class ThermostabilityFineTuner(TransferLearningModel):
    """Fine-tuning model specialized for thermostability prediction."""

    def __init__(
        self,
        pretrained_model: nn.Module,
        model_type: PretrainedModelType = PretrainedModelType.ESM2_150M,
        pooling: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__(pretrained_model, dropout=dropout)
        self.model_type = model_type
        self.pooling = pooling

        # Get embedding dimension
        if model_type in ESM_MODEL_CONFIGS:
            self.hidden_dim = ESM_MODEL_CONFIGS[model_type]["embedding_dim"]
        elif model_type == PretrainedModelType.PROTBERT:
            self.hidden_dim = PROTBERT_CONFIG["embedding_dim"]
        else:
            self.hidden_dim = 1280

        # Default thermostability head
        self.add_task_head("thermostability", output_dim=1)

        # Auxiliary heads for multi-task learning
        self.add_task_head("stability_class", output_dim=3)  # thermophile, mesophile, psychrophile

    def forward(
        self,
        sequences: list[str],
        task: str = "thermostability",
    ) -> dict[str, torch.Tensor]:
        """Forward pass for thermostability prediction.

        Args:
            sequences: List of protein sequences
            task: Target task

        Returns:
            Dictionary with predictions for each task
        """
        if hasattr(self.pretrained_model, "model") and hasattr(self.pretrained_model, "tokenizer"):
            # ProtBERT
            embeddings = self.pretrained_model(sequences, pooling=self.pooling)
        elif isinstance(self.pretrained_model, AlphaFoldWrapper):
            embeddings = self.pretrained_model(sequences)
        else:
            # ESM - need to tokenize
            batch_tokens = []
            for seq in sequences:
                tokens = self._sequence_to_tokens(seq)
                batch_tokens.append(tokens)

            # Pad sequences
            max_len = max(len(t) for t in batch_tokens)
            padded = torch.zeros(len(batch_tokens), max_len, dtype=torch.long, device=next(self.parameters()).device)
            for i, t in enumerate(batch_tokens):
                padded[i, :len(t)] = torch.tensor(t)

            embeddings, _ = self.pretrained_model(padded)
            embeddings = embeddings.mean(dim=1)

        embeddings = self.dropout(embeddings)

        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(embeddings)

        return outputs

    def _sequence_to_tokens(self, sequence: str) -> list[int]:
        """Convert sequence to token IDs."""
        aa_to_idx = {
            "<cls>": 2, "A": 5, "C": 7, "D": 8, "E": 9, "F": 10, "G": 11,
            "H": 12, "I": 13, "K": 15, "L": 16, "M": 17, "N": 18, "P": 20,
            "Q": 21, "R": 22, "S": 23, "T": 24, "V": 26, "W": 27, "Y": 29,
            "<sep>": 3,
        }
        tokens = [aa_to_idx.get(aa.upper(), 4) for aa in sequence]
        return [2] + tokens + [3]


class MutationEffectFineTuner(TransferLearningModel):
    """Fine-tuning model for mutation effect prediction."""

    def __init__(
        self,
        pretrained_model: nn.Module,
        model_type: PretrainedModelType = PretrainedModelType.ESM2_150M,
        dropout: float = 0.1,
    ):
        super().__init__(pretrained_model, dropout=dropout)
        self.model_type = model_type

        # Get embedding dimension
        if model_type in ESM_MODEL_CONFIGS:
            self.hidden_dim = ESM_MODEL_CONFIGS[model_type]["embedding_dim"]
        else:
            self.hidden_dim = 1280

        # Delta-delta G prediction head
        self.add_task_head("ddg", output_dim=1)

        # Stability classification head
        self.add_task_head("stability_change", output_dim=2)  # stabilizing/destabilizing

    def forward(
        self,
        wt_sequence: str,
        mut_sequence: str,
        mutation_pos: int,
        task: str = "ddg",
    ) -> dict[str, torch.Tensor]:
        """Forward pass for mutation effect prediction.

        Args:
            wt_sequence: Wild-type sequence
            mut_sequence: Mutant sequence
            mutation_pos: Position of mutation (0-indexed)
            task: Target task

        Returns:
            Predictions for each task
        """
        # Extract features for both sequences
        extractor = ProteinFeatureExtractor(self.pretrained_model, self.model_type)

        wt_emb = extractor.extract_sequence_features(wt_sequence, pooling="mean")
        mut_emb = extractor.extract_sequence_features(mut_sequence, pooling="mean")

        # Concatenate wild-type, mutant, and position encoding
        pos_encoding = torch.tensor(
            [mutation_pos / 1000.0],  # Normalized position
            device=next(self.parameters()).device,
        )

        combined = torch.cat([
            torch.tensor(wt_emb.embedding, device=next(self.parameters()).device),
            torch.tensor(mut_emb.embedding, device=next(self.parameters()).device),
            pos_encoding.unsqueeze(0).expand(1, self.hidden_dim),
        ], dim=-1)

        combined = self.dropout(combined)

        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(combined)

        return outputs


# =============================================================================
# Training Interface
# =============================================================================


class ProteinDataset(Dataset):
    """Dataset for protein sequences with thermostability labels."""

    def __init__(
        self,
        sequences: list[str],
        labels: list[ThermostabilityLabel],
        model_type: PretrainedModelType = PretrainedModelType.ESM2_150M,
        augment: bool = False,
    ):
        self.sequences = sequences
        self.labels = labels
        self.model_type = model_type
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[str, ThermostabilityLabel]:
        sequence = self.sequences[idx]

        if self.augment:
            sequence = self._augment_sequence(sequence)

        return sequence, self.labels[idx]

    def _augment_sequence(self, sequence: str) -> str:
        """Apply sequence augmentation (mutation simulation)."""
        import random

        # Random sequence masking (like MLM)
        if random.random() < 0.15:
            seq_list = list(sequence)
            mask_idx = random.randint(0, len(seq_list) - 1)
            seq_list[mask_idx] = "<mask>"
            sequence = "".join(seq_list)

        return sequence


class TransferLearningTrainer:
    """Trainer for transfer learning models."""

    def __init__(
        self,
        model: TransferLearningModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: TransferLearningConfig | None = None,
        output_dir: str | Path = "./outputs/transfer_learning",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TransferLearningConfig(
            model_type=PretrainedModelType.ESM2_150M
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Setup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.total_epochs - self.config.warmup_epochs,
            T_mult=1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs],
        )

        # Training history
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.val_metrics_history: dict[str, list[float]] = {}

    def train(
        self,
        epochs: int | None = None,
        task: str = "thermostability",
        checkpoint_dir: str | Path | None = None,
    ) -> FineTuningResult:
        """Train the model.

        Args:
            epochs: Number of epochs (overrides config)
            task: Task to train on
            checkpoint_dir: Directory for checkpoints

        Returns:
            FineTuningResult with training history
        """
        epochs = epochs or self.config.total_epochs
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(epoch, task)
            self.train_loss_history.append(train_loss)

            # Validation
            val_metrics = self._validate_epoch(epoch, task)
            val_loss = val_metrics.get("loss", float("inf"))
            self.val_loss_history.append(val_loss)

            # Update scheduler
            self.scheduler.step()

            # Track metrics
            for metric_name, value in val_metrics.items():
                if metric_name not in self.val_metrics_history:
                    self.val_metrics_history[metric_name] = []
                self.val_metrics_history[metric_name].append(value)

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self._save_checkpoint(checkpoint_dir / "best_model.pt", epoch)

            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Best: {best_val_loss:.4f} (epoch {best_epoch + 1})"
            )

        return FineTuningResult(
            train_loss=self.train_loss_history,
            val_loss=self.val_loss_history,
            val_metrics=self.val_metrics_history,
            best_epoch=best_epoch,
            best_model_path=str(checkpoint_dir / "best_model.pt"),
            frozen_layers=self.config.frozen_layers,
            total_params=self.model.get_total_params(),
            trainable_params=self.model.get_trainable_params(),
        )

    def _train_epoch(self, epoch: int, task: str) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            sequences, labels = batch
            sequences = [s for s, _ in zip(sequences, labels)]
            labels = [l for l in labels]

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(sequences, task=task)

            # Compute loss
            targets = torch.tensor(
                [self._label_to_target(lb, task) for lb in labels],
                device=self.device,
            ).float()

            predictions = outputs[task].squeeze(-1)
            loss = F.mse_loss(predictions, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch(self, epoch: int, task: str) -> dict[str, float]:
        """Validate for one epoch."""
        if self.val_loader is None:
            return {"loss": 0.0}

        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                sequences, labels = batch
                sequences = [s for s, _ in zip(sequences, labels)]
                labels = [l for l in labels]

                outputs = self.model(sequences, task=task)
                targets = torch.tensor(
                    [self._label_to_target(lb, task) for lb in labels],
                    device=self.device,
                ).float()

                predictions = outputs[task].squeeze(-1)
                loss = F.mse_loss(predictions, targets)
                mae = F.l1_loss(predictions, targets)

                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1

        return {
            "loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "mae": total_mae / num_batches if num_batches > 0 else 0.0,
        }

    def _label_to_target(self, label: ThermostabilityLabel, task: str) -> float:
        """Convert label to target value."""
        if task == "thermostability":
            return label.delta_tm if label.delta_tm is not None else (label.tm_value or 0.0)
        elif task == "stability_class":
            class_map = {"thermostable": 2.0, "mesostable": 1.0, "psychrophile": 0.0}
            return class_map.get(label.stability_class or "", 1.0)
        return 0.0

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


# =============================================================================
# Feature Transfer Utilities
# =============================================================================


class FeatureTransferrer:
    """Transfer features between different protein representations."""

    def __init__(
        self,
        source_model: PretrainedModelLoader,
        target_model: PretrainedModelLoader,
    ):
        self.source_model = source_model
        self.target_model = target_model

    def compute_sequence_similarity(
        self,
        sequence1: str,
        sequence2: str,
    ) -> float:
        """Compute similarity between two sequences using their embeddings.

        Args:
            sequence1: First protein sequence
            sequence2: Second protein sequence

        Returns:
            Cosine similarity between embeddings
        """
        emb1 = self.source_model.extract_sequence_features(sequence1)
        emb2 = self.source_model.extract_sequence_features(sequence2)

        similarity = np.dot(emb1.embedding, emb2.embedding) / (
            np.linalg.norm(emb1.embedding) * np.linalg.norm(emb2.embedding)
        )

        return float(similarity)

    def transfer_knowledge(
        self,
        source_sequences: list[str],
        target_sequences: list[str],
    ) -> dict[str, Any]:
        """Transfer knowledge by computing cross-model similarities.

        Args:
            source_sequences: Sequences from source model domain
            target_sequences: Sequences to transfer knowledge to

        Returns:
            Dictionary with transfer metrics
        """
        source_embeddings = [
            self.source_model.extract_sequence_features(seq)
            for seq in source_sequences
        ]

        target_embeddings = [
            self.target_model.extract_sequence_features(seq)
            for seq in target_sequences
        ]

        # Compute pairwise similarities
        similarity_matrix = np.zeros((len(source_embeddings), len(target_embeddings)))
        for i, src_emb in enumerate(source_embeddings):
            for j, tgt_emb in enumerate(target_embeddings):
                similarity_matrix[i, j] = np.dot(src_emb.embedding, tgt_emb.embedding) / (
                    np.linalg.norm(src_emb.embedding) * np.linalg.norm(tgt_emb.embedding)
                )

        return {
            "similarity_matrix": similarity_matrix,
            "mean_similarity": float(similarity_matrix.mean()),
            "max_similarity": float(similarity_matrix.max()),
            "source_diversity": float(np.std([e.embedding for e in source_embeddings])),
            "target_diversity": float(np.std([e.embedding for e in target_embeddings])),
        }


# =============================================================================
# Thermostability Task Adapter
# =============================================================================


class ThermostabilityAdapter:
    """Adapter for thermostability prediction tasks."""

    def __init__(
        self,
        fine_tuned_model: ThermostabilityFineTuner,
        feature_extractor: ProteinFeatureExtractor,
    ):
        self.model = fine_tuned_model
        self.extractor = feature_extractor

    def predict_stability(
        self,
        sequence: str,
        mutation: str | None = None,
    ) -> dict[str, float]:
        """Predict thermostability for a sequence.

        Args:
            sequence: Protein sequence
            mutation: Optional mutation string (e.g., "A123G")

        Returns:
            Dictionary with stability predictions
        """
        if mutation:
            mut_sequence = self._apply_mutation(sequence, mutation)
            pred = self.model([mut_sequence], task="thermostability")["thermostability"]
        else:
            pred = self.model([sequence], task="thermostability")["thermostability"]

        delta_tm = pred.squeeze().item()

        # Also get stability class prediction
        class_pred = self.model([sequence], task="stability_class")["stability_class"]
        class_probs = F.softmax(class_pred, dim=-1).squeeze()
        class_idx = class_probs.argmax().item()
        classes = ["psychrophile", "mesostable", "thermostable"]

        return {
            "delta_tm": delta_tm,
            "predicted_tm_change": delta_tm,
            "stability_class": classes[class_idx],
            "class_probabilities": {
                c: float(p) for c, p in zip(classes, class_probs.tolist())
            },
        }

    def batch_predict(
        self,
        sequences: list[str],
    ) -> list[dict[str, float]]:
        """Predict thermostability for multiple sequences.

        Args:
            sequences: List of protein sequences

        Returns:
            List of prediction dictionaries
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for seq in sequences:
                pred = self.predict_stability(seq)
                predictions.append(pred)

        return predictions

    def _apply_mutation(self, sequence: str, mutation: str) -> str:
        """Apply a mutation to a sequence.

        Args:
            sequence: Original sequence
            mutation: Mutation string (e.g., "A123G")

        Returns:
            Mutated sequence
        """
        import re

        # Parse mutation string
        match = re.match(r"([A-Z])(\d+)([A-Z])", mutation)
        if not match:
            return sequence

        wt_aa, pos, mut_aa = match.groups()
        pos = int(pos) - 1  # Convert to 0-indexed

        if pos >= len(sequence) or sequence[pos] != wt_aa:
            logger.warning(f"Mutation {mutation} doesn't match sequence")
            return sequence

        seq_list = list(sequence)
        seq_list[pos] = mut_aa
        return "".join(seq_list)

    def get_feature_importance(
        self,
        sequence: str,
        position: int,
    ) -> dict[str, float]:
        """Get feature importance for a specific position.

        Args:
            sequence: Protein sequence
            position: Position to analyze (0-indexed)

        Returns:
            Dictionary with importance scores per amino acid
        """
        # Extract per-residue features
        embedding = self.extractor.extract_per_residue_features(sequence)

        if position >= len(embedding.embedding):
            raise ValueError(f"Position {position} out of range for sequence length {len(sequence)}")

        position_embedding = embedding.embedding[position]

        # Compute importance as cosine similarity to global mean
        global_mean = embedding.embedding.mean(axis=0)
        importance = np.dot(position_embedding, global_mean) / (
            np.linalg.norm(position_embedding) * np.linalg.norm(global_mean)
        )

        # Return importance per amino acid type
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        importance_scores = {}

        for aa in amino_acids:
            # Create one-hot encoding
            aa_idx = {"A": 5, "C": 7, "D": 8, "E": 9, "F": 10, "G": 11,
                     "H": 12, "I": 13, "K": 15, "L": 16, "M": 17, "N": 18,
                     "P": 20, "Q": 21, "R": 22, "S": 23, "T": 24, "V": 26,
                     "W": 27, "Y": 29}.get(aa, 4)

            # Simplified importance: embedding norm contribution
            importance_scores[aa] = float(np.abs(position_embedding).mean())

        return importance_scores


# =============================================================================
# Convenience Functions
# =============================================================================


def create_fine_tuning_pipeline(
    model_type: PretrainedModelType = PretrainedModelType.ESM2_150M,
    cache_dir: str | Path = "./models",
    task: str = "thermostability",
) -> tuple[ThermostabilityFineTuner, TransferLearningTrainer]:
    """Create a complete fine-tuning pipeline.

    Args:
        model_type: Pretrained model to use
        cache_dir: Directory for cached models
        task: Task type for fine-tuning

    Returns:
        Tuple of (model, trainer)
    """
    # Load pretrained model
    loader = PretrainedModelLoader(cache_dir)
    pretrained = loader.load(model_type)

    # Create fine-tuner
    fine_tuner = ThermostabilityFineTuner(pretrained, model_type)

    # Freeze early layers (keep last few for adaptation)
    config = ESM_MODEL_CONFIGS.get(model_type, {})
    num_layers = config.get("num_layers", 30)
    fine_tuner.freeze_encoder_layers(num_layers=num_layers - 4)  # Keep last 4 layers trainable

    return fine_tuner


def extract_protein_features(
    sequence: str,
    model_type: PretrainedModelType = PretrainedModelType.ESM2_150M,
    pooling: str = "mean",
) -> np.ndarray:
    """Convenience function to extract features from a protein sequence.

    Args:
        sequence: Protein sequence
        model_type: Model to use
        pooling: Pooling strategy

    Returns:
        Feature vector
    """
    loader = PretrainedModelLoader()
    model = loader.load(model_type)
    extractor = ProteinFeatureExtractor(model, model_type)
    embedding = extractor.extract_sequence_features(sequence, pooling=pooling)
    return embedding.embedding
