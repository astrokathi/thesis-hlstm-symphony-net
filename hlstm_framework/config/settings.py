"""
Pydantic-based configuration for the entire H-LSTM framework.

Loads from .env file and/or environment variables with sensible defaults.
All values can be overridden at runtime for experiment tracking.

Usage:
    >>> from hlstm_framework.config import load_settings
    >>> cfg = load_settings()
    >>> cfg.model.hidden_dim
    512
    >>> cfg.training.batch_size
    32
"""

import os
from pathlib import Path
from typing import List, Optional, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to the PROJECT root (parent of hlstm_framework/)
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
_PROJECT_ENV_FILE = str(_PACKAGE_ROOT / ".env")


class ModelConfig(BaseSettings):
    """Configuration for the H-LSTM architecture.

    Controls all model dimensions, conditioning flags, and optional
    architectural upgrades (position encoding, cross-attention, etc.).
    """

    model_config = SettingsConfigDict(env_prefix="MODEL_", extra="ignore")

    # --- Core dimensions ---
    num_pitches: int = Field(128, description="MIDI pitch vocabulary size (0-127)")
    num_durations: int = Field(128, description="Duration bin vocabulary size")
    num_velocities: int = Field(128, description="Velocity vocabulary size (0-127)")
    num_instruments: int = Field(128, description="MIDI program vocabulary size (0-127)")
    style_classes: int = Field(2, description="Number of style labels (classical/contemporary)")
    embed_dim: int = Field(256, description="Embedding dimension for each feature")
    control_dim: int = Field(128, description="Control context vector dimension")
    hidden_dim: int = Field(512, description="LSTM hidden state dimension")
    dropout: float = Field(0.3, ge=0.0, le=1.0, description="Dropout rate after LSTM3")
    device: str = Field(
        "auto",
        description="Torch device: 'auto', 'mps', 'cuda', or 'cpu'"
    )

    # --- Sprint 3: Position encoding ---
    use_position_encoding: bool = Field(
        False, description="Enable bar/beat position encoding (5th input channel)"
    )
    num_positions: int = Field(64, description="Position vocabulary size")

    # --- Sprint 3: Memory optimizations ---
    use_checkpointing: bool = Field(
        False, description="Enable gradient checkpointing to reduce memory"
    )
    use_torch_compile: bool = Field(
        False, description="Enable torch.compile graph optimization"
    )

    # --- Sprint 4: Deep architecture ---
    use_cross_attention: bool = Field(
        False, description="Enable cross-attention between LSTM1->LSTM2"
    )
    use_instr_attention_pooling: bool = Field(
        False, description="Replace mean pooling with attention pooling for instr context"
    )
    use_token_fusion: bool = Field(
        False, description="Enable joint token fusion MLP after LSTM3"
    )

    @field_validator("device", mode="before")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        """Resolve 'auto' to the best available device."""
        if v.lower() == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return v.lower()


class TrainingConfig(BaseSettings):
    """Configuration for model training loops.

    Covers sequence dimensions, optimization hyperparameters, learning rate
    scheduling, and the Sprint 1 performance upgrades (AMP, multi-worker, etc.).
    """

    model_config = SettingsConfigDict(env_prefix="TRAIN_", extra="ignore")

    seq_len: int = Field(128, description="Sliding window sequence length")
    batch_size: int = Field(32, description="Training batch size")
    num_epochs: int = Field(30, description="Maximum number of training epochs")
    lr: float = Field(0.001, description="Initial learning rate")
    val_split: float = Field(0.2, ge=0.0, lt=1.0, description="Fraction of data for validation")
    patience: int = Field(7, description="Early stopping patience (epochs)")
    token_path: str = Field("data/encoded/encoded_tokens.pkl", description="Path to pickled encoded dataset")
    output_dir: str = Field("models", description="Directory for model checkpoints")
    log_dir: str = Field("runs/hlstm", description="TensorBoard log directory")

    # --- Sprint 1: Performance ---
    use_amp: bool = Field(True, description="Enable mixed precision training")
    num_workers: int = Field(2, description="DataLoader worker processes")
    pin_memory: bool = Field(True, description="Pin memory for faster GPU transfer")
    loss_weights: str = Field(
        "1.0,1.0,1.0,1.0",
        description="Comma-separated loss weights: pitch,duration,velocity,instrument"
    )
    lr_scheduler: Literal["plateau", "cosine", "none"] = Field(
        "plateau", description="Learning rate scheduler type"
    )
    lr_patience: int = Field(3, description="LR scheduler patience (ReduceLROnPlateau)")
    lr_min: float = Field(1e-6, description="Minimum learning rate")
    grad_clip_norm: float = Field(0.0, description="Max gradient norm (0=disabled)")

    @property
    def loss_weights_list(self) -> List[float]:
        """Parse loss_weights string into a float list."""
        return [float(w.strip()) for w in self.loss_weights.split(",")]


class DataConfig(BaseSettings):
    """Configuration for data ingestion and preprocessing.

    Controls input paths, style labeling, duration binning, and optional
    clustering for mixed (unlabeled) MIDI datasets.
    """

    model_config = SettingsConfigDict(env_prefix="DATA_", extra="ignore")

    dir: str = Field("/path/to/dataset", description="Root dataset directory")
    encoded_path: str = Field("data/encoded/encoded_tokens.pkl", description="Output path for pickled encodings")
    sub_dirs: str = Field("classical,contemporary", description="Comma-separated subdirectory names for style labels")
    min_files_per_class: int = Field(25, description="Files to sample per style class")
    enable_augmentation: bool = Field(False, description="Enable data augmentation (pitch, tempo, velocity)")
    use_fixed_bins: bool = Field(False, description="Use corpus-level fixed duration bins")
    cluster_styles: bool = Field(
        False,
        description="If True, use K-means on musical features to assign style labels "
                    "(required when MIDI files are not organized in style subdirectories)"
    )
    cluster_n_clusters: int = Field(2, description="Number of style clusters for K-means")

    @property
    def sub_dirs_list(self) -> List[str]:
        """Parse sub_dirs string into a list."""
        return [s.strip() for s in self.sub_dirs.split(",")]


class GenerationConfig(BaseSettings):
    """Configuration for music generation and inference."""

    model_config = SettingsConfigDict(env_prefix="GEN_", extra="ignore")

    prompt_file: str = Field("test/prompt.mid", description="Path to prompt MIDI file")
    preset: str = Field("gentle", description="Control preset name for Method 4")
    num_samples: int = Field(3, description="Number of samples to generate per method")
    style: int = Field(1, description="Style ID: 0=classical, 1=contemporary")
    instruments: str = Field("40,41,42,43", description="Comma-separated allowed instrument program numbers")
    num_steps: int = Field(200, description="Number of notes to generate")
    temperature: float = Field(0.7, ge=0.0, description="Sampling temperature")
    top_k: int = Field(40, description="Top-k sampling threshold (0=disabled)")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    notes_per_chord: int = Field(3, description="Instruments per chord (polyphonic methods)")
    output_dir: str = Field("output", description="Output directory for generated MIDI/WAV files")

    @property
    def instruments_list(self) -> List[int]:
        """Parse instruments string into a list of ints."""
        return [int(i.strip()) for i in self.instruments.split(",")]


class RayConfig(BaseSettings):
    """Configuration for Ray-based distributed training."""

    model_config = SettingsConfigDict(env_prefix="RAY_", extra="ignore")

    enabled: bool = Field(False, description="Enable Ray distributed training")
    num_workers: int = Field(1, description="Number of Ray training workers")
    use_gpu: bool = Field(False, description="Allocate GPU per Ray worker")
    storage_path: str = Field("/tmp/ray_results", description="Ray result storage path")
    checkpoint_freq: int = Field(1, description="Checkpoint frequency (epochs)")


class FrameworkSettings(BaseSettings):
    """Top-level settings aggregating all sub-configurations.

    Loads from .env file (in current or parent directory) and environment variables.
    All sub-configs are auto-discovered via their respective env prefixes.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    ray: RayConfig = Field(default_factory=RayConfig)


# Module-level singleton cache
_settings: Optional[FrameworkSettings] = None


def load_settings(reload: bool = False) -> FrameworkSettings:
    """Load (or reload) the framework settings singleton.

    Args:
        reload: If True, force re-read from .env and environment variables.

    Returns:
        A FrameworkSettings instance with all sub-configurations populated.

    Example:
        >>> cfg = load_settings()
        >>> cfg.model.hidden_dim
        512
        >>> cfg.training.batch_size
        32
    """
    global _settings
    if _settings is None or reload:
        _settings = FrameworkSettings()
    return _settings
