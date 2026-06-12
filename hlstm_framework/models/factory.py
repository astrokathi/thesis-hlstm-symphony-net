"""
Model factory — create HEventModel and HLSTMTrainer from settings.

Reduces boilerplate by constructing model + trainer from the global
FrameworkSettings, with optional overrides.

Usage:
    >>> from hlstm_framework.models.factory import create_model, create_trainer
    >>> model = create_model(use_cross_attention=True)
    >>> trainer = create_trainer(model, use_amp=True)
"""

from typing import Optional

from hlstm_framework.config import load_settings, ModelConfig
from hlstm_framework.models.hlstm import HEventModel, HLSTMTrainer


def create_model(
    config: Optional[ModelConfig] = None,
    **overrides,
) -> HEventModel:
    """Create an HEventModel from configuration with optional overrides.

    Args:
        config: A ModelConfig instance. Defaults to global settings.
        **overrides: Any HEventModel constructor parameter to override.

    Returns:
        Configured HEventModel.

    Example:
        >>> model = create_model(use_cross_attention=True, use_token_fusion=True)
    """
    cfg = (config or load_settings().model)
    kwargs = {
        "num_pitches": cfg.num_pitches,
        "num_durations": cfg.num_durations,
        "num_velocities": cfg.num_velocities,
        "num_instruments": cfg.num_instruments,
        "style_classes": cfg.style_classes,
        "embed_dim": cfg.embed_dim,
        "control_dim": cfg.control_dim,
        "hidden_dim": cfg.hidden_dim,
        "dropout": cfg.dropout,
        "device": cfg.device,
        "use_position_encoding": cfg.use_position_encoding,
        "num_positions": cfg.num_positions,
        "use_checkpointing": cfg.use_checkpointing,
        "use_cross_attention": cfg.use_cross_attention,
        "use_instr_attention_pooling": cfg.use_instr_attention_pooling,
        "use_token_fusion": cfg.use_token_fusion,
    }
    kwargs.update(overrides)
    return HEventModel(**kwargs)


def create_trainer(
    model: HEventModel,
    config: Optional[ModelConfig] = None,
    **overrides,
) -> HLSTMTrainer:
    """Create an HLSTMTrainer from configuration with optional overrides.

    Uses TrainingConfig for training-related settings and ModelConfig for device.

    Args:
        model: An initialized HEventModel.
        config: A ModelConfig (for device). Defaults to global settings.
        **overrides: Any HLSTMTrainer constructor parameter to override.

    Returns:
        Configured HLSTMTrainer.
    """
    train_cfg = load_settings().training
    kwargs = {
        "model": model,
        "lr": train_cfg.lr,
        "device": config.device if config else load_settings().model.device,
        "use_amp": train_cfg.use_amp,
        "loss_weights": train_cfg.loss_weights_list,
        "scheduler_type": train_cfg.lr_scheduler,
        "scheduler_params": {
            "patience": train_cfg.lr_patience,
            "min_lr": train_cfg.lr_min,
            "t_max": train_cfg.num_epochs,
        },
        "grad_clip_norm": train_cfg.grad_clip_norm,
        "weight_decay": 1e-4,
    }
    kwargs.update(overrides)
    return HLSTMTrainer(**kwargs)
