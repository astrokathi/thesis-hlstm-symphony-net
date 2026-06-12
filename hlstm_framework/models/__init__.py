"""
Models module — H-LSTM architecture, trainer, factory, and distributed training.

Architecture (3-layer Hierarchical LSTM):
    Layer 1 — Style conditioning (global genre/rhythm)
    Layer 2 — Instrumentation conditioning (timbre/ensemble)
    Layer 3 — Control context (expression/micro-timing)

All architectural upgrades are config-gated and opt-in:
    - Position encoding (Sprint 3)
    - Cross-attention between layers (Sprint 4)
    - Attention pooling for instrument context (Sprint 4)
    - Joint token fusion MLP (Sprint 4)
    - Gradient checkpointing (Sprint 3)
    - AMP mixed precision (Sprint 1)

Usage:
    >>> from hlstm_framework.models import HEventModel, HLSTMTrainer
    >>> model = HEventModel()
    >>> trainer = HLSTMTrainer(model)
    >>> trainer.train(train_loader, val_loader, num_epochs=30)
"""

from hlstm_framework.models.hlstm import HEventModel, HLSTMTrainer
from hlstm_framework.models.layers import CrossAttention, AttentionPooling, TokenFusion
from hlstm_framework.models.factory import create_model, create_trainer
from hlstm_framework.models.trainer import train_model

__all__ = [
    "HEventModel",
    "HLSTMTrainer",
    "CrossAttention",
    "AttentionPooling",
    "TokenFusion",
    "create_model",
    "create_trainer",
    "train_model",
]
