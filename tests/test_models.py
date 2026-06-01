"""Tests for the H-LSTM model and trainer."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from hlstm_framework.models import HEventModel, HLSTMTrainer, CrossAttention, AttentionPooling, TokenFusion
from hlstm_framework.models.factory import create_model, create_trainer


def test_model_forward_shape():
    """Verify forward pass returns correct logit shapes."""
    model = HEventModel(device="cpu")
    B, T = 4, 32
    x = torch.randint(0, 128, (B, T, 4))
    style = torch.randint(0, 2, (B,))
    instr = torch.randint(0, 128, (B, 4))
    ctrl = torch.randn(B, T, 128)

    pitch, dur, vel, instr_out, hidden = model(x, style=style, instr_context=instr, control_context=ctrl)

    assert pitch.shape == (B, T, 128), f"pitch: {pitch.shape}"
    assert dur.shape == (B, T, 128)
    assert vel.shape == (B, T, 128)
    assert instr_out.shape == (B, T, 128)
    assert len(hidden) == 3
    print(f"  [PASS] test_model_forward_shape")


def test_model_no_conditions():
    """Verify forward pass with no conditioning."""
    model = HEventModel(device="cpu")
    B, T = 2, 16
    x = torch.randint(0, 128, (B, T, 4))
    pitch, dur, vel, instr_out, hidden = model(x)
    assert pitch.shape == (B, T, 128)
    print(f"  [PASS] test_model_no_conditions")


def test_hidden_carry():
    """Verify hidden state carry across calls."""
    model = HEventModel(device="cpu")
    x1 = torch.randint(0, 128, (1, 16, 4))
    x2 = torch.randint(0, 128, (1, 16, 4))
    _, _, _, _, all_h1 = model(x1)  # all_h1 = (h1_1, h2_1, h3_1) each = (h_tensor, c_tensor)
    _, _, _, _, all_h2 = model(x2, hidden_states=all_h1)
    # Each layer state: (h_tensor, c_tensor) — check h_tensor shape per layer
    for h_new, h_old in zip(all_h2, all_h1):
        assert h_new[0].shape == h_old[0].shape == (1, 1, 512), \
            f"Expected (1,1,512) got {h_new[0].shape} vs {h_old[0].shape}"
    print(f"  [PASS] test_hidden_carry")


def test_position_encoding():
    """Verify position encoding forward pass."""
    model = HEventModel(device="cpu", use_position_encoding=True, num_positions=64)
    B, T = 2, 16
    x = torch.randint(0, 128, (B, T, 5))
    x[:, :, 4] = torch.randint(0, 64, (B, T))
    pitch, *_ = model(x)
    assert pitch.shape == (B, T, 128)
    print(f"  [PASS] test_position_encoding")


def test_cross_attention():
    """Verify cross-attention module shapes."""
    ca = CrossAttention(hidden_dim=512, num_heads=4)
    B, T, S = 2, 16, 32
    q = torch.randn(B, T, 512)
    kv = torch.randn(B, S, 512)
    out = ca(q, kv)
    assert out.shape == (B, T, 512)
    print(f"  [PASS] test_cross_attention")


def test_attention_pooling():
    """Verify attention pooling module."""
    ap = AttentionPooling(hidden_dim=512, num_heads=4)
    x = torch.randn(2, 8, 512)
    out = ap(x)
    assert out.shape == (2, 1, 512)
    print(f"  [PASS] test_attention_pooling")


def test_token_fusion():
    """Verify token fusion module shapes."""
    tf = TokenFusion(hidden_dim=512, fusion_dim=256)
    x = torch.randn(2, 16, 512)
    out = tf(x)
    assert out.shape == (2, 16, 512)
    print(f"  [PASS] test_token_fusion")


def test_trainer_step():
    """Verify HLSTMTrainer train_step."""
    model = HEventModel(device="cpu")
    trainer = HLSTMTrainer(model, lr=1e-3, device="cpu", use_amp=False)
    B, T = 4, 32
    x = torch.randint(0, 128, (B, T, 4))
    y = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
    loss = trainer.train_step(x, y)
    assert isinstance(loss, float) and 0 < loss < 50
    print(f"  [PASS] test_trainer_step (loss={loss:.4f})")


def test_create_model_factory():
    """Verify model factory creates model from config."""
    model = create_model(use_cross_attention=True, use_token_fusion=True)
    assert model.use_cross_attention
    assert model.use_token_fusion
    print(f"  [PASS] test_create_model_factory")


def test_save_load():
    """Verify model save/load."""
    import tempfile
    model = HEventModel(device="cpu")
    trainer = HLSTMTrainer(model, device="cpu")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    trainer.save(path)
    assert os.path.exists(path)

    model2 = HEventModel(device="cpu")
    trainer2 = HLSTMTrainer(model2, device="cpu")
    trainer2.load(path)
    os.remove(path)
    print(f"  [PASS] test_save_load")


if __name__ == "__main__":
    print("Running model tests...")
    test_model_forward_shape()
    test_model_no_conditions()
    test_hidden_carry()
    test_position_encoding()
    test_cross_attention()
    test_attention_pooling()
    test_token_fusion()
    test_trainer_step()
    test_create_model_factory()
    test_save_load()
    print("All model tests passed!")
