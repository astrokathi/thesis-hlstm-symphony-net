"""Unit tests for HEventModel shape correctness and conditioning paths."""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import HEventModel, HLSTMTrainer


def test_forward_shape():
    """Verify forward pass returns correct logit shapes."""
    model = HEventModel(device="cpu")
    B, T = 4, 32
    x = torch.randint(0, 128, (B, T, 4))
    style = torch.randint(0, 2, (B,))
    instr = torch.randint(0, 128, (B, 4))
    ctrl = torch.randn(B, T, 128)

    pitch, dur, vel, instr_out, hidden = model(x, style=style, instr_context=instr, control_context=ctrl)

    assert pitch.shape == (B, T, 128), f"pitch shape: {pitch.shape}"
    assert dur.shape == (B, T, 128), f"dur shape: {dur.shape}"
    assert vel.shape == (B, T, 128), f"vel shape: {vel.shape}"
    assert instr_out.shape == (B, T, 128), f"instr shape: {instr_out.shape}"
    assert len(hidden) == 3, f"hidden len: {len(hidden)}"
    print(f"  [PASS] test_forward_shape")


def test_forward_no_conditions():
    """Verify forward pass works with no conditioning inputs."""
    model = HEventModel(device="cpu")
    B, T = 2, 16
    x = torch.randint(0, 128, (B, T, 4))

    pitch, dur, vel, instr_out, hidden = model(x)

    assert pitch.shape == (B, T, 128)
    assert dur.shape == (B, T, 128)
    print(f"  [PASS] test_forward_no_conditions")


def test_hidden_state_carry():
    """Verify hidden state carry works across calls."""
    model = HEventModel(device="cpu")
    B, T = 1, 16
    x1 = torch.randint(0, 128, (B, T, 4))
    x2 = torch.randint(0, 128, (B, T, 4))

    _, _, _, _, h1 = model(x1)
    _, _, _, _, h2 = model(x2, hidden_states=h1)

    assert all(h[0].shape == (1, B, 512) for h in [h1, h2]), "hidden state shape mismatch"
    print(f"  [PASS] test_hidden_state_carry")


def test_trainer_step():
    """Verify HLSTMTrainer train_step runs without error."""
    model = HEventModel(device="cpu")
    trainer = HLSTMTrainer(model, lr=1e-3, device="cpu")
    B, T = 4, 32
    x = torch.randint(0, 128, (B, T, 4))
    # y is x shifted by 1
    y = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
    style = torch.randint(0, 2, (B,))
    instr = torch.randint(0, 128, (B, 4))
    ctrl = torch.randn(B, T, 128)

    loss = trainer.train_step(x, y, style=style, instr_context=instr, control_context=ctrl)
    assert isinstance(loss, float), f"loss should be float, got {type(loss)}"
    assert 0 < loss < 50, f"loss out of range: {loss}"
    print(f"  [PASS] test_trainer_step (loss={loss:.4f})")


def test_eval_step():
    """Verify HLSTMTrainer eval_step runs without error."""
    model = HEventModel(device="cpu")
    trainer = HLSTMTrainer(model, lr=1e-3, device="cpu")
    B, T = 4, 32
    x = torch.randint(0, 128, (B, T, 4))
    y = torch.cat([x[:, 1:], x[:, -1:]], dim=1)

    loss = trainer.eval_step(x, y)
    assert isinstance(loss, float)
    print(f"  [PASS] test_eval_step (loss={loss:.4f})")


def test_save_load(tmp_path="/tmp/test_hlstm"):
    """Verify model save/load works."""
    import os
    os.makedirs(tmp_path, exist_ok=True)
    model = HEventModel(device="cpu")
    trainer = HLSTMTrainer(model, device="cpu")
    path = os.path.join(tmp_path, "test_model.pt")
    trainer.save(path)
    assert os.path.exists(path), "save file not created"

    # Load into fresh model
    model2 = HEventModel(device="cpu")
    trainer2 = HLSTMTrainer(model2, device="cpu")
    trainer2.load(path)
    os.remove(path)
    print(f"  [PASS] test_save_load")


def test_position_encoding():
    """Verify position encoding doesn't break forward pass."""
    model = HEventModel(device="cpu", use_position_encoding=True, num_positions=64)
    B, T = 2, 16
    x = torch.randint(0, 128, (B, T, 5))  # 5th channel = beat position
    x[:, :, 4] = torch.randint(0, 64, (B, T))

    pitch, *_ = model(x)
    assert pitch.shape == (B, T, 128)
    print(f"  [PASS] test_position_encoding")


if __name__ == "__main__":
    print("Running model tests...")
    test_forward_shape()
    test_forward_no_conditions()
    test_hidden_state_carry()
    test_trainer_step()
    test_eval_step()
    test_save_load()
    test_position_encoding()
    print("All model tests passed!")
