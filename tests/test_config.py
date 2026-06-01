"""Tests for the config module."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hlstm_framework.config import load_settings, ModelConfig, TrainingConfig


def test_load_settings():
    """Verify settings load with defaults."""
    cfg = load_settings(reload=True)
    assert cfg.model.hidden_dim == 512
    assert cfg.training.batch_size == 32
    assert cfg.data.min_files_per_class == 25
    print(f"  [PASS] test_load_settings")


def test_device_auto():
    """Verify device auto-detection doesn't crash."""
    mc = ModelConfig(device="auto")
    assert mc.device in ("cpu", "cuda", "mps")
    print(f"  [PASS] test_device_auto (resolved to '{mc.device}')")


def test_loss_weights_parsing():
    """Verify loss_weights_list property."""
    tc = TrainingConfig(loss_weights="1.0,1.2,1.5,1.0")
    assert tc.loss_weights_list == [1.0, 1.2, 1.5, 1.0]
    print(f"  [PASS] test_loss_weights_parsing")


def test_env_override():
    """Verify environment variable overrides work."""
    os.environ["TRAIN_BATCH_SIZE"] = "64"
    cfg = load_settings(reload=True)
    assert cfg.training.batch_size == 64
    del os.environ["TRAIN_BATCH_SIZE"]
    print(f"  [PASS] test_env_override")


if __name__ == "__main__":
    print("Running config tests...")
    test_load_settings()
    test_device_auto()
    test_loss_weights_parsing()
    test_env_override()
    print("All config tests passed!")
