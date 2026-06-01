"""Tests for the generation and sampler modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from hlstm_framework.generation.sampler import (
    top_k_top_p_filter,
    sample_top_k_top_p,
    limit_leap,
)
from hlstm_framework.generation.presets import CONTROL_PRESETS, list_presets


def test_top_k_filter():
    """Verify top-k filtering preserves only k values."""
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    filtered = top_k_top_p_filter(logits, top_k=2)
    assert torch.isfinite(filtered).sum() == 2, f"Expected 2 finite, got {torch.isfinite(filtered).sum()}"
    print(f"  [PASS] test_top_k_filter")


def test_top_p_filter():
    """Verify top-p filtering."""
    logits = torch.tensor([1.0, 10.0, 1.0, 1.0, 1.0])
    filtered = top_k_top_p_filter(logits, top_p=0.5)
    assert torch.isfinite(filtered).sum() >= 1
    print(f"  [PASS] test_top_p_filter")


def test_sample_top_k_top_p():
    """Verify sampling returns a valid integer."""
    logits = torch.randn(128)
    idx = sample_top_k_top_p(logits, temperature=1.0, top_k=20, top_p=0.9)
    assert isinstance(idx, int)
    assert 0 <= idx < 128
    print(f"  [PASS] test_sample_top_k_top_p (idx={idx})")


def test_sample_temperature_0():
    """Verify argmax at temperature=0."""
    logits = torch.tensor([0.1, 0.2, 10.0, 0.3, 0.4])
    idx = sample_top_k_top_p(logits, temperature=0.0)
    assert idx == 2, f"Expected 2 (argmax), got {idx}"
    print(f"  [PASS] test_sample_temperature_0")


def test_limit_leap():
    """Verify leap limiting works."""
    logits = torch.randn(128)
    filtered = limit_leap(prev_pitch=60, pitch_logits=logits, max_leap=12)
    valid_range = set(range(60 - 12, 60 + 12 + 1))
    finite_indices = torch.where(torch.isfinite(filtered))[0].tolist()
    assert all(i in valid_range for i in finite_indices), "Leap limit violated"
    print(f"  [PASS] test_limit_leap")


def test_control_presets():
    """Verify all presets have correct structure."""
    assert len(CONTROL_PRESETS) >= 6
    for name, values in CONTROL_PRESETS.items():
        assert len(values) == 4, f"{name}: expected 4 values, got {len(values)}"
        assert all(0 <= v <= 1 for v in values), f"{name}: values out of range"
    print(f"  [PASS] test_control_presets ({len(CONTROL_PRESETS)} presets)")


if __name__ == "__main__":
    print("Running generation tests...")
    test_top_k_filter()
    test_top_p_filter()
    test_sample_top_k_top_p()
    test_sample_temperature_0()
    test_limit_leap()
    test_control_presets()
    print("All generation tests passed!")
