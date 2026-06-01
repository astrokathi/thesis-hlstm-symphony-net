"""
Shared sampling utilities for music generation.

Provides temperature scaling, top-k filtering, top-p (nucleus) filtering,
and combined sampling functions.

Usage:
    >>> from hlstm_framework.generation.sampler import sample_top_k_top_p
    >>> idx = sample_top_k_top_p(logits, temperature=0.7, top_k=20, top_p=0.9)
"""

import torch
import torch.nn.functional as F


def top_k_top_p_filter(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """Apply top-k and/or top-p (nucleus) filtering to a 1D logits tensor.

    Args:
        logits: 1D tensor of raw logits.
        top_k: If > 0, keep only the top-k values (set rest to -inf).
        top_p: If > 0.0, keep the smallest set of tokens whose cumulative
               probability exceeds top_p.

    Returns:
            Filtered logits tensor (same shape).
    """
    logits = logits.clone()
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        min_topk = topk_vals[-1]
        logits[logits < min_topk] = -float("Inf")

    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        sorted_idx_to_remove = cumulative_probs > top_p
        sorted_idx_to_remove[..., 0] = False  # keep at least one

        indices_to_remove = sorted_idx[sorted_idx_to_remove]
        logits[indices_to_remove] = -float("Inf")

    return logits


def sample_top_k_top_p(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> int:
    """Sample a token index from logits with temperature, top-k, and top-p.

    Args:
        logits: 1D tensor of raw logits for a single head.
        temperature: Scaling factor (<1.0 sharpens, >1.0 flattens).
        top_k: Top-k filtering threshold (0 = disabled).
        top_p: Nucleus filtering threshold (0.0 = disabled).

    Returns:
        Sampled integer token index.
    """
    if temperature > 0:
        logits = logits / (temperature + 1e-9)
    else:
        # argmax sampling at temperature=0
        return int(torch.argmax(logits).item())

    filtered = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(filtered, dim=-1)

    # Fallback if everything was filtered
    if torch.all(torch.isinf(filtered)):
        probs = F.softmax(logits, dim=-1)

    return int(torch.multinomial(probs, num_samples=1).item())


def limit_leap(
    prev_pitch: int,
    pitch_logits: torch.Tensor,
    max_leap: int = 12,
) -> torch.Tensor:
    """Zero out probabilities for pitches beyond max_leap from prev_pitch.

    Args:
        prev_pitch: Previous pitch value.
        pitch_logits: 1D tensor of pitch logits.
        max_leap: Maximum allowed interval in semitones.

    Returns:
        Filtered pitch logits.
    """
    v = pitch_logits.clone()
    pitch_indices = torch.arange(v.size(-1), device=v.device)
    mask = torch.abs(pitch_indices - prev_pitch) > max_leap
    v[mask] = -float("Inf")

    # If mask removes everything, return original
    if torch.all(torch.isinf(v)):
        return pitch_logits
    return v
