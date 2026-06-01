"""
Generation module — samplers, presets, and MusicGenerator.

Usage:
    >>> from hlstm_framework.generation import MusicGenerator
    >>> gen = MusicGenerator(model, processor, device="mps")
    >>> midi, tokens = gen.generate(prompt, style=0, num_steps=200)
    >>> midi, tokens = gen.generate_with_preset("gentle", ...)
"""

from hlstm_framework.generation.sampler import sample_top_k_top_p, top_k_top_p_filter
from hlstm_framework.generation.presets import CONTROL_PRESETS
from hlstm_framework.generation.generator import MusicGenerator

__all__ = [
    "sample_top_k_top_p",
    "top_k_top_p_filter",
    "CONTROL_PRESETS",
    "MusicGenerator",
]
