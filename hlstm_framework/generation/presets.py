"""
Control presets for expressive music generation.

Each preset is a 4-element array: [modulation, volume, expression, sustain]
with values in [0, 1] mapped to MIDI controller ranges.

Usage:
    >>> from hlstm_framework.generation.presets import CONTROL_PRESETS, list_presets
    >>> list_presets()
    >>> ctrl = CONTROL_PRESETS["expressive"]
"""

from typing import Dict, List

CONTROL_PRESETS: Dict[str, List[float]] = {
    "expressive": [0.8, 0.7, 0.9, 0.3],
    "gentle": [0.2, 0.6, 0.7, 0.1],
    "bright": [0.4, 0.8, 0.8, 0.2],
    "sustained": [0.3, 0.7, 0.7, 0.9],
    "percussive": [0.1, 0.8, 0.6, 0.0],
    "dreamy": [0.9, 0.5, 0.8, 0.7],
    "neutral": [0.0, 0.7, 0.7, 0.0],
    "nothing": [0.0, 0.0, 0.0, 0.0],
}


def list_presets() -> None:
    """Print all available control presets with their values."""
    print("Available control presets:")
    print("-" * 40)
    for name, values in CONTROL_PRESETS.items():
        labels = ["mod", "vol", "expr", "sus"]
        parts = [f"{l}={v:.1f}" for l, v in zip(labels, values)]
        print(f"  {name:15s}  ({', '.join(parts)})")
