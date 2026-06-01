"""
Evaluation module — metrics, visualizers, comparison tools.

Usage:
    >>> from hlstm_framework.evaluation import MusicMetrics, compare_generations
    >>> metrics = MusicMetrics(instruments=[40,41,42])
    >>> scores = metrics.compute_all(midi_music)
    >>> compare_generations([midi1, midi2], labels=["A", "B"])
"""

from hlstm_framework.evaluation.metrics import MusicMetrics
from hlstm_framework.evaluation.visualizer import MetricsVisualizer, compare_generations

__all__ = [
    "MusicMetrics",
    "MetricsVisualizer",
    "compare_generations",
]
