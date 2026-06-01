"""
Visualization and comparison utilities for generated music.

Produces piano-roll plots, velocity time series, metrics bar charts,
and radar comparison charts.

Usage:
    >>> from hlstm_framework.evaluation import MetricsVisualizer, compare_generations
    >>> viz = MetricsVisualizer(output_dir="output/plots")
    >>> viz.plot_piano_roll(midi_music, "My Generation")
    >>> compare_generations([m1, m2], labels=["A", "B"])
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import muspy


class MetricsVisualizer:
    """Plot piano rolls, velocity, metrics, and comparison charts.

    Args:
        output_dir: Directory to save plots.
    """

    def __init__(self, output_dir: str = "output/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_piano_roll(
        self,
        music: muspy.Music,
        title: str = "Generated Music",
        filename: str = "piano_roll.png",
    ) -> str:
        """Plot a piano roll time-series of the generated music.

        Args:
            music: muspy.Music object to visualize.
            title: Plot title.
            filename: Output filename (saved to output_dir).

        Returns:
            Full path to the saved PNG.
        """
        fig, ax = plt.subplots(figsize=(15, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, len(music.tracks)))

        for i, track in enumerate(music.tracks):
            if len(track.notes) == 0:
                continue
            color = colors[i]
            for note in track.notes:
                rect = patches.Rectangle(
                    (note.time, note.pitch - 0.4),
                    note.duration, 0.8,
                    linewidth=1, edgecolor="black",
                    facecolor=color, alpha=0.7,
                )
                ax.add_patch(rect)

        total_notes = sum(len(t.notes) for t in music.tracks)
        ax.set_xlabel("Time (ticks)")
        ax.set_ylabel("Pitch")
        ax.set_title(f"{title}  |  {total_notes} notes")
        ax.set_ylim(0, 128)
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    def plot_velocity(
        self,
        music: muspy.Music,
        title: str = "Velocity Over Time",
        filename: str = "velocity.png",
    ) -> str:
        """Plot velocity values as a scatter plot over time.

        Args:
            music: muspy.Music object.
            title: Plot title.
            filename: Output filename.

        Returns:
            Full path to the saved PNG.
        """
        fig, ax = plt.subplots(figsize=(15, 4))
        for track in music.tracks:
            if len(track.notes) == 0:
                continue
            times = [n.time for n in track.notes]
            velocities = [n.velocity for n in track.notes]
            ax.scatter(times, velocities, alpha=0.6, s=20, label=f"Instr {track.program}")

        ax.set_xlabel("Time (ticks)")
        ax.set_ylabel("Velocity")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    def plot_metrics_bar(
        self,
        metrics: Dict[str, float],
        title: str = "Generation Metrics",
        filename: str = "metrics.png",
    ) -> str:
        """Plot a bar chart of evaluation metrics.

        Args:
            metrics: Dict of metric name → value.
            title: Plot title.
            filename: Output filename.

        Returns:
            Full path to the saved PNG.
        """
        categories = [
            ("Structural", ["ssm_score", "scale_consistency"], "blue"),
            ("Rhythmic", ["ioi_variance"], "green"),
            ("Conditional", ["instrument_usage_ratio"], "orange"),
            ("Style", ["style_score", "pitch_entropy"], "red"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for ax, (cat_name, keys, color) in zip(axes, categories):
            available = {k: metrics[k] for k in keys if k in metrics}
            if available:
                names = list(available.keys())
                values = list(available.values())
                ax.bar(names, values, color=color, alpha=0.7)
                for n, v in zip(names, values):
                    ax.text(n, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
                ax.set_title(cat_name)
                ax.set_ylim(0, 1.1)
                ax.tick_params(axis="x", rotation=30)
                ax.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    def plot_radar(
        self,
        metrics_list: List[Dict[str, float]],
        labels: List[str],
        title: str = "Generation Comparison",
        filename: str = "radar_comparison.png",
    ) -> str:
        """Plot a radar chart comparing multiple generations.

        Args:
            metrics_list: List of metric dicts (one per generation).
            labels: Labels for each generation.
            title: Chart title.
            filename: Output filename.

        Returns:
            Full path to the saved PNG.
        """
        if not metrics_list:
            return ""

        all_keys = sorted(set().union(*metrics_list))
        angles = np.linspace(0, 2 * np.pi, len(all_keys), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_list)))

        for metrics, label, color in zip(metrics_list, labels, colors):
            values = [min(metrics.get(k, 0), 1.0) for k in all_keys]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=label, color=color, alpha=0.7)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_keys, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(title, y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path


def compare_generations(
    music_list: List[muspy.Music],
    labels: List[str],
    instruments: Optional[List[int]] = None,
    output_dir: str = "output/comparison",
) -> str:
    """Convenience function: compute metrics for multiple generations and plot a radar.

    Args:
        music_list: List of generated muspy.Music objects.
        labels: Labels for each generation.
        instruments: Target instrument list (for IUR).
        output_dir: Output directory.

    Returns:
        Path to the saved radar chart.
    """
    from hlstm_framework.evaluation.metrics import MusicMetrics

    calc = MusicMetrics(instruments=instruments)
    all_metrics = [calc.compute_all(m) for m in music_list]

    viz = MetricsVisualizer(output_dir=output_dir)
    return viz.plot_radar(all_metrics, labels, title="Generation Comparison")
