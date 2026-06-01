"""
Objective evaluation metrics for symbolic music generation.

Metrics align with the H-LSTM hierarchical layers:
    - SSM Score:      Structural coherence (Layer 1 validation)
    - IOI Variance:   Rhythmic consistency (Layer 2 validation)
    - IUR:            Instrument condition fidelity (Layer 2 validation)
    - Style Score:    Style adherence (proxy)
    - MusPy metrics:  Pitch range, entropy, scale consistency, note density

Usage:
    >>> from hlstm_framework.evaluation import MusicMetrics
    >>> calc = MusicMetrics(instruments=[40, 41, 42])
    >>> scores = calc.compute_all(midi_music)
    >>> scores["ssm_score"]
    0.42
"""

from typing import Dict, List, Optional

import muspy
import numpy as np
from scipy.spatial.distance import pdist, squareform

from hlstm_framework.config import load_settings


class MusicMetrics:
    """Compute objective metrics for generated symbolic music.

    Args:
        instruments: List of target MIDI program numbers (for IUR).
        style: Style label (0=classical, 1=contemporary) for style scoring.
        window_size: SSM window size in time ticks.

    Attributes:
        instruments: Target instrument list.
        style: Style label.
        window_size: SSM analysis window.
    """

    def __init__(
        self,
        instruments: Optional[List[int]] = None,
        style: Optional[int] = None,
        window_size: int = 128,
    ):
        self.instruments = instruments or []
        self.style = style
        self.window_size = window_size

    def __help__(self) -> None:
        """Print usage information for MusicMetrics."""
        print("MusicMetrics — objective evaluation metrics")
        print("=" * 50)
        print("Methods:")
        print("  compute_all(music)             — All metrics at once")
        print("  self_similarity(music)          — Structural coherence (SSM)")
        print("  ioi_variance(music)             — Rhythmic consistency")
        print("  instrument_usage(music)          — Conditional fidelity (IUR)")
        print("  style_score(music)               — Style adherence proxy")
        print("  muspy_metrics(music)             — Built-in muspy metrics")

    # --- SSM Score: Structural coherence ---

    def self_similarity(self, music: muspy.Music) -> float:
        """Self-Similarity Matrix Score — measures motif recurrence.

        Constructs pitch-time vectors over sliding windows and computes
        cosine similarity between consecutive windows. Higher scores
        indicate stronger phrase repetition and structural coherence.

        Args:
            music: Generated muspy.Music object.

        Returns:
            SSM score in [0, 1], higher = more coherent.
        """
        # Extract per-track note sequences
        sequences = []
        for track in music.tracks:
            if len(track.notes) == 0:
                continue
            sequences.append([
                {"pitch": n.pitch, "time": n.time, "duration": n.duration}
                for n in track.notes
            ])

        if not sequences:
            return 0.0

        max_time = music.get_end_time()
        if max_time == 0:
            return 0.0

        # Create pitch-time vectors per window
        vectors = []
        for start in range(0, max_time, self.window_size):
            vec = []
            for seq in sequences:
                vec.append(sum(1 for n in seq if start <= n["time"] < start + self.window_size))
            vectors.append(vec)

        if len(vectors) < 2:
            return 0.0

        try:
            arr = np.array(vectors)
            if np.all(arr == 0):
                return 0.0
            sim = 1 - squareform(pdist(arr, metric="cosine"))
            if np.any(np.isnan(sim)):
                return 0.0
            score = float(np.mean([sim[i, i + 1] for i in range(len(vectors) - 1)]))
            return score if not np.isnan(score) else 0.0
        except Exception:
            return 0.0

    # --- IOI Variance: Rhythmic consistency ---

    def ioi_variance(self, music: muspy.Music) -> float:
        """Inter-Onset Interval Variance — measures rhythmic stability.

        Computes variance of time differences between consecutive note onsets.
        Lower variance = more regular pulse = better rhythmic consistency.

        Args:
            music: Generated muspy.Music object.

        Returns:
            IOI variance value (lower = more stable).
        """
        all_iois = []
        for track in music.tracks:
            if len(track.notes) < 2:
                continue
            sorted_notes = sorted(track.notes, key=lambda n: n.time)
            for i in range(1, len(sorted_notes)):
                ioi = sorted_notes[i].time - sorted_notes[i - 1].time
                if ioi > 0:
                    all_iois.append(ioi)

        if len(all_iois) < 2:
            return 0.0
        return float(np.var(all_iois))

    # --- IUR: Instrument conditional fidelity ---

    def instrument_usage(self, music: muspy.Music) -> float:
        """Instrument Usage Ratio — measures conditional fidelity.

        Ratio of generated notes that use the requested instrument set,
        multiplied by a penalty for using non-target instruments.

        Args:
            music: Generated muspy.Music object.

        Returns:
            IUR score in [0, 1], higher = better fidelity.
        """
        if not self.instruments:
            return 1.0

        target_set = set(self.instruments)
        used = set()
        total_notes = 0
        target_notes = 0

        for track in music.tracks:
            instr = track.program
            used.add(instr)
            count = len(track.notes)
            total_notes += count
            if instr in target_set:
                target_notes += count

        if total_notes == 0:
            return 0.0

        iur = target_notes / total_notes
        extra = len(used - target_set)
        penalty = 1.0 / (1.0 + extra)
        return iur * penalty

    # --- Style score (proxy) ---

    def style_score(self, music: muspy.Music) -> float:
        """Style perplexity proxy — measures style adherence.

        Lower scores indicate better alignment with typical style features.
        Uses pitch range, note density, and rhythm variance as proxies.

        Args:
            music: Generated muspy.Music object.

        Returns:
            Style score (lower = more stylistically consistent).
        """
        features = []

        try:
            features.append(muspy.pitch_range(music))
        except Exception:
            features.append(0)

        total_notes = sum(len(t.notes) for t in music.tracks)
        total_time = max(music.get_end_time(), 1)
        features.append(total_notes / total_time)

        features.append(self.ioi_variance(music))

        normalized = [
            min(features[0] / 60.0, 1.0) if len(features) > 0 else 0.5,
            min(features[1] / 0.5, 1.0) if len(features) > 1 else 0.5,
            min(features[2] / 100.0, 1.0) if len(features) > 2 else 0.5,
        ]
        return float(np.mean(normalized))

    # --- MusPy built-in metrics ---

    def muspy_metrics(self, music: muspy.Music) -> Dict[str, float]:
        """Collect built-in muspy metrics.

        Args:
            music: Generated muspy.Music object.

        Returns:
            Dict of metric name → float value.
        """
        result: Dict[str, float] = {}

        try:
            result["pitch_range"] = muspy.pitch_range(music)
            result["pitch_entropy"] = muspy.pitch_entropy(music)
            result["scale_consistency"] = muspy.scale_consistency(music)
        except Exception:
            pass

        total_notes = sum(len(t.notes) for t in music.tracks)
        result["note_density"] = total_notes / max(music.get_end_time(), 1)

        return result

    # --- Composite ---

    def compute_all(self, music: muspy.Music) -> Dict[str, float]:
        """Compute all evaluation metrics for a generated piece.

        Args:
            music: Generated muspy.Music object.

        Returns:
            Dict with keys: ssm_score, ioi_variance, instrument_usage_ratio,
            style_score, plus muspy metrics.
        """
        metrics = {
            "ssm_score": self.self_similarity(music),
            "ioi_variance": self.ioi_variance(music),
            "instrument_usage_ratio": self.instrument_usage(music),
            "style_score": self.style_score(music),
        }
        metrics.update(self.muspy_metrics(music))
        return metrics
