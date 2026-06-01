"""Tests for the evaluation module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import muspy
import numpy as np

from hlstm_framework.evaluation import MusicMetrics, MetricsVisualizer


def _create_test_music(num_tracks=2, notes_per_track=5):
    """Create a minimal test muspy.Music object."""
    music = muspy.Music()
    for i in range(num_tracks):
        track = muspy.Track(program=40 + i)
        for j in range(notes_per_track):
            track.notes.append(
                muspy.Note(time=j * 10, pitch=60 + j, duration=4, velocity=80)
            )
        music.tracks.append(track)
    return music


def test_self_similarity():
    """Verify SSM returns a float in [0,1]."""
    music = _create_test_music()
    calc = MusicMetrics()
    score = calc.self_similarity(music)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    print(f"  [PASS] test_self_similarity (score={score:.4f})")


def test_ioi_variance():
    """Verify IOI variance returns non-negative float."""
    music = _create_test_music()
    calc = MusicMetrics()
    var = calc.ioi_variance(music)
    assert isinstance(var, float)
    assert var >= 0
    print(f"  [PASS] test_ioi_variance (var={var:.4f})")


def test_instrument_usage():
    """Verify IUR with matching instruments."""
    music = _create_test_music()
    calc = MusicMetrics(instruments=[40, 41])
    iur = calc.instrument_usage(music)
    assert isinstance(iur, float)
    assert 0.0 <= iur <= 1.0
    assert iur > 0.5  # Most notes use target instruments
    print(f"  [PASS] test_instrument_usage (iur={iur:.4f})")


def test_instrument_usage_wrong():
    """Verify IUR with non-matching instruments."""
    music = _create_test_music()
    calc = MusicMetrics(instruments=[0])  # None of our tracks use program 0
    iur = calc.instrument_usage(music)
    assert iur == 0.0
    print(f"  [PASS] test_instrument_usage_wrong (iur={iur:.4f})")


def test_compute_all():
    """Verify compute_all returns dict with expected keys."""
    music = _create_test_music()
    calc = MusicMetrics(instruments=[40, 41])
    metrics = calc.compute_all(music)
    expected_keys = {"ssm_score", "ioi_variance", "instrument_usage_ratio", "style_score"}
    assert expected_keys.issubset(metrics.keys()), f"Missing keys: {expected_keys - metrics.keys()}"
    print(f"  [PASS] test_compute_all ({len(metrics)} metrics)")


def test_visualizer_plots(tmp_path="/tmp/test_hlstm_viz"):
    """Verify visualizer generates plot files."""
    import os
    music = _create_test_music()
    viz = MetricsVisualizer(output_dir=tmp_path)

    path = viz.plot_piano_roll(music, "Test")
    assert os.path.exists(path), f"Plot not created: {path}"

    path2 = viz.plot_velocity(music, "Test Vel")
    assert os.path.exists(path2)

    print(f"  [PASS] test_visualizer_plots")


if __name__ == "__main__":
    print("Running evaluation tests...")
    test_self_similarity()
    test_ioi_variance()
    test_instrument_usage()
    test_instrument_usage_wrong()
    test_compute_all()
    test_visualizer_plots()
    print("All evaluation tests passed!")
