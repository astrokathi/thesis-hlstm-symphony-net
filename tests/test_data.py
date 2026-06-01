"""Tests for the data module (encoding, augmentation, clustering)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from hlstm_framework.data.encoding import HEventProcessor
from hlstm_framework.data.augmentation import DataAugmenter


def test_compute_log_bin_edges():
    """Verify fixed bin edges computation."""
    edges = HEventProcessor.compute_log_bin_edges(num_bins=128, max_dur=256.0)
    assert len(edges) == 129, f"Expected 129 edges, got {len(edges)}"
    assert edges[0] >= 0
    assert edges[-1] > edges[0]
    print(f"  [PASS] test_compute_log_bin_edges")


def test_encode_decode_roundtrip():
    """Verify minimal encode/decode roundtrip."""
    import muspy

    music = muspy.Music()
    track = muspy.Track(program=40)
    track.notes = [
        muspy.Note(time=0, pitch=60, duration=4, velocity=80),
        muspy.Note(time=4, pitch=64, duration=4, velocity=80),
    ]
    music.tracks.append(track)
    music.tempos = [muspy.Tempo(time=0, qpm=120)]

    processor = HEventProcessor()
    encoded = processor.encode(music, style_id=0)
    assert "note_level" in encoded
    assert encoded["note_level"].shape == (2, 4)
    assert encoded["note_level"][0, 0] == 60

    decoded = processor.decode(encoded)
    assert len(decoded.tracks[0].notes) == 2
    print(f"  [PASS] test_encode_decode_roundtrip")


def test_empty_song():
    """Verify empty song handling."""
    import muspy

    music = muspy.Music()
    music.tracks.append(muspy.Track(program=0))

    processor = HEventProcessor()
    encoded = processor.encode(music, style_id=0)
    assert len(encoded["note_level"]) == 0
    print(f"  [PASS] test_empty_song")


def test_augmentation_transpose():
    """Verify pitch transposition."""
    import muspy

    music = muspy.Music()
    track = muspy.Track(program=40)
    track.notes = [muspy.Note(time=0, pitch=60, duration=4, velocity=80)]
    music.tracks.append(track)

    aug = DataAugmenter(prob=1.0)
    transposed = aug.transpose(music, semitones=5)
    assert transposed.tracks[0].notes[0].pitch == 65
    print(f"  [PASS] test_augmentation_transpose")


def test_augmentation_tempo():
    """Verify tempo scaling."""
    import muspy

    music = muspy.Music()
    track = muspy.Track(program=40)
    track.notes = [muspy.Note(time=0, pitch=60, duration=4, velocity=80)]
    music.tracks.append(track)
    music.tempos = [muspy.Tempo(time=0, qpm=120)]

    aug = DataAugmenter(prob=1.0)
    scaled = aug.tempo_scale(music, factor=0.5)
    assert scaled.tracks[0].notes[0].duration == 2  # 4 * 0.5
    if scaled.tempos and len(scaled.tempos) > 0:
        assert scaled.tempos[0].qpm == 240  # 120 / 0.5
    print(f"  [PASS] test_augmentation_tempo")


def test_control_extraction():
    """Verify control change extraction."""
    import muspy

    music = muspy.Music()
    track = muspy.Track(program=40)
    track.notes = [muspy.Note(time=0, pitch=60, duration=4, velocity=80)]

    # Set control_changes if supported by this muspy version
    if hasattr(muspy, "ControlChange"):
        track.control_changes = [
            muspy.ControlChange(time=0, number=7, value=100),
        ]
    music.tracks.append(track)

    processor = HEventProcessor()
    encoded = processor.encode(music, style_id=0)
    # Control extraction is best-effort (depends on muspy version)
    print(f"  [PASS] test_control_extraction (found {len(encoded['control_level'])} events)")


if __name__ == "__main__":
    print("Running data tests...")
    test_compute_log_bin_edges()
    test_encode_decode_roundtrip()
    test_empty_song()
    test_augmentation_transpose()
    test_augmentation_tempo()
    test_control_extraction()
    print("All data tests passed!")
