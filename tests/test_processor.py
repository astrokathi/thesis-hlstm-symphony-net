"""Unit tests for HEventProcessor encoding/decoding."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from h_event_processor import HEventProcessor


def test_encode_decode_roundtrip():
    """Create a minimal muspy Music, encode, decode, and verify structure."""
    import muspy

    music = muspy.Music()
    track = muspy.Track(program=40)  # Violin
    track.notes = [
        muspy.Note(time=0, pitch=60, duration=4, velocity=80),
        muspy.Note(time=4, pitch=64, duration=4, velocity=80),
        muspy.Note(time=8, pitch=67, duration=4, velocity=80),
    ]
    music.tracks.append(track)
    music.tempos = [muspy.Tempo(time=0, qpm=120)]

    processor = HEventProcessor()
    encoded = processor.encode(music, style_id=0)

    assert "note_level" in encoded, "missing note_level"
    assert "instr_level" in encoded, "missing instr_level"
    assert "song_level" in encoded, "missing song_level"
    assert "control_level" in encoded, "missing control_level"

    note_level = encoded["note_level"]
    assert note_level.shape == (3, 4), f"unexpected shape: {note_level.shape}"
    assert note_level[0, 0] == 60, f"first pitch should be 60, got {note_level[0, 0]}"

    # Decode and verify
    decoded = processor.decode(encoded)
    assert len(decoded.tracks) > 0, "decoded has no tracks"
    assert len(decoded.tracks[0].notes) == 3, f"expected 3 notes, got {len(decoded.tracks[0].notes)}"

    print(f"  [PASS] test_encode_decode_roundtrip")


def test_fixed_duration_bins():
    """Verify fixed binning produces consistent bin indices."""
    import muspy

    processor = HEventProcessor()
    edges = processor.compute_log_bin_edges(num_bins=128, max_dur=256.0)
    assert len(edges) == 129, f"expected 129 edges, got {len(edges)}"
    assert edges[0] >= 0, "first edge should be >= 0"
    assert edges[-1] > edges[0], "edges should be increasing"
    print(f"  [PASS] test_fixed_duration_bins (first={edges[0]:.2f}, last={edges[-1]:.2f})")


def test_empty_song():
    """Verify encoder handles empty songs gracefully."""
    import muspy

    music = muspy.Music()
    track = muspy.Track(program=0)
    music.tracks.append(track)

    processor = HEventProcessor()
    encoded = processor.encode(music, style_id=0)

    assert len(encoded["note_level"]) == 0, "empty song should have 0 notes"
    print(f"  [PASS] test_empty_song")


def test_control_extraction():
    """Verify control changes are captured during encoding."""
    import muspy

    music = muspy.Music()
    track = muspy.Track(program=40)
    track.notes = [muspy.Note(time=0, pitch=60, duration=4, velocity=80)]
    track.control_changes = [
        muspy.ControlChange(time=0, number=7, value=100),
        muspy.ControlChange(time=2, number=11, value=90),
    ]
    music.tracks.append(track)

    processor = HEventProcessor()
    encoded = processor.encode(music, style_id=0)

    assert len(encoded["control_level"]) == 2, f"expected 2 control events, got {len(encoded['control_level'])}"
    print(f"  [PASS] test_control_extraction")


if __name__ == "__main__":
    print("Running processor tests...")
    test_encode_decode_roundtrip()
    test_fixed_duration_bins()
    test_empty_song()
    test_control_extraction()
    print("All processor tests passed!")
