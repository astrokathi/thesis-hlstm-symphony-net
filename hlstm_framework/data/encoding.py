"""
Hierarchical event encoding/decoding for H-LSTM symbolic music processing.

Transforms raw muspy.Music objects into structured 4-level representations:
    - Note level:   [pitch, duration, velocity, instrument_id] per event
    - Instrument level: Active instrument program numbers
    - Song level:   [tempo, time_sig, num_tracks, length, style_id]
    - Control level: MIDI controller events [instr_id, time, cc_number, value]

Supports both per-song log-scaled duration binning and global fixed bins.

Usage:
    >>> from hlstm_framework.data.encoding import HEventProcessor
    >>> processor = HEventProcessor()
    >>> encoded = processor.encode(song, style_id=0)
    >>> decoded = processor.decode(encoded)
"""

from typing import Dict, List, Optional, Tuple

import muspy
import numpy as np


class HEventProcessor:
    """Hierarchical event processor for H-LSTM symbolic music encoding/decoding.

    Encodes muspy.Music objects into a 4-level dictionary consumable by the
    H-LSTM model. Decodes model output back into playable muspy.Music.

    Args:
        num_duration_bins: Number of duration quantization bins (default: 128).
        generate_expression: If True, adds heuristic expression controls on decode.
        duration_bin_edges: Pre-computed fixed bin edges. None = per-song bins.
        max_raw_duration: Max duration for fixed bin edge computation.

    Attributes:
        num_duration_bins: Duration vocabulary size.
        reverse_instrument_map: Mapping from encoded to original instrument IDs.
    """

    def __init__(
        self,
        num_duration_bins: int = 128,
        generate_expression: bool = True,
        duration_bin_edges: Optional[np.ndarray] = None,
        max_raw_duration: float = 256.0,
    ):
        self.num_duration_bins = num_duration_bins
        self.generate_expression = generate_expression
        self.duration_bin_edges = duration_bin_edges
        self.max_raw_duration = max_raw_duration
        self.reverse_instrument_map: Dict[int, int] = {i: i for i in range(128)}
        self.max_time: int = 0

    def __help__(self) -> None:
        """Print usage information for HEventProcessor."""
        print("HEventProcessor — Hierarchical event encoding/decoding")
        print("=" * 50)
        print("Levels:")
        print("  note_level:     [pitch, duration, velocity, instr_id] — per note")
        print("  instr_level:    Active instrument IDs")
        print("  song_level:     [tempo, time_sig, tracks, length, style]")
        print("  control_level:  [instr_id, time, cc_number, cc_value]")
        print()
        print("Methods:")
        print("  encode(song, style_id, use_fixed_bins) → dict")
        print("  decode(encoded_dict)                   → muspy.Music")
        print()
        print(f"Duration bins: {self.num_duration_bins}")
        print(f"Fixed edges: {'configured' if self.duration_bin_edges is not None else 'per-song'}")

    # --- Public API ---

    def encode(
        self,
        song: muspy.Music,
        style_id: int = 0,
        use_fixed_bins: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Encode a muspy.Music object into hierarchical token arrays.

        Args:
            song: Input musical composition.
            style_id: Integer style label (0=classical, 1=contemporary, etc.).
            use_fixed_bins: If True, use corpus-level fixed duration bin edges.

        Returns:
            Dictionary with keys:
                - "note_level":  (N, 4) int16 array [pitch, dur, vel, instr]
                - "instr_level": (M,) int16 array of active instrument IDs
                - "song_level":  (5,) int array [tempo, time_sig, tracks, length, style]
                - "control_level": (C, 4) int16 array of control events
        """
        note_dicts: List[Dict] = []
        instr_list: List[Dict] = []
        control_list: List[Dict] = []

        tempo = float(np.mean([t.qpm for t in song.tempos])) if song.tempos else 120.0
        time_sig = song.time_signatures[0].numerator if song.time_signatures else 4
        raw_durations: List[float] = []

        self.max_time = 0

        for track in song.tracks:
            instr_id = track.program
            self.reverse_instrument_map[instr_id] = track.program

            instr_list.append({
                "instrument_id": instr_id,
                "program": track.program,
                "num_notes": len(track.notes),
            })

            for note in track.notes:
                raw_durations.append(note.duration)
                note_dicts.append({
                    "instrument_id": instr_id,
                    "pitch": note.pitch,
                    "start": note.start,
                    "duration": note.duration,
                    "velocity": note.velocity,
                })
                self.max_time = max(self.max_time, note.end)

            # Capture control changes
            if hasattr(track, "control_changes") and track.control_changes:
                for cc in track.control_changes:
                    control_list.append({
                        "instrument_id": instr_id,
                        "time": cc.time,
                        "control_number": cc.number,
                        "value": cc.value,
                    })

        # -- Duration quantization --
        raw_dur_arr = np.array([n["duration"] for n in note_dicts], dtype=np.float32)

        if use_fixed_bins and self.duration_bin_edges is not None:
            log_durs = np.digitize(raw_dur_arr, self.duration_bin_edges) - 1
            log_durs = np.clip(log_durs, 0, self.num_duration_bins - 1).astype(int)
        elif use_fixed_bins and self.duration_bin_edges is None:
            # Compute on first use
            self.duration_bin_edges = self.compute_log_bin_edges(
                self.num_duration_bins, self.max_raw_duration
            )
            log_durs = np.digitize(raw_dur_arr, self.duration_bin_edges) - 1
            log_durs = np.clip(log_durs, 0, self.num_duration_bins - 1).astype(int)
        else:
            # Per-song log-scaled bins
            if len(raw_dur_arr) > 0:
                log_durs = np.log1p(raw_dur_arr)
                max_log = log_durs.max()
                if max_log > 0:
                    log_durs = (log_durs / max_log * (self.num_duration_bins - 1)).astype(int)
                else:
                    log_durs = np.zeros(len(note_dicts), dtype=int)
            else:
                log_durs = np.zeros(len(note_dicts), dtype=int)

        # Build arrays
        if note_dicts:
            encoded_notes = np.array([
                (n["pitch"], log_durs[i], n["velocity"], n["instrument_id"])
                for i, n in enumerate(note_dicts)
            ], dtype=np.int16)
        else:
            encoded_notes = np.zeros((0, 4), dtype=np.int16)

        encoded_instruments = np.unique(encoded_notes[:, 3]) if len(encoded_notes) > 0 else np.array([], dtype=np.int16)
        encoded_song = np.array([
            tempo, time_sig, len(instr_list), self.max_time, style_id
        ], dtype=np.int32)

        encoded_controls = np.array([
            (c["instrument_id"], c["time"], c["control_number"], c["value"])
            for c in control_list
        ], dtype=np.int16) if control_list else np.zeros((0, 4), dtype=np.int16)

        return {
            "note_level": encoded_notes,
            "instr_level": encoded_instruments,
            "song_level": encoded_song,
            "control_level": encoded_controls,
        }

    def decode(self, encoded_dict: Dict[str, np.ndarray]) -> muspy.Music:
        """Decode a token dictionary back into a muspy.Music object.

        Args:
            encoded_dict: Dictionary from encode() or from model generation.

        Returns:
            Reconstructed muspy.Music with tracks, notes, and control changes.
        """
        import muspy

        music = muspy.Music()
        note_level = encoded_dict.get("note_level", np.zeros((0, 4), dtype=np.int16))

        if len(note_level) == 0:
            return music

        max_dur = max(note_level[:, 1])
        max_log_val = np.log1p(max_dur) if max_dur > 0 else 1.0

        tracks: Dict[int, dict] = {}
        for note in note_level:
            pitch, quantized_duration, velocity, instr_id = note
            duration = int(np.expm1(
                quantized_duration / max(self.num_duration_bins - 1, 1) * max_log_val
            ))

            if instr_id not in tracks:
                program = self.reverse_instrument_map.get(int(instr_id), 0)
                tracks[int(instr_id)] = {
                    "track": muspy.Track(program=program),
                    "last_end": 0,
                }

            track_info = tracks[int(instr_id)]
            start_time = track_info["last_end"]
            track_info["track"].notes.append(
                muspy.Note(
                    time=int(start_time),
                    pitch=int(pitch),
                    duration=max(duration, 1),
                    velocity=int(velocity),
                )
            )
            track_info["last_end"] = start_time + max(duration, 1)

        for info in tracks.values():
            music.tracks.append(info["track"])

        # Decode control changes (guard against missing muspy.ControlChange)
        has_control_api = hasattr(muspy, "ControlChange")
        if "control_level" in encoded_dict and len(encoded_dict["control_level"]) > 0 and has_control_api:
            for ctrl in encoded_dict["control_level"]:
                instr_id, time_val, ctrl_num, value = ctrl
                while len(music.tracks) <= int(instr_id):
                    music.tracks.append(muspy.Track(program=0))
                track = music.tracks[int(instr_id)]
                if not hasattr(track, "control_changes") or track.control_changes is None:
                    track.control_changes = []
                track.control_changes.append(
                    muspy.ControlChange(
                        time=int(time_val),
                        number=int(ctrl_num),
                        value=int(value),
                    )
                )
        elif self.generate_expression:
            self._add_expression_controls(music)

        return music

    # --- Static helpers ---

    @staticmethod
    def compute_log_bin_edges(num_bins: int = 128, max_dur: float = 256.0) -> np.ndarray:
        """Compute fixed log-spaced bin edges for corpus-level duration quantization.

        Args:
            num_bins: Target number of bins.
            max_dur: Maximum duration to cover in the bin range.

        Returns:
            NumPy array of (num_bins + 1) bin edges in raw duration units.
        """
        log_space = np.logspace(0, np.log1p(max_dur), num_bins + 1, base=np.e)
        return np.expm1(log_space)

    # --- Private ---

    def _add_expression_controls(self, music: muspy.Music) -> None:
        """Add heuristic expression/modulation controls for natural playback.
        Uses muspy.ControlChange if available, otherwise silently skips."""
        for track in music.tracks:
            if not hasattr(track, "control_changes") or not hasattr(muspy, "ControlChange"):
                continue
            if track.control_changes is None:
                track.control_changes = []
            for note in track.notes:
                expr = int(70 + 40 * np.sin(note.time / 50.0))
                mod = int(40 + 25 * np.cos(note.time / 70.0))
                track.control_changes.append(
                    muspy.ControlChange(time=note.time, number=11, value=expr)
                )
                track.control_changes.append(
                    muspy.ControlChange(time=note.time, number=1, value=mod)
                )
