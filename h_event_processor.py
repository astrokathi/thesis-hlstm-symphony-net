import muspy
import numpy as np


class HEventProcessor:
    """Sprint 5: Configurable fixed or per-song duration binning."""

    def __init__(self, num_duration_bins=128, generate_expression=True,
                 duration_bin_edges=None, max_raw_duration=256.0):
        self.instrument_map = {}
        self.reverse_instrument_map = {i: i for i in range(128)}
        self.max_time = 0
        self.num_duration_bins = num_duration_bins
        self.generate_expression = generate_expression
        self.duration_bin_edges = duration_bin_edges
        self.max_raw_duration = max_raw_duration

    @staticmethod
    def compute_log_bin_edges(num_bins=128, max_dur=256.0):
        """Compute fixed log-spaced bin edges for global duration quantization."""
        log_space = np.logspace(0, np.log1p(max_dur), num_bins + 1, base=np.e)
        return np.expm1(log_space)

    def encode(self, song: muspy.Music, style_id: 0, use_fixed_bins=False):
        note_level, instr_level, song_level, control_level = [], [], [], []

        tempo = np.mean([int(t.qpm) for t in song.tempos]) if song.tempos else 120
        time_sig = song.time_signatures[0].numerator if song.time_signatures else 4
        song_level.append({
            "tempo": tempo,
            "time_signature": time_sig,
            "num_tracks": len(song.tracks),
            "length": song.get_end_time(),
            "style": style_id
        })

        raw_durations = list()

        for track in song.tracks:
            instr_id = track.program
            self.reverse_instrument_map[instr_id] = track.program

            instr_level.append({
                "instrument_id": instr_id,
                "program": track.program,
                "num_notes": len(track.notes),
            })

            for note in track.notes:
                raw_durations.append(note.duration)
                note_level.append({
                    "instrument_id": instr_id,
                    "pitch": note.pitch,
                    "start": note.start,
                    "duration": note.duration,
                    "velocity": note.velocity
                })
                self.max_time = max(self.max_time, note.end)

            # Capture control changes if available
            if hasattr(track, "control_changes"):
                for cc in track.control_changes:
                    control_level.append({
                        "instrument_id": instr_id,
                        "time": cc.time,
                        "control_number": cc.number,
                        "value": cc.value
                    })

        # ----- Sprint 5: Duration quantization (fixed or per-song) -----
        raw_durations = np.array([n["duration"] for n in note_level], dtype=np.float32)

        if use_fixed_bins:
            if self.duration_bin_edges is None:
                self.duration_bin_edges = self.compute_log_bin_edges(
                    self.num_duration_bins, self.max_raw_duration
                )
            log_durs = np.digitize(raw_durations, self.duration_bin_edges) - 1
            log_durs = np.clip(log_durs, 0, self.num_duration_bins - 1).astype(int)
        else:
            # Original per-song log-scaled binning
            if len(raw_durations) > 0:
                log_durs = np.log1p(raw_durations)
                max_log = log_durs.max()
                if max_log > 0:
                    log_durs = (log_durs / max_log * (self.num_duration_bins - 1)).astype(int)
                else:
                    log_durs = np.zeros(len(note_level), dtype=int)
            else:
                log_durs = np.zeros(len(note_level), dtype=int)

        encoded_notes = self._encode_note_tokens(note_level, log_durs)
        encoded_instruments = encoded_notes[:, 3]
        encoded_song = np.array([tempo, time_sig, len(instr_level), self.max_time, style_id])

        encoded_controls = np.array([
            (c["instrument_id"], c["time"], c["control_number"], c["value"])
            for c in control_level
        ], dtype=np.int16) if len(control_level) > 0 else np.zeros((0, 4), dtype=np.int16)

        return {
            "note_level": encoded_notes,      # shape [num_notes, 4]
            "instr_level": encoded_instruments,
            "song_level": encoded_song,
            "control_level": encoded_controls
        }

    def _encode_note_tokens(self, note_level, log_durs):
        """Return np array: [num_notes, 4] -> pitch, duration, velocity, instr_id"""
        return np.array([
            (n["pitch"], log_durs[i], n["velocity"], n["instrument_id"])
            for i, n in enumerate(note_level)
        ], dtype=np.int16)

    # def decode(self, encoded_dict):
    #     """Reconstruct a muspy.Music object with expressive control support."""
    #     music = muspy.Music()
    #     note_level = encoded_dict["note_level"]
    #
    #     if len(note_level) == 0:
    #         return music
    #
    #     max_dur = max(note_level[:, 1])
    #     max_log = np.log1p(max_dur)  # approximate max log from encoding
    #
    #     # Track time per instrument
    #     num_instruments = max(note_level[:, 3]) + 1
    #     instr_times = [0] * num_instruments
    #
    #     for note in note_level:
    #         pitch, quantized_duration, velocity, instr_id = note
    #         duration = int(np.expm1(quantized_duration / 127 * max_log))
    #         program = instr_id  # (or self.reverse_instrument_map[instr_id] if remapped)
    #         start = instr_times[instr_id]
    #
    #         # Ensure track exists
    #         while len(music.tracks) <= instr_id:
    #             music.tracks.append(muspy.Track(program=program))
    #
    #         music.tracks[instr_id].notes.append(
    #             muspy.Note(
    #                 time=start,
    #                 pitch=int(pitch),
    #                 duration=int(duration),
    #                 velocity=int(velocity)
    #             )
    #         )
    #         instr_times[instr_id] = start + duration
    #
    #     # Decode control changes if present
    #     if "control_level" in encoded_dict and len(encoded_dict["control_level"]) > 0:
    #         for ctrl in encoded_dict["control_level"]:
    #             instr_id, time, ctrl_num, value = ctrl
    #             while len(music.tracks) <= instr_id:
    #                 music.tracks.append(muspy.Track(program=instr_id))
    #             music.tracks[instr_id].control_changes.append(
    #                 muspy.ControlChange(time=int(time),
    #                                     number=int(ctrl_num),
    #                                     value=int(value))
    #             )
    #
    #     # If no controls exist, synthesize emotion dynamically
    #     elif self.generate_expression:
    #         self._add_emotional_controls(music)
    #
    #     return music

    def decode(self, encoded_dict):
        import muspy
        import numpy as np

        music = muspy.Music()
        note_level = encoded_dict["note_level"]

        if len(note_level) == 0:
            return music

        # Calculate actual durations from quantized values
        max_dur = max(note_level[:, 1]) if len(note_level) > 0 else 1
        max_log = np.log1p(max_dur)

        # Group by instrument and create tracks
        tracks = {}
        current_time = 0

        for i, note in enumerate(note_level):
            pitch, quantized_duration, velocity, instr_id = note
            duration = int(np.expm1(quantized_duration / 127 * max_log))

            # Get or create track for this instrument
            if instr_id not in tracks:
                program = self.reverse_instrument_map.get(instr_id, 0)
                tracks[instr_id] = {
                    'track': muspy.Track(program=program),
                    'last_end_time': 0
                }

            track_info = tracks[instr_id]

            # Simple polyphonic timing: allow some overlap, but mostly sequential
            if i == 0:
                start_time = 0
            else:
                # 40% chance to create chord/overlap (use same start time as previous in this track)
                if np.random.random() < 0.4 and i > 0:
                    start_time = track_info['last_end_time'] - 2  # Small overlap
                else:
                    # Sequential placement with small gap
                    start_time = track_info['last_end_time'] + np.random.randint(0, 4)

            start_time = max(0, start_time)  # Ensure non-negative

            track_info['track'].notes.append(
                muspy.Note(
                    time=int(start_time),
                    pitch=int(pitch),
                    duration=int(duration),
                    velocity=int(velocity),
                )
            )

            # Update end time for this track
            track_info['last_end_time'] = start_time + duration

        # Add all tracks to music
        for track_info in tracks.values():
            music.tracks.append(track_info['track'])

        return music

    def _add_emotional_controls(self, music):
        """Add heuristic dynamic and modulation controls to make playback more expressive."""
        for track in music.tracks:
            for note in track.notes:
                # Smooth sinusoidal dynamics
                expr = int(70 + 40 * np.sin(note.time / 50.0))
                mod = int(40 + 25 * np.cos(note.time / 70.0))

                track.control_changes.append(
                    muspy.ControlChange(time=note.time, number=11, value=expr)  # expression (dynamics)
                )
                track.control_changes.append(
                    muspy.ControlChange(time=note.time, number=1, value=mod)    # modulation (vibrato)
                )
