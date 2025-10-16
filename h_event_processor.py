import muspy
import numpy as np


class HEventProcessor:
    def __init__(self, num_duration_bins=128):
        self.instrument_map = {}
        self.reverse_instrument_map = {i: i for i in range(128)}
        self.max_time = 0
        self.num_duration_bins = num_duration_bins

    def encode(self, song: muspy.Music):
        note_level, instr_level, song_level = [], [], []

        tempo = np.mean([int(t.qpm) for t in song.tempos]) if song.tempos else 120
        time_sig = song.time_signatures[0].numerator if song.time_signatures else 4
        song_level.append({
            "tempo": tempo,
            "time_signature": time_sig,
            "num_tracks": len(song.tracks),
            "length": song.get_end_time(),
        })

        raw_durations = list()

        for track in song.tracks:
            instr_id = self.instrument_map.setdefault(track.program, len(self.instrument_map))
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

        # Quantize durations
        raw_durations = np.array([n["duration"] for n in note_level])
        if len(raw_durations) > 0:
            # Log scale helps spread values better
            log_durs = np.log1p(raw_durations)
            log_durs = (log_durs / log_durs.max() * (self.num_duration_bins - 1)).astype(int)
        else:
            log_durs = np.zeros(len(note_level), dtype=int)

        encoded_notes = self._encode_note_tokens(note_level, log_durs)
        # encoded_instruments = np.array([x["instrument_id"] for x in instr_level])
        encoded_instruments = encoded_notes[3]
        encoded_song = np.array([tempo, time_sig, len(instr_level), self.max_time])

        return {
            "note_level": encoded_notes,      # shape [num_notes, 4]
            "instr_level": encoded_instruments,
            "song_level": encoded_song
        }

    def _encode_note_tokens(self, note_level, log_durs):
        """Return np array: [num_notes, 4] -> pitch, duration, velocity, instr_id"""
        return np.array([
            (n["pitch"], log_durs[i], n["velocity"], n["instrument_id"])
            for i, n in enumerate(note_level)
        ], dtype=np.int16)

    # def decode(self, encoded_dict):
    #     music = muspy.Music()
    #     max_dur = max(encoded_dict["note_level"][:, 1]) if len(encoded_dict["note_level"]) > 0 else 1
    #     max_log = np.log1p(max_dur)  # approximate max log from encoding
    #     time = 0
    #     start = 0
    #     for note in encoded_dict["note_level"]:
    #         pitch, quantized_duration, velocity, instr_id = note
    #         duration = int(np.expm1(quantized_duration / 127 * max_log))
    #         program = self.reverse_instrument_map.get(instr_id, 0)
    #         start = time
    #         while len(music.tracks) <= instr_id:
    #             music.tracks.append(muspy.Track(program=program))
    #         music.tracks[instr_id].notes.append(
    #             muspy.Note(time=start, pitch=int(pitch), duration=int(duration), velocity=int(velocity))
    #         )
    #         time += duration
    #     return music

    def decode(self, encoded_dict):
        import muspy
        import numpy as np

        music = muspy.Music()
        note_level = encoded_dict["note_level"]

        if len(note_level) == 0:
            return music

        max_dur = max(note_level[:, 1])
        max_log = np.log1p(max_dur)  # approximate max log from encoding

        # Keep track of current time for each instrument
        num_instruments = max(note_level[:, 3]) + 1
        instr_times = [0] * num_instruments

        for note in note_level:
            pitch, quantized_duration, velocity, instr_id = note
            duration = int(np.expm1(quantized_duration / 127 * max_log))
            program = self.reverse_instrument_map.get(instr_id, 0)

            # Determine start time for this note
            start = instr_times[instr_id]

            # Ensure track exists
            while len(music.tracks) <= instr_id:
                music.tracks.append(muspy.Track(program=program))

            # Append note
            music.tracks[instr_id].notes.append(
                muspy.Note(
                    time=start,
                    pitch=int(pitch),
                    duration=int(duration),
                    velocity=int(velocity)
                )
            )

            # Move forward only this instrument's time
            instr_times[instr_id] = start + duration

        return music
