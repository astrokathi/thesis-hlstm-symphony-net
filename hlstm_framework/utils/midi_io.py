"""
MIDI file I/O utilities — write, convert to WAV, combine.

Usage:
    >>> from hlstm_framework.utils.midi_io import write_midi, write_wav
    >>> write_midi(music, "output/generated.mid")
    >>> write_wav("output/generated.mid", "output/generated.wav")
"""

from typing import Optional

import muspy
import numpy as np


def write_midi(music: muspy.Music, path: str) -> str:
    """Write a muspy.Music object to a standard MIDI (.mid) file.

    Args:
        music: muspy.Music object to export.
        path: Output file path.

    Returns:
        The path the file was written to.
    """
    music.write_midi(path)
    return path


def write_wav(
    midi_path: str,
    wav_path: str,
    sf2_path: Optional[str] = None,
    sample_rate: int = 44100,
) -> str:
    """Convert a MIDI file to WAV audio using FluidSynth.

    Requires a SoundFont (.sf2) file for realistic instrument synthesis.
    A GM SoundFont can be downloaded from:
    https://member.keymusician.com/Member/FluidR3_GM/

    Args:
        midi_path: Path to input MIDI file.
        wav_path: Path to output WAV file.
        sf2_path: Path to SoundFont (.sf2) file.
        sample_rate: Audio sample rate.

    Returns:
        The path the WAV was written to.
    """
    import pretty_midi
    from scipy.io import wavfile

    midi = pretty_midi.PrettyMIDI(midi_path)
    audio = midi.fluidsynth(fs=sample_rate) if sf2_path is None else midi.fluidsynth(fs=sample_rate, sf2_path=sf2_path)

    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
    wavfile.write(wav_path, sample_rate, audio)
    return wav_path


def combine_midi(
    file1: str,
    file2: str,
    output_file: str,
    clip_to_shortest: bool = False,
) -> str:
    """Combine two MIDI files by merging their tracks.

    Args:
        file1: Path to first MIDI file.
        file2: Path to second MIDI file.
        output_file: Path to output combined MIDI file.
        clip_to_shortest: If True, clip both to the shorter file's duration.

    Returns:
        Path to the combined file.
    """
    import pretty_midi

    midi1 = pretty_midi.PrettyMIDI(file1)
    midi2 = pretty_midi.PrettyMIDI(file2)
    combined = pretty_midi.PrettyMIDI()

    if clip_to_shortest:
        min_time = min(midi1.get_end_time(), midi2.get_end_time())

        for midi in (midi1, midi2):
            for instrument in midi.instruments:
                new_inst = pretty_midi.Instrument(
                    program=instrument.program,
                    is_drum=instrument.is_drum,
                    name=instrument.name,
                )
                for note in instrument.notes:
                    if note.start < min_time:
                        new_inst.notes.append(
                            pretty_midi.Note(
                                velocity=note.velocity,
                                pitch=note.pitch,
                                start=note.start,
                                end=min(note.end, min_time),
                            )
                        )
                combined.instruments.append(new_inst)
    else:
        for midi in (midi1, midi2):
            for instrument in midi.instruments:
                combined.instruments.append(instrument)

    combined.write(output_file)
    return output_file
