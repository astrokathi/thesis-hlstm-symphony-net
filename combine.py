import pretty_midi


def combine_midi(file1, file2, output_file):
    # Load both MIDI files
    midi1 = pretty_midi.PrettyMIDI(file1)
    midi2 = pretty_midi.PrettyMIDI(file2)

    # Create a new PrettyMIDI object
    combined = pretty_midi.PrettyMIDI()

    # Add instruments from the first MIDI
    for instrument in midi1.instruments:
        combined.instruments.append(instrument)

    # Add instruments from the second MIDI
    for instrument in midi2.instruments:
        combined.instruments.append(instrument)

    # Save the combined MIDI
    combined.write(output_file)


def combine_midi_clip(file1, file2, output_file):
    # Load both MIDI files
    midi1 = pretty_midi.PrettyMIDI(file1)
    midi2 = pretty_midi.PrettyMIDI(file2)

    # Create a new PrettyMIDI object
    combined = pretty_midi.PrettyMIDI()

    # Find the shorter duration
    min_time = min(midi1.get_end_time(), midi2.get_end_time())

    # Clip and add instruments from first MIDI
    for instrument in midi1.instruments:
        new_inst = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)
        for note in instrument.notes:
            if note.start < min_time:  # only include notes within min_time
                clipped_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=min(note.end, min_time)  # trim if it exceeds min_time
                )
                new_inst.notes.append(clipped_note)
        combined.instruments.append(new_inst)

    # Clip and add instruments from second MIDI
    for instrument in midi2.instruments:
        new_inst = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)
        for note in instrument.notes:
            if note.start < min_time:
                clipped_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=min(note.end, min_time)
                )
                new_inst.notes.append(clipped_note)
        combined.instruments.append(new_inst)

    # Save the combined MIDI
    combined.write(output_file)


# Example usage
# combine_midi("data/mid/12bar1.mid", "data/mid/generated_music_grammar6_full_new_drums.mid", "data/mid/combined.mid")
combine_midi_clip("test/interstellar.mid", "generated_song_interstellar_diff_train.mid", "data/gen/combined_clip_match.mid")
