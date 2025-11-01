import pretty_midi
import muspy
import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 44100


def write_mid_and_wav(
        music: muspy.Music = None,
        mid_file_path=None,
        wav_file_path=None, write_mid=True):
    if write_mid:
        print(f"Writing MID file {mid_file_path} started")

        music.write_midi(mid_file_path)

        print(f"Writing MID file {mid_file_path} completed")

    sf2_path = "fs2/FluidR3_GM.sf2"
    # The above can be downloaded from https://member.keymusician.com/Member/FluidR3_GM/

    midi = pretty_midi.PrettyMIDI(mid_file_path)

    # Generate audio samples using the SoundFont
    audio = midi.fluidsynth(sf2_path=sf2_path, fs=SAMPLE_RATE) if sf2_path else midi.fluidsynth(fs=SAMPLE_RATE)

    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)

    # 3️⃣ Write audio to WAV
    wavfile.write(wav_file_path, SAMPLE_RATE, audio)

    print(f"Saved as {wav_file_path}")
