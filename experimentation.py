import muspy

from h_event_processor import HEventProcessor
from model import HEventModel
from generator import MusicGenerator
import torch
import numpy as np
from visualize_metrics import GeneratorMetrics
import matplotlib.pyplot as plt
import wav_composer as wc

from config import Config

np.random.seed(42)

# Load model & processor
processor = HEventProcessor()

INPUT_FILE_PATH = Config.INPUT_FILE

song = muspy.read_midi(INPUT_FILE_PATH)
d = processor.encode(song, 2)

multi = muspy.to_pypianoroll(song)
multi.plot(
    mode="hybrid",  # 'separate' for stacked, 'same' for overlayed
    track_label="program",  # show instrument names
    preset="frame",  # color preset ('frame', 'full', 'plain', etc.)
)

plt.savefig("assets/master_piano_roll.png", dpi=300, bbox_inches='tight')

wc.write_mid_and_wav(
    music=song,
    write_mid=False,
    mid_file_path=INPUT_FILE_PATH,
    wav_file_path=f"assets/master.wav"
)

model = HEventModel()
model.load_state_dict(torch.load("models/hlstm_epoch_9.pt", map_location="mps"))

generator = MusicGenerator(model, processor, device="mps")

# Example prompt
prompt_tokens = d['note_level']

# LIST_OF_INSTRUMENTS = [56, 57, 58, 42, 60, 114]
LIST_OF_INSTRUMENTS = Config.INSTRUMENTS

"""
CONTROL_PRESETS = {
        "expressive": [0.8, 0.7, 0.9, 0.3],  # Lots of modulation and expression
        "gentle": [0.2, 0.6, 0.7, 0.1],  # Soft and delicate
        "bright": [0.4, 0.8, 0.8, 0.2],  # Clear and vibrant
        "sustained": [0.3, 0.7, 0.7, 0.9],  # Long notes with sustain
        "percussive": [0.1, 0.8, 0.6, 0.0],  # Short, punchy notes
        "dreamy": [0.9, 0.5, 0.8, 0.7],  # Lots of modulation, medium sustain
        "neutral": [0.0, 0.7, 0.7, 0.0],  # Default neutral values
        "nothing": [0.0, 0.0, 0.0, 0.0]
    }
"""

PRESET = Config.PRESET
NUM_SAMPLES = Config.NUM_SAMPLES
STYLE = Config.STYLE


def plot_metrics(gen):
    for i in range(1, 5):
        # we are calling all the method starting from method 1

        gm = GeneratorMetrics(
            generator=gen,
            preset_name=PRESET,
            style=STYLE,
            instruments_used=LIST_OF_INSTRUMENTS,
            prompt_tokens=prompt_tokens,
            num_generations=NUM_SAMPLES,
            num_steps=100,
            control_context=[],
            notes_per_chord=3,
            temperature=0.7,
            include_initial=False,
            top_k=50,
            top_p=0.9,
            generation_method=i,
            seq_length=128
        )
        gm.compare_multiple_generations()


plot_metrics(generator)
