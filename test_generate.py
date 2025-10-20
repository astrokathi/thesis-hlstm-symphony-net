import muspy

from h_event_processor import HEventProcessor
from model import HEventModel
from generator import MusicGenerator
from generator_v2 import MusicGeneratorV2
import torch
import numpy as np

np.random.seed(42)

# Load model & processor
processor = HEventProcessor()

song = muspy.read_midi("test/interstellar.mid")
d = processor.encode(song, 1)

model = HEventModel()
model.load_state_dict(torch.load("models/hlstm_epoch_4.pt", map_location="mps"))

generator = MusicGenerator(model, processor, device="mps")

gen_v2 = MusicGeneratorV2(model, processor, device="mps")

# Example prompt
prompt_tokens = d['note_level']
print(list(set(prompt_tokens[:, 3])))

LIST_OF_INSTRUMENTS = [96, 99, 103]
print(LIST_OF_INSTRUMENTS)

# midi, tokens = generator.generate_force_polyphonic(prompt_tokens, style=0, num_steps=100,
#                                                    allowed_instruments=LIST_OF_INSTRUMENTS,
#                                                    temperature=0.7, notes_per_chord=4, include_initial=False, seq_len=256)
# print(tokens)

# midi, tokens = generator.generate_natural(prompt_tokens, style=0, num_steps=1000,
#                                           allowed_instruments=LIST_OF_INSTRUMENTS,
#                                           temperature=1.5, seq_len=len(prompt_tokens), include_initial=False)
# print(tokens)

expressive_controls = np.array([0.8, 0.7, 0.9, 0.3])  # [modulation, volume, expression, sustain]
midi, tokens = generator.generate_with_preset(
    preset_name="dreamy",
    num_steps=100,
    prompt_tokens=prompt_tokens,
    style=0,
    control_context=expressive_controls,
    allowed_instruments=LIST_OF_INSTRUMENTS,
    notes_per_chord=3,
    temperature=0.5,
    include_initial=False,
    top_k=10,
    top_p=0.9
)

muspy.write_midi("data/gen/interstellar_trio2_dreamy.mid", midi)
print("✅ MIDI saved as generated_song.mid")

# TODO, INCREASE THE DURATION SUM, INTRODUCE DELAY AND ADVANCE THE GENERATION, IT CREATES PANNING EFFECTS
