import muspy

from h_event_processor import HEventProcessor
from model import HEventModel
from generator import MusicGenerator
import torch
import numpy as np

np.random.seed(42)

# Load model & processor
processor = HEventProcessor()

song = muspy.read_midi("test/interstellar.mid")
d = processor.encode(song)

model = HEventModel()
model.load_state_dict(torch.load("models/hlstm_epoch_3.pt", map_location="mps"))

generator = MusicGenerator(model, processor, device="mps")

# Example prompt
prompt_tokens = d['note_level']
# print(f"Initial prompt tokens {prompt_tokens}")

# Randomly assigning instruments

LIST_OF_INSTRUMENTS = list(d['instr_level'])
print(LIST_OF_INSTRUMENTS)

possible_numbers = np.array(LIST_OF_INSTRUMENTS)

# Randomly choose 4 numbers (one for each row) from the list
# The 'size=4' ensures we get a list of 4 choices
random_choices = np.random.choice(possible_numbers, size=prompt_tokens.shape[0])

# # Assign these 4 random choices to the 4th column (index 3)
# # data_array[:, 3] selects ALL rows (:) and the 4th column (3)
# print(prompt_tokens.shape)
# print(prompt_tokens[:, 3])
# prompt_tokens[:, 3] = random_choices

midi, tokens = generator.generate(prompt_tokens, style=0, num_steps=300, include_initial=False,
                                  instruments_lst=[0,3],
                                  temperature=0.5,
                                  top_k=50)
muspy.write_midi("data/gen/generated_song_interst.mid", midi)
print("✅ MIDI saved as generated_song.mid")

# TODO, INCREASE THE DURATION SUM, INTRODUCE DELAY AND ADVANCE THE GENERATION, IT CREATES PANNING EFFECTS
