import muspy

from h_event_processor import HEventProcessor
from model import HEventModel
from generator import MusicGenerator
from generator_v2 import MusicGeneratorV2
import torch
import numpy as np
from metrics import MusicGenerationMetrics
from visualize_metrics import GeneratorMetrics

np.random.seed(42)

# Load model & processor
processor = HEventProcessor()

song = muspy.read_midi("test/pirates.mid")
d = processor.encode(song, 2)

model = HEventModel()
model.load_state_dict(torch.load("models/hlstm_epoch_9.pt", map_location="mps"))

generator = MusicGenerator(model, processor, device="mps")

# Example prompt
prompt_tokens = d['note_level']
print(list(set(prompt_tokens[:, 3])))

# LIST_OF_INSTRUMENTS = [56, 57, 58, 42, 60, 114]
# LIST_OF_INSTRUMENTS = [114, 115]
LIST_OF_INSTRUMENTS = [52, 53, 54]
print(LIST_OF_INSTRUMENTS)
PRESET = "neutral"
NUM_SAMPLES = 5
STYLE = 0
FILE_NAME = "new_gen/mop_gen_mul_v5.mid"

# midi, tokens = generator.generate_force_polyphonic(prompt_tokens, style=0, num_steps=100,
#                                                    allowed_instruments=LIST_OF_INSTRUMENTS,
#                                                    temperature=0.7, notes_per_chord=4, include_initial=False, seq_len=256)`
# print(tokens)

# midi, tokens = generator.generate_natural(prompt_tokens, style=0, num_steps=1000,
#                                           allowed_instruments=LIST_OF_INSTRUMENTS,
#                                           temperature=1.5, seq_len=len(prompt_tokens), include_initial=False)
# print(tokens)

# expressive_controls = np.array([0.8, 0.7, 0.9, 0.3])  # [modulation, volume, expression, sustain]
midi, tokens = generator.generate_with_preset(
    preset_name=PRESET,
    num_steps=100,
    prompt_tokens=prompt_tokens,
    style=STYLE,
    control_context=[],
    allowed_instruments=LIST_OF_INSTRUMENTS,
    notes_per_chord=2,
    temperature=0.7,
    include_initial=False,
    top_k=40,
    top_p=0.9
)
#
muspy.write_midi(FILE_NAME, midi)
# print("✅ MIDI saved as generated_song.mid")


# def validate_generation_quality(generator, num_samples=NUM_SAMPLES, preset_name=PRESET):
#     """Validate generation quality across multiple samples"""
#     all_metrics = []
#     metrics_calculator = MusicGenerationMetrics(instruments_used=LIST_OF_INSTRUMENTS, style=STYLE)
#
#     for i in range(num_samples):
#         print(f"Computing metrics for the Sample {i}")
#         # Generate sample
#         midi_music, _ = generator.generate_with_preset(
#             preset_name=preset_name,
#             num_steps=100,
#             prompt_tokens=prompt_tokens,
#             style=STYLE,
#             allowed_instruments=LIST_OF_INSTRUMENTS,
#             notes_per_chord=2,
#             temperature=0.7,
#             include_initial=False,
#             top_k=2,
#             top_p=0.9
#         )
#
#         # Compute metrics
#         metrics = metrics_calculator.compute_all_metrics(midi_music)
#         print(f"Metrics for the sample {i} are {metrics}")
#         all_metrics.append(metrics)
#
#     # Average across samples
#     avg_metrics = {}
#     for key in all_metrics[0].keys():
#         avg_metrics[key] = np.mean([m[key] for m in all_metrics])
#
#     return avg_metrics, all_metrics


def plot_metrics(gen):
    gm = GeneratorMetrics(
        generator=gen,
        preset_name=PRESET,
        style=STYLE,
        instruments_used=LIST_OF_INSTRUMENTS,
        prompt_tokens=prompt_tokens,
        num_generations=3,
        midi_music=midi,
        tokens=tokens
    )
    gm.compare_multiple_generations()


# plot_metrics(generator)
# print(f"Now calculating metrics for {NUM_SAMPLES} samples")
# avg_metrics, _ = validate_generation_quality(generator, num_samples=NUM_SAMPLES)
# print(f"The average metrics among all the samples are {avg_metrics}")

# TODO, INCREASE THE DURATION SUM, INTRODUCE DELAY AND ADVANCE THE GENERATION, IT CREATES PANNING EFFECTS
