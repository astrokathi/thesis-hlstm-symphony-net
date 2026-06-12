"""Generate music from a trained H-LSTM checkpoint.

Usage:
    python execution_steps/generate.py                          # uses default checkpoint
    python execution_steps/generate.py models/hlstm_epoch_5.pt  # custom checkpoint
"""

import sys
import os
from hlstm_framework.data.encoding import HEventProcessor
from hlstm_framework.models import HEventModel
from hlstm_framework.generation import MusicGenerator
from hlstm_framework.config import load_settings
import muspy
import torch

# --- Load checkpoint ---
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "models/hlstm_epoch_2.pt"
if not os.path.exists(ckpt_path):
    print(f"Checkpoint not found: {ckpt_path}")
    print("Train a model first with: python execution_steps/e2e.py")
    sys.exit(1)

device = load_settings().model.device
checkpoint = torch.load(ckpt_path, map_location=device)

# HEventModel() must match the architecture used during training.
# The saved state_dict contains all expected keys.
model = HEventModel()
model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
model.to(device)
print(f"Loaded checkpoint: {ckpt_path} ({sum(p.numel() for p in model.parameters()):,} params)")

# --- Generate ---
processor = HEventProcessor()
gen = MusicGenerator(model, processor)

prompt_path = "test/prompt.mid"
if not os.path.exists(prompt_path):
    print(f"Prompt file not found: {prompt_path}")
    print("Create a prompt MIDI file or update the path.")
    sys.exit(1)

prompt = processor.encode(muspy.read_midi(prompt_path), 0)
print(f"Prompt: {len(prompt['note_level'])} notes")

midi, tokens = gen.generate_with_preset(
    "gentle",
    prompt_tokens=prompt["note_level"],
    style=0,
    num_steps=200,
    temperature=0.7,
    allowed_instruments=[40, 41, 42],
    notes_per_chord=3,
)

os.makedirs("output", exist_ok=True)
out_path = "output/generated.mid"
midi.write_midi(out_path)
print(f"Generated {len(tokens)} notes → {out_path}")
