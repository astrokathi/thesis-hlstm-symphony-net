from typing import List

import torch
import numpy as np
from h_event_processor import HEventProcessor
from model import HEventModel


class MusicGenerator:
    def __init__(self, model: HEventModel, processor: HEventProcessor, device="mps"):
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate(
            self,
            prompt_tokens: np.ndarray,  # shape: (seq_len, 4)
            style: int = 0,
            num_steps: int = 200,  # how many notes to generate
            temperature: float = 0.5,
            top_k: int = 20,
            include_initial=False,
            base_instr=40,
            instruments_lst=None,
            instr_range=6
    ):
        """
        Generate a sequence of music tokens from a prompt.
        """
        if instruments_lst is None:
            instruments_lst = list()
        if include_initial:
            generated = list(prompt_tokens)
        else:
            generated = list()  # start with prompt
        # seq_len = len(prompt_tokens)
        seq_len = 50
        hidden_states = None

        # Convert prompt to tensor
        x = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        style_tensor = torch.tensor([style], dtype=torch.long).to(self.device)

        for step in range(num_steps):

            if step == 0:  # Print once
                vocab_size = (
                        self.model.pitch_embed.num_embeddings +
                        self.model.duration_embed.num_embeddings +
                        self.model.velocity_embed.num_embeddings +
                        self.model.instrument_embed.num_embeddings
                )
                print(f"[GENERATOR INFO]")
                print(f"  Pitch: {self.model.pitch_embed.num_embeddings}")
                print(f"  Duration: {self.model.duration_embed.num_embeddings}")
                print(f"  Velocity: {self.model.velocity_embed.num_embeddings}")
                print(f"  Instrument: {self.model.instrument_embed.num_embeddings}")
                print(f"  --> Effective combined vocab size: {vocab_size:,}")
                print("=" * 60)

            pitch_logits, dur_logits, vel_logits, instr_logits, hidden_states = self.model(
                x, style=style_tensor, instr_context=instruments_lst,  hidden_states=hidden_states
            )

            # Take only last token prediction
            pitch_logits = pitch_logits[:, -1, :] / temperature
            dur_logits = dur_logits[:, -1, :] / temperature
            vel_logits = vel_logits[:, -1, :] / temperature
            instr_logits = instr_logits[:, -1, :] / temperature

            # Sample each independently using top-k
            def sample_topk(logits):
                top_logits, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                probs = torch.softmax(top_logits, dim=-1)
                sampled = top_idx[0, torch.multinomial(probs, num_samples=1)]
                return sampled.item()

            def map_to_range(num: int) -> int:
                # Clamp number between 0 and 127
                num = max(0, min(127, num))

                # Map cyclically to range 40–47
                return base_instr + (num % instr_range)

            pitch = sample_topk(pitch_logits)
            duration = sample_topk(dur_logits)
            velocity = sample_topk(vel_logits)
            instr = sample_topk(instr_logits)
            # TODO, comment the below line, now generating based on the model that is trained.
            # instr = map_to_range(instr)
            next_note = np.array([pitch, duration, velocity, instr], dtype=np.int16)
            # print(f"The next note is {next_note} for the step {step}")
            generated.append(next_note)

            # Update context
            x = torch.tensor(np.array(generated[-seq_len:], dtype=np.int64)).unsqueeze(0).to(self.device)

            if step % 10 == 0:
                print(f"Step {step}: pitch={pitch}, dur={duration}, vel={velocity}, instr={instr}")

        generated_tokens = np.array(generated, dtype=np.int16)

        # If we make the duration constant
        # generated_tokens[:, 1] = prompt_tokens[:, 1]

        print(f"The generated tokens have a shape of {generated_tokens.shape}")

        # Decode to MIDI
        midi_music = self.processor.decode({
            "note_level": generated_tokens,
            "instr_level": np.unique(generated_tokens[:, 3]),
            "song_level": [0, 4, len(np.unique(generated_tokens[:, 3])), generated_tokens.shape[0]]
        })

        return midi_music, generated_tokens
