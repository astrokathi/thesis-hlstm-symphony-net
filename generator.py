from typing import List
import torch
import numpy as np
from h_event_processor import HEventProcessor
from model import HEventModel
import muspy


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
            allowed_instruments: List[int] = None  # List of allowed instrument IDs
    ):
        """
        Generate a sequence of music tokens from a prompt.
        Sprint 1: Pre-built tensor buffers + circular context buffer for speed.
        """
        if allowed_instruments is None:
            allowed_instruments = list(range(128))

        if include_initial:
            generated = list(prompt_tokens)
        else:
            generated = []

        seq_len = 50
        hidden_states = None

        # --- Sprint 1: Pre-allocate style/instr tensors (reused every step) ---
        style_tensor = torch.tensor([style], dtype=torch.long, device=self.device)
        instrument_tensor = torch.tensor([allowed_instruments], dtype=torch.long, device=self.device)

        # Pre-build instrument mask (reused every step)
        instr_mask = None
        if allowed_instruments:
            # Shape will match instr_logits at runtime; build a template and slice
            instr_mask = torch.full((1, 1, Config.NUM_INSTRUMENTS), -1e9, dtype=torch.float, device=self.device)
            instr_mask[:, :, allowed_instruments] = 0.0

        # --- Sprint 1: Circular context buffer (avoids numpy→list→tensor round-trips) ---
        if len(prompt_tokens) > 0:
            init_ctx = prompt_tokens[-seq_len:].copy()
        else:
            init_ctx = np.zeros((1, 4), dtype=np.int64)

        # Pad or truncate to seq_len
        if len(init_ctx) < seq_len:
            pad = np.zeros((seq_len - len(init_ctx), 4), dtype=np.int64)
            ctx_buf = np.concatenate([pad, init_ctx])
        else:
            ctx_buf = init_ctx[-seq_len:]

        x = torch.from_numpy(ctx_buf.astype(np.int64)).unsqueeze(0).to(self.device)

        print(f"[GENERATOR INFO]")
        print(f"  Allowed instruments: {allowed_instruments}")
        print(f"  Style: {style}")
        print("=" * 60)

        # Helper to sample with top-k
        def sample_topk(logits):
            top_logits, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            probs = torch.softmax(top_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            return top_idx[0, sampled_idx].item()

        for step in range(num_steps):
            pitch_logits, dur_logits, vel_logits, instr_logits, hidden_states = self.model(
                x, style=style_tensor, instr_context=instrument_tensor, hidden_states=hidden_states
            )

            # Apply instrument masking (reuse pre-built mask template)
            if instr_mask is not None:
                mask = instr_mask.expand(-1, instr_logits.size(1), -1)
                instr_logits = instr_logits + mask

            # Take only last token prediction
            pitch_logits = pitch_logits[:, -1, :] / temperature
            dur_logits = dur_logits[:, -1, :] / temperature
            vel_logits = vel_logits[:, -1, :] / temperature
            instr_logits = instr_logits[:, -1, :] / temperature

            pitch = sample_topk(pitch_logits)
            duration = sample_topk(dur_logits)
            velocity = sample_topk(vel_logits)
            instrument = sample_topk(instr_logits)

            next_note = np.array([pitch, duration, velocity, instrument], dtype=np.int16)
            generated.append(next_note)

            # --- Sprint 1: Circular buffer update (in-place on GPU tensor) ---
            # Shift left by 1 and assign new note at end
            x = torch.roll(x, shifts=-1, dims=1)
            x[0, -1, 0] = pitch
            x[0, -1, 1] = duration
            x[0, -1, 2] = velocity
            x[0, -1, 3] = instrument

            if step % 20 == 0:
                print(f"Step {step}: pitch={pitch}, dur={duration}, vel={velocity}, instr={instrument}")

        generated_tokens = np.array(generated, dtype=np.int16)
        print(f"Generated {len(generated_tokens)} notes")

        midi_music = self.processor.decode({
            "note_level": generated_tokens,
            "instr_level": np.unique(generated_tokens[:, 3]),
            "song_level": [120, 4, len(np.unique(generated_tokens[:, 3])), generated_tokens.shape[0], style]
        })

        return midi_music, generated_tokens

    @torch.no_grad()
    def generate_natural(
            self,
            prompt_tokens: np.ndarray,
            style: int = 0,
            num_steps: int = 200,
            temperature: float = 0.8,
            top_k: int = 40,
            allowed_instruments: List[int] = None,
            include_initial=False,
            seq_len=50
    ):
        """
        Let the model generate naturally - it will create polyphony if it learned it.
        Sprint 1: Pre-built tensors + circular buffer.
        """
        if allowed_instruments is None:
            allowed_instruments = list(range(128))

        if include_initial:
            generated = list(prompt_tokens)
        else:
            generated = list()
        hidden_states = None

        # --- Sprint 1: Pre-allocate tensors ---
        style_tensor = torch.tensor([style], dtype=torch.long, device=self.device)
        instrument_tensor = torch.tensor([allowed_instruments], dtype=torch.long, device=self.device)
        instr_mask = None
        if allowed_instruments:
            instr_mask = torch.full((1, 1, Config.NUM_INSTRUMENTS), -1e9, dtype=torch.float, device=self.device)
            instr_mask[:, :, allowed_instruments] = 0.0

        # --- Sprint 1: Circular buffer ---
        if len(prompt_tokens) > 0:
            init_ctx = prompt_tokens[-seq_len:].copy()
        else:
            init_ctx = np.zeros((1, 4), dtype=np.int64)
        if len(init_ctx) < seq_len:
            pad = np.zeros((seq_len - len(init_ctx), 4), dtype=np.int64)
            ctx_buf = np.concatenate([pad, init_ctx])
        else:
            ctx_buf = init_ctx[-seq_len:]
        x = torch.from_numpy(ctx_buf.astype(np.int64)).unsqueeze(0).to(self.device)

        print(f"Generating naturally with instruments: {allowed_instruments}")

        def sample_single(logits):
            top_logits, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            probs = torch.softmax(top_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            return top_idx[0, sampled_idx].item()

        for step in range(num_steps):
            pitch_logits, dur_logits, vel_logits, instr_logits, hidden_states = self.model(
                x, style=style_tensor, instr_context=instrument_tensor, hidden_states=hidden_states
            )

            # Apply instrument masking (reuse pre-built mask)
            if instr_mask is not None:
                mask = instr_mask.expand(-1, instr_logits.size(1), -1)
                instr_logits = instr_logits + mask

            # Take only last token prediction
            pitch_logits = pitch_logits[:, -1, :] / temperature
            dur_logits = dur_logits[:, -1, :] / temperature
            vel_logits = vel_logits[:, -1, :] / temperature
            instr_logits = instr_logits[:, -1, :] / temperature

            pitch = sample_single(pitch_logits)
            duration = sample_single(dur_logits)
            velocity = sample_single(vel_logits)
            instrument = sample_single(instr_logits)

            next_note = np.array([pitch, duration, velocity, instrument], dtype=np.int16)
            generated.append(next_note)

            # --- Sprint 1: Circular buffer update on GPU ---
            x = torch.roll(x, shifts=-1, dims=1)
            x[0, -1, 0] = pitch
            x[0, -1, 1] = duration
            x[0, -1, 2] = velocity
            x[0, -1, 3] = instrument

            if step % 25 == 0:
                current_instruments = np.unique([note[3] for note in generated[-20:]])
                print(f"Step {step}: pitch={pitch}, instr={instrument}, recent_instruments={list(current_instruments)}")

        generated_tokens = np.array(generated, dtype=np.int16)

        unique_instruments = np.unique(generated_tokens[:, 3])
        instrument_counts = {instr: np.sum(generated_tokens[:, 3] == instr) for instr in unique_instruments}

        print(f"Natural generation completed:")
        print(f"  Total notes: {len(generated_tokens)}")
        print(f"  Instruments used: {instrument_counts}")
        print(f"  Most common instrument: {max(instrument_counts, key=instrument_counts.get)}")

        return self.processor.decode({
            "note_level": generated_tokens,
            "instr_level": np.unique(generated_tokens[:, 3]),
            "song_level": [120, 4, len(unique_instruments), generated_tokens.shape[0], style]
        }), generated_tokens

    @torch.no_grad()
    def generate_force_polyphonic(
            self,
            prompt_tokens: np.ndarray,
            style: int = 0,
            num_steps: int = 200,
            temperature: float = 0.7,
            allowed_instruments: List[int] = None,
            notes_per_chord: int = 3,
            include_initial=False,
            seq_len: int = 64,
            top_k: int = 30,
            top_p: float = 0.9
    ):
        """
        Sprint 1/2: Pre-built tensors, circular buffer, batched instrument sampling.
        """
        if allowed_instruments is None:
            allowed_instruments = list(range(128))

        if include_initial:
            generated = list(prompt_tokens)
        else:
            generated = list()

        hidden_states = None

        # --- Sprint 1: Pre-allocate tensors ---
        style_tensor = torch.tensor([style], dtype=torch.long, device=self.device)
        instrument_tensor = torch.tensor([allowed_instruments], dtype=torch.long, device=self.device)
        instr_mask = None
        if allowed_instruments:
            instr_mask = torch.full((1, Config.NUM_INSTRUMENTS), -1e9, dtype=torch.float, device=self.device)
            instr_mask[:, allowed_instruments] = 0.0

        # --- Sprint 1: Circular buffer ---
        if len(prompt_tokens) > 0:
            init_ctx = prompt_tokens[-seq_len:].copy()
        else:
            init_ctx = np.zeros((1, 4), dtype=np.int64)
        if len(init_ctx) < seq_len:
            pad = np.zeros((seq_len - len(init_ctx), 4), dtype=np.int64)
            ctx_buf = np.concatenate([pad, init_ctx])
        else:
            ctx_buf = init_ctx[-seq_len:]
        x = torch.from_numpy(ctx_buf.astype(np.int64)).unsqueeze(0).to(self.device)

        def sample_single(logits):
            logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            return sampled_idx.item()

        print(f"[GENERATOR CONFIG]")
        print(f"  Sequence length: {seq_len}")
        print(f"  Top-k: {top_k if top_k > 0 else 'disabled'}")
        print(f"  Top-p: {top_p if top_p < 1.0 else 'disabled'}")
        print(f"  Temperature: {temperature}")
        print(f"  Notes per chord: {notes_per_chord}")
        print(f"  Allowed instruments: {allowed_instruments}")
        print("=" * 50)

        for step in range(num_steps):
            pitch_logits, dur_logits, vel_logits, instr_logits, hidden_states = self.model(
                x, style=style_tensor, instr_context=instrument_tensor, hidden_states=hidden_states
            )

            pitch_logits = pitch_logits[:, -1, :]
            dur_logits = dur_logits[:, -1, :]
            vel_logits = vel_logits[:, -1, :]
            instr_logits = instr_logits[:, -1, :]

            # Apply instrument masking (reuse pre-built mask)
            if instr_mask is not None:
                instr_logits = instr_logits + instr_mask

            # --- Sprint 2: Batch instrument sampling (single multinomial call) ---
            instr_probs = torch.softmax(instr_logits / temperature, dim=-1)
            sampled_instrs = torch.multinomial(instr_probs.squeeze(0), num_samples=min(notes_per_chord * 2, instr_logits.size(-1)), replacement=False)
            selected_instruments = []
            for idx in sampled_instrs.tolist():
                if idx in allowed_instruments and idx not in selected_instruments:
                    selected_instruments.append(idx)
                if len(selected_instruments) >= notes_per_chord:
                    break
            while len(selected_instruments) < notes_per_chord:
                remaining = [i for i in allowed_instruments if i not in selected_instruments]
                if remaining:
                    selected_instruments.append(remaining[0])
                else:
                    break

            chord_notes = []
            for instrument in selected_instruments:
                pitch = sample_single(pitch_logits)
                duration = sample_single(dur_logits)
                velocity = sample_single(vel_logits)
                next_note = np.array([pitch, duration, velocity, instrument], dtype=np.int16)
                chord_notes.append(next_note)
                generated.append(next_note)

            # Circular buffer update
            for note in chord_notes:
                x = torch.roll(x, shifts=-1, dims=1)
                x[0, -1, 0] = note[0]
                x[0, -1, 1] = note[1]
                x[0, -1, 2] = note[2]
                x[0, -1, 3] = note[3]

            if step % 10 == 0:
                pitches = [note[0] for note in chord_notes]
                print(f"Step {step}: Chord with {len(selected_instruments)} instruments - "
                      f"Instruments: {selected_instruments}, Pitches: {pitches}")

        generated_tokens = np.array(generated, dtype=np.int16)

        # Analyze results
        unique_instruments = np.unique(generated_tokens[:, 3])
        instrument_counts = {instr: np.sum(generated_tokens[:, 3] == instr) for instr in unique_instruments}

        print(f"\n[GENERATION COMPLETE]")
        print(f"  Total notes: {len(generated_tokens)}")
        print(f"  Instruments used: {len(unique_instruments)}")
        print(f"  Instrument distribution: {instrument_counts}")
        print(f"  Sequence length used: {seq_len}")

        return self.processor.decode({
            "note_level": generated_tokens,
            "instr_level": np.unique(generated_tokens[:, 3]),
            "song_level": [120, 4, len(unique_instruments), generated_tokens.shape[0], style]
        }), generated_tokens

    @torch.no_grad()
    def generate_with_control_conditioning(
            self,
            prompt_tokens: np.ndarray,
            style: int = 0,
            num_steps: int = 200,
            temperature: float = 0.5,
            allowed_instruments: List[int] = None,
            notes_per_chord: int = 3,
            include_initial=False,
            seq_len: int = 50,
            top_k: int = 0,
            top_p: float = 1.0,
            control_context: np.ndarray = None
    ):
        """
        Generate music with control value conditioning.
        Sprint 1/2: Pre-built tensors, circular buffer, batched sampling.
        """
        if allowed_instruments is None:
            allowed_instruments = list(range(128))

        if include_initial:
            generated = list(prompt_tokens)
        else:
            generated = list()

        hidden_states = None

        # --- Sprint 1: Pre-allocate tensors ---
        style_tensor = torch.tensor([style], dtype=torch.long, device=self.device)
        instrument_tensor = torch.tensor([allowed_instruments], dtype=torch.long, device=self.device)
        instr_mask = None
        if allowed_instruments:
            instr_mask = torch.full((1, 1, Config.NUM_INSTRUMENTS), -1e9, dtype=torch.float, device=self.device)
            instr_mask[:, :, allowed_instruments] = 0.0

        # --- Prepare control context (one-time) ---
        if control_context is not None:
            if isinstance(control_context, list):
                control_context = np.array(control_context, dtype=np.float32)
            control_vector = np.zeros(Config.CONTROL_DIM, dtype=np.float32)
            for i in range(min(len(control_context), 4)):
                control_vector[i] = control_context[i]
        else:
            control_vector = np.zeros(Config.CONTROL_DIM, dtype=np.float32)
            control_vector[1] = 0.7  # volume
            control_vector[2] = 0.7  # expression
        control_tensor = torch.from_numpy(control_vector).float().to(self.device)
        control_tensor = control_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D)

        # --- Sprint 1: Circular buffer ---
        if len(prompt_tokens) > 0:
            init_ctx = prompt_tokens[-seq_len:].copy()
        else:
            init_ctx = np.zeros((1, 4), dtype=np.int64)
        if len(init_ctx) < seq_len:
            pad = np.zeros((seq_len - len(init_ctx), 4), dtype=np.int64)
            ctx_buf = np.concatenate([pad, init_ctx])
        else:
            ctx_buf = init_ctx[-seq_len:]
        x = torch.from_numpy(ctx_buf.astype(np.int64)).unsqueeze(0).to(self.device)

        def sample_single(logits):
            if logits.dim() > 2:
                logits = logits.squeeze(0)
            logits = logits / temperature
            if top_k > 0:
                if logits.dim() == 2:
                    top_logits, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                    filtered_logits = torch.full_like(logits, -float('Inf'))
                    filtered_logits.scatter_(-1, top_idx, top_logits)
                    logits = filtered_logits
                else:
                    top_logits, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    filtered_logits = torch.full_like(logits, -float('Inf'))
                    filtered_logits[top_idx] = top_logits
                    logits = filtered_logits
            if top_p < 1.0:
                if logits.dim() == 2:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                else:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            if probs.dim() == 2:
                probs = probs.squeeze(0)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            return sampled_idx.item()

        print(f"[GENERATOR CONFIG]")
        if control_context is not None:
            print(f"  Control conditioning: {control_context.tolist()}")
        else:
            print(f"  Control conditioning: default")
        print(f"  Style: {style}")
        print(f"  Allowed instruments: {allowed_instruments}")
        print(f"  Notes per chord: {notes_per_chord}")
        print(f"  Temperature: {temperature}")
        print(f"  Top-k: {top_k if top_k > 0 else 'disabled'}")
        print(f"  Top-p: {top_p if top_p < 1.0 else 'disabled'}")
        print("=" * 50)

        for step in range(num_steps):
            # Expand control_tensor to match current seq_len
            cur_control = control_tensor.expand(1, x.size(1), -1)

            pitch_logits, dur_logits, vel_logits, instr_logits, hidden_states = self.model(
                x, style=style_tensor, instr_context=instrument_tensor,
                control_context=cur_control, hidden_states=hidden_states
            )

            # Apply instrument masking (reuse pre-built mask)
            if instr_mask is not None:
                mask = instr_mask.expand(-1, instr_logits.size(1), -1)
                instr_logits = instr_logits + mask

            pitch_logits_last = pitch_logits[:, -1, :]
            dur_logits_last = dur_logits[:, -1, :]
            vel_logits_last = vel_logits[:, -1, :]
            instr_logits_last = instr_logits[:, -1, :]

            # --- Sprint 2: Batch instrument sampling ---
            instr_probs = torch.softmax(instr_logits_last / temperature, dim=-1)
            sampled_instrs = torch.multinomial(
                instr_probs.squeeze(0), num_samples=min(notes_per_chord * 2, instr_logits_last.size(-1)),
                replacement=False
            )
            selected_instruments = []
            for idx in sampled_instrs.tolist():
                if idx in allowed_instruments and idx not in selected_instruments:
                    selected_instruments.append(idx)
                if len(selected_instruments) >= notes_per_chord:
                    break
            while len(selected_instruments) < notes_per_chord:
                remaining = [i for i in allowed_instruments if i not in selected_instruments]
                if remaining:
                    selected_instruments.append(remaining[0])
                else:
                    break

            chord_notes = []
            for instrument in selected_instruments:
                pitch = sample_single(pitch_logits_last)
                duration = sample_single(dur_logits_last)
                velocity = sample_single(vel_logits_last)
                next_note = np.array([pitch, duration, velocity, instrument], dtype=np.int16)
                chord_notes.append(next_note)
                generated.append(next_note)

            # Circular buffer update
            for note in chord_notes:
                x = torch.roll(x, shifts=-1, dims=1)
                x[0, -1, 0] = note[0]
                x[0, -1, 1] = note[1]
                x[0, -1, 2] = note[2]
                x[0, -1, 3] = note[3]

            if step % 10 == 0:
                pitches = [note[0] for note in chord_notes]
                print(f"Step {step}: Instruments {selected_instruments}, Pitches {pitches}")

        generated_tokens = np.array(generated, dtype=np.int16)

        # Analyze the generated output
        unique_instruments = np.unique(generated_tokens[:, 3])
        instrument_counts = {instr: np.sum(generated_tokens[:, 3] == instr) for instr in unique_instruments}

        print(f"\n[GENERATION COMPLETE]")
        print(f"  Total notes generated: {len(generated_tokens)}")
        print(f"  Instruments used: {len(unique_instruments)}")
        print(f"  Instrument distribution: {instrument_counts}")

        # Add control changes to the output based on conditioning
        midi_music = self._decode_with_controls(generated_tokens, style, control_context)

        return midi_music, generated_tokens

    def _decode_with_controls(self, generated_tokens, style, control_context):
        """Decode tokens to MIDI with added control changes based on conditioning"""
        # First decode normally using the processor
        midi_music = self.processor.decode({
            "note_level": generated_tokens,
            "instr_level": np.unique(generated_tokens[:, 3]),
            "song_level": [120, 4, len(np.unique(generated_tokens[:, 3])), generated_tokens.shape[0], style]
        })

        # Add control changes based on the conditioning
        if control_context is not None:
            self._add_conditioned_controls(midi_music, control_context)

        return midi_music

    def _add_conditioned_controls(self, music, control_context):
        """Add control changes based on the conditioning values"""
        control_mapping = {
            0: 1,  # modulation
            1: 7,  # volume
            2: 11,  # expression
            3: 64,  # sustain
        }

        # Add control changes to all tracks
        for track in music.tracks:
            # Clear any existing control changes to avoid conflicts
            if hasattr(track, 'control_changes'):
                track.control_changes.clear()

            # Add control changes based on conditioning values
            for control_idx, control_value in enumerate(control_context):
                if control_idx in control_mapping and control_value > 0:
                    control_number = control_mapping[control_idx]
                    midi_value = int(control_value * 127)

                    if not hasattr(track, 'control_changes'):
                        track.control_changes = []
                    # Add control change at the beginning
                    track.control_changes.append({
                        'time': 0,
                        'number': control_number,
                        'value': midi_value
                    })

                    # Optionally add periodic control changes for more dynamic expression
                    if control_number in [1, 11]:  # For modulation and expression
                        for time_point in range(50, 200, 50):
                            # Add slight variations to make it more natural
                            varied_value = max(0, min(127, midi_value + np.random.randint(-10, 11)))
                            track.control_changes.append({
                                'time': time_point,
                                'number': control_number,
                                'value': varied_value
                            })

    # Add control presets for easy use
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

    def generate_with_preset(self, preset_name: str, **kwargs):
        """Generate using a predefined control preset"""
        if preset_name not in self.CONTROL_PRESETS:
            available_presets = list(self.CONTROL_PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available_presets}")

        control_context = self.CONTROL_PRESETS[preset_name]
        print(f"Using preset: {preset_name} - {control_context}")

        kwargs["control_context"] = control_context

        return self.generate_with_control_conditioning(
            **kwargs
        )

    def method_1(self, **kwargs):
        return self.generate(**kwargs)

    def method_2(self, **kwargs):
        return self.generate_natural(**kwargs)

    def method_3(self, **kwargs):
        return self.generate_force_polyphonic(**kwargs)

    def method_4(self, **kwargs):
        return self.generate_with_preset(**kwargs)

    # Add this method to your MusicGenerator class
    def list_control_presets(self):
        """List all available control presets"""
        print("Available control presets:")
        for preset_name, values in self.CONTROL_PRESETS.items():
            print(f"  {preset_name}: {values}")
