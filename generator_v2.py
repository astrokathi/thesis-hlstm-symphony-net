# generator_v2.py
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from h_event_processor import HEventProcessor
from model import HEventModel


class MusicGeneratorV2:
    def __init__(
            self,
            model: HEventModel,
            processor: HEventProcessor,
            device: str = "cpu",
            window_size: int = 256,
    ):
        self.device = device
        self.model = model.to(device)
        self.processor = processor
        self.model.eval()
        self.window_size = window_size

    # --- utility samplers ---
    @staticmethod
    def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
        """Filter logits by top_k and/or top_p (nucleus). logits is 1D tensor."""
        logits = logits.clone()
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            min_topk = topk_vals[-1]
            logits[logits < min_topk] = -float("Inf")

        if top_p > 0.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            # mask tokens with cumulative prob > top_p
            sorted_idx_to_remove = cumulative_probs > top_p
            # keep first token that crosses threshold
            sorted_idx_to_remove[..., 0] = False
            indices_to_remove = sorted_idx[sorted_idx_to_remove]
            logits[indices_to_remove] = -float("Inf")

        return logits

    def sample_from_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 0.0,
    ) -> int:
        """Return single sampled index from logits (1D) after filtering."""
        if temperature != 1.0:
            logits = logits / (temperature + 1e-9)
        filtered = self.top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered, dim=-1)
        if torch.all(torch.isinf(filtered)):  # defensive: if everything filtered
            probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()
        return int(idx)

    # --- musical heuristics ---
    @staticmethod
    def limit_leap(prev_pitch: int, pitch_logits: torch.Tensor, max_leap: int = 12):
        """Zero out probabilities for pitches with larger than max_leap interval from prev_pitch."""
        # pitch_logits shape: [vocab]
        v = pitch_logits.clone()
        pitch_indices = torch.arange(v.size(-1), device=v.device)
        mask = (torch.abs(pitch_indices - prev_pitch) > max_leap)
        v[mask] = -float("Inf")
        # if mask removes all, return original logits
        if torch.all(torch.isinf(v)):
            return pitch_logits
        return v

    def map_instr_to_list(self, instr_pred: int, instruments_lst: List[int], base_instr: Optional[int] = None):
        """Map model predicted instrument id to closest in instruments_lst (if provided)."""
        if not instruments_lst:
            return int(instr_pred)
        # pick instrument from instruments_lst by minimum absolute difference
        instr_array = np.array(instruments_lst, dtype=int)
        distances = np.abs(instr_array - instr_pred)
        return int(instr_array[distances.argmin()])

    # --- main generate method ---
    @torch.no_grad()
    def generate(
            self,
            prompt_tokens: np.ndarray,  # (prompt_len, 4)
            style: int = 0,
            instruments_lst: Optional[List[int]] = None,  # e.g. [0, 40, 42]
            instr_context: Optional[List[int]] = None,  # instrument context to pass (list ints)
            control_context: Optional[np.ndarray] = None,  # optional float context [control_dim] or [1,control_dim]
            num_steps: int = 400,
            temperature: float = 1.0,
            top_k: int = 20,
            top_p: float = 0.0,
            window_size: Optional[int] = None,
            max_leap: int = 12,
            motif_repeat_boost: float = 1.25,
            motif_length: int = 3,
            include_prompt_in_output: bool = False,
            base_instr: int = 40,
            instr_range: int = 8,
    ):
        """
        Advanced generator:
          - seq2seq "encode" of prompt: run the prompt through the model once to initialize hidden_states
          - then autoregressively generate num_steps new notes, using sliding window context
          - instruments_lst controls allowed instruments; instr_context passed to model
          - control_context (float) projected by model if supported
        """
        window_size = window_size or self.window_size

        # convert prompt to np array and ensure shape
        prompt_tokens = np.array(prompt_tokens, dtype=np.int64)
        prompt_len = len(prompt_tokens)

        # Build initial generated list
        generated = [tuple(tok.tolist()) for tok in prompt_tokens] if include_prompt_in_output else []
        # We'll feed the model with prompt to get initial hidden states.
        # Use a tensor version of the whole prompt (batch_size=1)
        if prompt_len == 0:
            # If no prompt given, use a small default seed: C major arpeggio
            seed = np.array([[60, 8, 80, base_instr], [64, 8, 80, base_instr], [67, 8, 80, base_instr]], dtype=np.int64)
            prompt_tokens = seed
            prompt_len = len(prompt_tokens)
            if include_prompt_in_output:
                generated = [tuple(tok.tolist()) for tok in prompt_tokens]

        # Prepare tensors once (avoid repeated slow list->tensor conversions)
        # We'll use int64 for embedding indices, but control_context should be float if used.
        prompt_tensor = torch.from_numpy(prompt_tokens.astype(np.int64)).unsqueeze(0).to(
            self.device)  # [1, prompt_len, 4]
        style_tensor = torch.tensor([style], dtype=torch.long, device=self.device)

        # Prepare instr_context tensor: accept list of ints (global context) or tensor
        if instr_context is None:
            instr_context_tensor = None
        else:
            instr_context_tensor = torch.tensor(instr_context, dtype=torch.long, device=self.device)
            if instr_context_tensor.dim() == 1:
                instr_context_tensor = instr_context_tensor.unsqueeze(0)  # [1, num_instrs]

        # Prepare instruments list for mapping
        instruments_lst = instruments_lst or []

        # Prepare control_context if given (float)
        control_tensor = None
        if control_context is not None:
            control_np = np.array(control_context, dtype=np.float32)
            # if shape (control_dim,) -> expand to [1, control_dim]
            if control_np.ndim == 1:
                control_np = control_np[np.newaxis, :]
            control_tensor = torch.from_numpy(control_np).to(self.device)  # float tensor

        # --- Run the prompt through the model to get initial hidden_states ---
        # We feed the full prompt (note-level features) to obtain initial hidden states.
        # The model returns logits and hidden states; we only care about hidden_states to seed generation.
        # We'll discard logits for the prompt (unless you want teacher forcing / seq2seq training)
        _, _, _, _, hidden_states = self.model(prompt_tensor, style=style_tensor, instr_context=instr_context_tensor)

        # Maintain a context buffer (list of tokens) for autoregression
        context = list(prompt_tokens.tolist())
        if len(context) > window_size:
            context = context[-window_size:]

        # Keep track of recent durations/pitches for heuristic biasing
        recent_durations = [int(t[1]) for t in context[-16:]] if len(context) > 0 else []
        recent_pitches = [int(t[0]) for t in context[-16:]] if len(context) > 0 else []

        # --- Generation loop ---
        for step in range(num_steps):
            # Prepare input tensor from context: use last window_size tokens
            ctx_window = np.array(context[-window_size:], dtype=np.int64)
            x = torch.from_numpy(ctx_window).unsqueeze(0).to(self.device)  # [1, seq_len, 4]

            # If model supports control_context and we have a float tensor, expand if needed
            control_ctx_to_pass = None
            if control_tensor is not None:
                # control_tensor shape: [1, control_dim] or [1, seq_len, control_dim]
                if control_tensor.dim() == 2:
                    # expand to seq_len
                    control_ctx_to_pass = control_tensor.unsqueeze(1).expand(-1, x.size(1), -1)
                else:
                    # if already seq_len, maybe mismatch; trim/pad if necessary
                    control_ctx_to_pass = control_tensor[..., :x.size(1), :]

            # Call model once (returns logits for whole context and new hidden states)
            pitch_logits_all, dur_logits_all, vel_logits_all, instr_logits_all, hidden_states = self.model(
                x,
                style=style_tensor,
                instr_context=instr_context_tensor,
                hidden_states=hidden_states
            )

            # pick last time-step logits
            pitch_logits = pitch_logits_all[:, -1, :].squeeze(0)  # shape [pitch_vocab]
            dur_logits = dur_logits_all[:, -1, :].squeeze(0)
            vel_logits = vel_logits_all[:, -1, :].squeeze(0)
            instr_logits = instr_logits_all[:, -1, :].squeeze(0)

            # Apply simple musical constraints & biases:

            # 1) limit large leaps relative to most recent pitch
            if len(recent_pitches) > 0:
                prev_pitch = int(recent_pitches[-1])
                pitch_logits = self.limit_leap(prev_pitch, pitch_logits, max_leap=max_leap)

            # 2) motif repetition boost: if a motif (last motif_length pitches) exists, boost repeating it
            if motif_length > 0 and len(recent_pitches) >= motif_length:
                motif = recent_pitches[-motif_length:]
                # check immediate previous occurrence (naive)
                for i in range(len(recent_pitches) - motif_length):
                    if recent_pitches[i:i + motif_length] == motif:
                        # boost probability of repeating motif by raising logits of the next expected pitch (if within vocab)
                        # here we only boost next pitch predicted equal to motif[0] (very simple)
                        expected_next = motif[0]
                        if 0 <= expected_next < pitch_logits.size(0):
                            pitch_logits[expected_next] = pitch_logits[expected_next] * motif_repeat_boost
                        break

            # 3) bias duration sampling to recent durations distribution (simple: if recent durations exist, slightly favor them)
            if recent_durations:
                duration_bias = torch.zeros_like(dur_logits)
                for d in recent_durations[-8:]:
                    if 0 <= d < dur_logits.size(0):
                        duration_bias[d] += 1.0
                if duration_bias.sum() > 0:
                    duration_bias = duration_bias / (duration_bias.sum() + 1e-9)
                    dur_logits = dur_logits + torch.log1p(duration_bias * 5.0)  # boost probabilities

            # Sampling
            pitch_idx = self.sample_from_logits(pitch_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            dur_idx = self.sample_from_logits(dur_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            vel_idx = self.sample_from_logits(vel_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            instr_idx = self.sample_from_logits(instr_logits, temperature=temperature, top_k=top_k, top_p=top_p)

            # Map instrument to allowed instruments_list (if provided) to keep ensemble coherent
            if instruments_lst:
                final_instr = self.map_instr_to_list(instr_idx, instruments_lst, base_instr=base_instr)
            else:
                final_instr = instr_idx

            # Append new note
            next_note = np.array([pitch_idx, dur_idx, vel_idx, final_instr], dtype=np.int16)
            context.append(next_note.tolist())
            generated.append(tuple(next_note.tolist()))

            # update recent trackers
            recent_pitches.append(int(pitch_idx))
            recent_durations.append(int(dur_idx))
            if len(recent_pitches) > 128:
                recent_pitches = recent_pitches[-128:]
            if len(recent_durations) > 128:
                recent_durations = recent_durations[-128:]

            # debug log occasionally
            if (step + 1) % 50 == 0 or step < 10:
                print(
                    f"[gen] step {step + 1}/{num_steps}  pitch={pitch_idx} dur={dur_idx} vel={vel_idx} instr={final_instr}")

        # Build generated_tokens numpy array
        generated_tokens = np.array(generated, dtype=np.int16)

        # Decode to MIDI using your processor (it expects note-level tokens: pitch,duration,vel,instr)
        midi_music = self.processor.decode({
            "note_level": generated_tokens,
            "instr_level": np.unique(generated_tokens[:, 3]),
            "song_level": [0, 4, len(np.unique(generated_tokens[:, 3])), generated_tokens.shape[0]]
        })

        return midi_music, generated_tokens
