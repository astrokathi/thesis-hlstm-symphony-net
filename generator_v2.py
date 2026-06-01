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
        Sprint 1: Pre-built tensors, circular buffer for context.
        """
        window_size = window_size or self.window_size

        prompt_tokens = np.array(prompt_tokens, dtype=np.int64)
        prompt_len = len(prompt_tokens)

        generated = [tuple(tok.tolist()) for tok in prompt_tokens] if include_prompt_in_output else []

        if prompt_len == 0:
            seed = np.array([[60, 8, 80, base_instr], [64, 8, 80, base_instr], [67, 8, 80, base_instr]], dtype=np.int64)
            prompt_tokens = seed
            prompt_len = len(prompt_tokens)
            if include_prompt_in_output:
                generated = [tuple(tok.tolist()) for tok in prompt_tokens]

        # --- Sprint 1: Pre-allocate tensors once ---
        prompt_tensor = torch.from_numpy(prompt_tokens.astype(np.int64)).unsqueeze(0).to(self.device)
        style_tensor = torch.tensor([style], dtype=torch.long, device=self.device)

        if instr_context is None:
            instr_context_tensor = None
        else:
            instr_context_tensor = torch.tensor(instr_context, dtype=torch.long, device=self.device)
            if instr_context_tensor.dim() == 1:
                instr_context_tensor = instr_context_tensor.unsqueeze(0)

        instruments_lst = instruments_lst or []

        # Pre-build instrument mask
        instr_mask = None
        if instruments_lst:
            instr_mask = torch.full((Config.NUM_INSTRUMENTS,), -float("Inf"), dtype=torch.float, device=self.device)
            instr_mask[instruments_lst] = 0.0

        # Control tensor (one-time)
        control_tensor = None
        if control_context is not None:
            control_np = np.array(control_context, dtype=np.float32)
            if control_np.ndim == 1:
                control_np = control_np[np.newaxis, :]
            control_tensor = torch.from_numpy(control_np).to(self.device)

        # --- Run the prompt through the model to get initial hidden_states ---
        _, _, _, _, hidden_states = self.model(prompt_tensor, style=style_tensor, instr_context=instr_context_tensor)

        # --- Sprint 1: Circular buffer on GPU ---
        context_list = list(prompt_tokens.tolist())
        if len(context_list) > window_size:
            context_list = context_list[-window_size:]

        # Build GPU circular buffer
        ctx_buf = np.array(context_list[-window_size:], dtype=np.int64)
        if len(ctx_buf) < window_size:
            pad = np.zeros((window_size - len(ctx_buf), 4), dtype=np.int64)
            ctx_buf = np.concatenate([pad, ctx_buf])
        x = torch.from_numpy(ctx_buf).unsqueeze(0).to(self.device)

        # Keep track of recent durations/pitches for heuristic biasing
        recent_durations = [int(t[1]) for t in context_list[-16:]] if len(context_list) > 0 else []
        recent_pitches = [int(t[0]) for t in context_list[-16:]] if len(context_list) > 0 else []

        # --- Generation loop ---
        for step in range(num_steps):
            # Control context expansion (on the fly, but no numpy round-trip)
            control_ctx_to_pass = None
            if control_tensor is not None:
                if control_tensor.dim() == 2:
                    control_ctx_to_pass = control_tensor.unsqueeze(1).expand(-1, x.size(1), -1)
                else:
                    control_ctx_to_pass = control_tensor[..., :x.size(1), :]

            pitch_logits_all, dur_logits_all, vel_logits_all, instr_logits_all, hidden_states = self.model(
                x, style=style_tensor, instr_context=instr_context_tensor,
                hidden_states=hidden_states
            )

            pitch_logits = pitch_logits_all[:, -1, :].squeeze(0)
            dur_logits = dur_logits_all[:, -1, :].squeeze(0)
            vel_logits = vel_logits_all[:, -1, :].squeeze(0)
            instr_logits = instr_logits_all[:, -1, :].squeeze(0)

            # Apply instrument masking (reuse pre-built mask)
            if instr_mask is not None:
                instr_logits = instr_logits + instr_mask

            # Musical constraints
            if len(recent_pitches) > 0:
                prev_pitch = int(recent_pitches[-1])
                pitch_logits = self.limit_leap(prev_pitch, pitch_logits, max_leap=max_leap)

            if motif_length > 0 and len(recent_pitches) >= motif_length:
                motif = recent_pitches[-motif_length:]
                for i in range(len(recent_pitches) - motif_length):
                    if recent_pitches[i:i + motif_length] == motif:
                        expected_next = motif[0]
                        if 0 <= expected_next < pitch_logits.size(0):
                            pitch_logits[expected_next] = pitch_logits[expected_next] * motif_repeat_boost
                        break

            if recent_durations:
                duration_bias = torch.zeros_like(dur_logits)
                for d in recent_durations[-8:]:
                    if 0 <= d < dur_logits.size(0):
                        duration_bias[d] += 1.0
                if duration_bias.sum() > 0:
                    duration_bias = duration_bias / (duration_bias.sum() + 1e-9)
                    dur_logits = dur_logits + torch.log1p(duration_bias * 5.0)

            # Sampling
            pitch_idx = self.sample_from_logits(pitch_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            dur_idx = self.sample_from_logits(dur_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            vel_idx = self.sample_from_logits(vel_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            instr_idx = self.sample_from_logits(instr_logits, temperature=temperature, top_k=top_k, top_p=top_p)

            if instruments_lst:
                final_instr = self.map_instr_to_list(instr_idx, instruments_lst, base_instr=base_instr)
            else:
                final_instr = instr_idx

            # --- Sprint 1: Circular buffer update (GPU in-place, no numpy round-trip) ---
            x = torch.roll(x, shifts=-1, dims=1)
            x[0, -1, 0] = pitch_idx
            x[0, -1, 1] = dur_idx
            x[0, -1, 2] = vel_idx
            x[0, -1, 3] = final_instr

            generated.append(tuple([pitch_idx, dur_idx, vel_idx, final_instr]))

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
