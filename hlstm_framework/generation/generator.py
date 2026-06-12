"""
MusicGenerator with 4 generation methods.

All methods use pre-allocated tensor buffers and a GPU circular buffer
for the autoregressive context window (Sprint 1/2 optimizations).

Methods:
    1. Vanilla generation with top-k sampling
    2. Natural generation (higher temperature, relaxed constraints)
    3. Force polyphonic (multiple instruments per chord)
    4. Preset generation (control conditioning with expressive presets)

Usage:
    >>> gen = MusicGenerator(model, processor)
    >>> midi, tokens = gen.generate(prompt, style=0, num_steps=200)
    >>> midi, tokens = gen.generate_natural(prompt, style=0)
    >>> midi, tokens = gen.generate_force_polyphonic(prompt, style=0, notes_per_chord=4)
    >>> midi, tokens = gen.generate_with_preset("expressive", prompt_tokens=prompt, ...)
"""

from typing import List, Optional

import numpy as np
import torch

from hlstm_framework.config import load_settings
from hlstm_framework.data.encoding import HEventProcessor
from hlstm_framework.models.hlstm import HEventModel
from hlstm_framework.generation.sampler import (
    sample_top_k_top_p,
    top_k_top_p_filter,
)
from hlstm_framework.generation.presets import CONTROL_PRESETS


class MusicGenerator:
    """Symbolic music generator using the H-LSTM model.

    Args:
        model: Trained HEventModel instance.
        processor: HEventProcessor for encoding/decoding.
        device: Torch device string.

    Attributes:
        model: The H-LSTM model in eval mode.
        processor: Event encoder/decoder.
        device: Computed device string.
    """

    def __init__(
        self,
        model: HEventModel,
        processor: HEventProcessor,
        device: Optional[str] = None,
    ):
        cfg = load_settings()
        if device is None:
            device = cfg.model.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model = model.to(self.device).eval()
        self.processor = processor

    def __help__(self) -> None:
        """Print usage information for MusicGenerator."""
        print("MusicGenerator — 4-method symbolic music generator")
        print("=" * 50)
        print("Methods:")
        print("  1. generate()                    — Vanilla top-k sampling")
        print("  2. generate_natural()             — Higher temperature, relaxed")
        print("  3. generate_force_polyphonic()    — Multi-instrument chords")
        print("  4. generate_with_preset()         — Expressive control presets")
        print()
        print("Key params: style, temperature, top_k, top_p, num_steps")
        print("            allowed_instruments, notes_per_chord")
        print()
        print("Presets: expressive, gentle, bright, sustained, percussive,")
        print("         dreamy, neutral, nothing")

    # ------------------------------------------------------------------
    # Method 1: Vanilla Generate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: np.ndarray,
        style: int = 0,
        num_steps: int = 200,
        temperature: float = 0.5,
        top_k: int = 20,
        include_initial: bool = False,
        allowed_instruments: Optional[List[int]] = None,
    ):
        """Method 1 — Vanilla generation with top-k sampling and instrument masking.

        Args:
            prompt_tokens: Primer sequence, shape (prompt_len, 4).
            style: Style ID (0=classical, 1=contemporary).
            num_steps: Number of notes to generate.
            temperature: Sampling temperature (<1 = more deterministic).
            top_k: Top-k threshold (0=disabled).
            include_initial: Include prompt tokens in output.
            allowed_instruments: List of allowed MIDI program numbers.

        Returns:
            Tuple of (muspy.Music, np.ndarray of generated tokens).
        """
        return self._generate_base(
            prompt_tokens=prompt_tokens,
            style=style,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            include_initial=include_initial,
            allowed_instruments=allowed_instruments,
        )

    # ------------------------------------------------------------------
    # Method 2: Natural Generate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_natural(
        self,
        prompt_tokens: np.ndarray,
        style: int = 0,
        num_steps: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
        allowed_instruments: Optional[List[int]] = None,
        include_initial: bool = False,
        seq_len: int = 50,
    ):
        """Method 2 — Natural generation with relaxed sampling.

        Higher temperature and top-k encourage more creative variations.
        The model freely chooses instruments from the allowed set.

        Args:
            Same as generate() with additional seq_len parameter.

        Returns:
            Tuple of (muspy.Music, np.ndarray of generated tokens).
        """
        return self._generate_base(
            prompt_tokens=prompt_tokens,
            style=style,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            include_initial=include_initial,
            allowed_instruments=allowed_instruments,
            seq_len=seq_len,
        )

    # ------------------------------------------------------------------
    # Method 3: Force Polyphonic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_force_polyphonic(
        self,
        prompt_tokens: np.ndarray,
        style: int = 0,
        num_steps: int = 200,
        temperature: float = 0.7,
        allowed_instruments: Optional[List[int]] = None,
        notes_per_chord: int = 3,
        include_initial: bool = False,
        seq_len: int = 64,
        top_k: int = 30,
        top_p: float = 0.9,
    ):
        """Method 3 — Force polyphonic generation with batched instrument sampling.

        Generates multiple instruments per chord step for rich textures.
        Uses batched multinomial sampling for efficiency.

        Args:
            prompt_tokens: Primer sequence, shape (prompt_len, 4).
            style: Style ID.
            num_steps: Number of chord steps.
            temperature: Sampling temperature.
            allowed_instruments: List of allowed MIDI program numbers.
            notes_per_chord: Instruments per chord step.
            include_initial: Include prompt tokens in output.
            seq_len: Context window size.
            top_k: Top-k threshold (0=disabled).
            top_p: Nucleus threshold (1.0=disabled).

        Returns:
            Tuple of (muspy.Music, np.ndarray of generated tokens).
        """
        if allowed_instruments is None:
            allowed_instruments = list(range(128))
        generated = list(prompt_tokens) if include_initial else []

        # Pre-allocate tensors
        style_tensor = torch.tensor([style], dtype=torch.long, device=self.device)
        instr_tensor = torch.tensor([allowed_instruments], dtype=torch.long, device=self.device)

        num_instr = load_settings().model.num_instruments
        instr_mask = None
        if allowed_instruments:
            instr_mask = torch.full((1, num_instr), -1e9, dtype=torch.float, device=self.device)
            instr_mask[:, allowed_instruments] = 0.0

        # Circular buffer
        x = self._init_context_buffer(prompt_tokens, seq_len)

        hidden_states = None
        instruments_np = np.array(allowed_instruments)

        for _ in range(num_steps):
            pitch_logits, dur_logits, vel_logits, instr_logits, hidden_states = self.model(
                x, style=style_tensor, instr_context=instr_tensor,
                hidden_states=hidden_states
            )

            pitch_l = pitch_logits[:, -1, :]
            dur_l = dur_logits[:, -1, :]
            vel_l = vel_logits[:, -1, :]
            instr_l = instr_logits[:, -1, :]

            if instr_mask is not None:
                instr_l = instr_l + instr_mask

            # Batch instrument sampling
            instr_probs = torch.softmax(instr_l / temperature, dim=-1)
            sampled = torch.multinomial(
                instr_probs.squeeze(0),
                num_samples=min(notes_per_chord * 2, num_instr),
                replacement=False,
            )
            selected = []
            for idx in sampled.tolist():
                if idx in allowed_instruments and idx not in selected:
                    selected.append(idx)
                if len(selected) >= notes_per_chord:
                    break
            while len(selected) < notes_per_chord:
                remaining = [i for i in allowed_instruments if i not in selected]
                if remaining:
                    selected.append(remaining[0])
                else:
                    break

            # Generate notes for each instrument
            for instr in selected:
                pitch = sample_top_k_top_p(pitch_l.squeeze(0), temperature, top_k, top_p)
                dur = sample_top_k_top_p(dur_l.squeeze(0), temperature, top_k, top_p)
                vel = sample_top_k_top_p(vel_l.squeeze(0), temperature, top_k, top_p)
                note = np.array([pitch, dur, vel, instr], dtype=np.int16)
                generated.append(note)

                # Circular buffer update
                x = torch.roll(x, shifts=-1, dims=1)
                x[0, -1, 0] = pitch
                x[0, -1, 1] = dur
                x[0, -1, 2] = vel
                x[0, -1, 3] = instr

        return self._decode_output(generated, style)

    # ------------------------------------------------------------------
    # Method 4: Preset Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_with_control_conditioning(
        self,
        prompt_tokens: np.ndarray,
        style: int = 0,
        num_steps: int = 200,
        temperature: float = 0.5,
        allowed_instruments: Optional[List[int]] = None,
        notes_per_chord: int = 3,
        include_initial: bool = False,
        seq_len: int = 50,
        top_k: int = 0,
        top_p: float = 1.0,
        control_context: Optional[np.ndarray] = None,
    ):
        """Method 4 base — Generation with control value conditioning.

        Uses a control vector injected at LSTM3 to influence dynamics,
        modulation, and expression.

        Args:
            prompt_tokens: Primer sequence.
            style: Style ID.
            num_steps: Number of chord steps.
            temperature: Sampling temperature.
            allowed_instruments: Allowed MIDI program numbers.
            notes_per_chord: Instruments per chord.
            include_initial: Include prompt in output.
            seq_len: Context window.
            top_k: Top-k threshold.
            top_p: Nucleus threshold.
            control_context: Control vector [mod, vol, expr, sustain] or None.

        Returns:
            Tuple of (muspy.Music, np.ndarray).
        """
        if allowed_instruments is None:
            allowed_instruments = list(range(128))
        generated = list(prompt_tokens) if include_initial else []

        # Pre-allocate tensors
        style_tensor = torch.tensor([style], dtype=torch.long, device=self.device)
        instr_tensor = torch.tensor([allowed_instruments], dtype=torch.long, device=self.device)

        num_instr = load_settings().model.num_instruments
        instr_mask = None
        if allowed_instruments:
            instr_mask = torch.full((1, 1, num_instr), -1e9, dtype=torch.float, device=self.device)
            instr_mask[:, :, allowed_instruments] = 0.0

        # Control vector
        ctrl_dim = load_settings().model.control_dim
        cv = np.zeros(ctrl_dim, dtype=np.float32)
        if control_context is not None:
            for i in range(min(len(control_context), 4)):
                cv[i] = control_context[i]
        else:
            cv[1] = 0.7  # default volume
            cv[2] = 0.7  # default expression
        control_tensor = torch.from_numpy(cv).float().to(self.device)
        control_tensor = control_tensor.unsqueeze(0).unsqueeze(0)

        x = self._init_context_buffer(prompt_tokens, seq_len)
        hidden_states = None

        for _ in range(num_steps):
            cur_ctrl = control_tensor.expand(1, x.size(1), -1)
            pitch_logits, dur_logits, vel_logits, instr_logits, hidden_states = self.model(
                x, style=style_tensor, instr_context=instr_tensor,
                control_context=cur_ctrl, hidden_states=hidden_states
            )

            if instr_mask is not None:
                mask = instr_mask.expand(-1, instr_logits.size(1), -1)
                instr_logits = instr_logits + mask

            pitch_l = pitch_logits[:, -1, :]
            dur_l = dur_logits[:, -1, :]
            vel_l = vel_logits[:, -1, :]
            instr_l = instr_logits[:, -1, :]

            # Batch instrument sampling
            instr_probs = torch.softmax(instr_l / temperature, dim=-1)
            sampled = torch.multinomial(
                instr_probs.squeeze(0),
                num_samples=min(notes_per_chord * 2, num_instr),
                replacement=False,
            )
            selected = []
            for idx in sampled.tolist():
                if idx in allowed_instruments and idx not in selected:
                    selected.append(idx)
                if len(selected) >= notes_per_chord:
                    break
            while len(selected) < notes_per_chord:
                remaining = [i for i in allowed_instruments if i not in selected]
                if remaining:
                    selected.append(remaining[0])
                else:
                    break

            for instr in selected:
                pitch = sample_top_k_top_p(pitch_l.squeeze(0), temperature, top_k, top_p)
                dur = sample_top_k_top_p(dur_l.squeeze(0), temperature, top_k, top_p)
                vel = sample_top_k_top_p(vel_l.squeeze(0), temperature, top_k, top_p)
                note = np.array([pitch, dur, vel, instr], dtype=np.int16)
                generated.append(note)

                x = torch.roll(x, shifts=-1, dims=1)
                x[0, -1, 0] = pitch
                x[0, -1, 1] = dur
                x[0, -1, 2] = vel
                x[0, -1, 3] = instr

        midi_music, _ = self._decode_output(generated, style)
        return midi_music, np.array(generated, dtype=np.int16)

    def generate_with_preset(self, preset_name: str, **kwargs):
        """Method 4 — Generate using a named control preset.

        Args:
            preset_name: One of expressive, gentle, bright, sustained,
                         percussive, dreamy, neutral, nothing.
            **kwargs: Passed to generate_with_control_conditioning().

        Returns:
            Tuple of (muspy.Music, np.ndarray).

        Raises:
            ValueError: If preset_name is unknown.
        """
        if preset_name not in CONTROL_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {list(CONTROL_PRESETS.keys())}"
            )
        kwargs["control_context"] = CONTROL_PRESETS[preset_name]
        return self.generate_with_control_conditioning(**kwargs)

    # ------------------------------------------------------------------
    # Aliases matching the original thesis codebase
    # ------------------------------------------------------------------

    def method_1(self, **kwargs):
        return self.generate(**kwargs)

    def method_2(self, **kwargs):
        return self.generate_natural(**kwargs)

    def method_3(self, **kwargs):
        return self.generate_force_polyphonic(**kwargs)

    def method_4(self, **kwargs):
        return self.generate_with_preset(**kwargs)

    # ------------------------------------------------------------------
    # Shared internals
    # ------------------------------------------------------------------

    def _init_context_buffer(
        self, prompt_tokens: np.ndarray, seq_len: int
    ) -> torch.Tensor:
        """Initialize the GPU circular context buffer from prompt tokens.

        Args:
            prompt_tokens: Primer token array (N, 4).
            seq_len: Desired context window length.

        Returns:
            GPU tensor of shape (1, seq_len, 4).
        """
        prompt_tokens = np.array(prompt_tokens, dtype=np.int64)
        plen = len(prompt_tokens)

        if plen >= seq_len:
            ctx = prompt_tokens[-seq_len:]
        elif plen > 0:
            pad = np.zeros((seq_len - plen, 4), dtype=np.int64)
            ctx = np.concatenate([pad, prompt_tokens])
        else:
            ctx = np.zeros((seq_len, 4), dtype=np.int64)

        return torch.from_numpy(ctx).unsqueeze(0).to(self.device)

    def _generate_base(
        self,
        prompt_tokens: np.ndarray,
        style: int,
        num_steps: int,
        temperature: float,
        top_k: int,
        include_initial: bool,
        allowed_instruments: Optional[List[int]],
        seq_len: int = 50,
    ) -> tuple:
        """Shared base for Methods 1 and 2 (single-note generation)."""
        if allowed_instruments is None:
            allowed_instruments = list(range(128))
        generated = list(prompt_tokens) if include_initial else []

        # Pre-allocated tensors
        style_tensor = torch.tensor([style], dtype=torch.long, device=self.device)
        instr_tensor = torch.tensor([allowed_instruments], dtype=torch.long, device=self.device)

        num_instr = load_settings().model.num_instruments
        instr_mask = None
        if allowed_instruments:
            instr_mask = torch.full((1, 1, num_instr), -1e9, dtype=torch.float, device=self.device)
            instr_mask[:, :, allowed_instruments] = 0.0

        x = self._init_context_buffer(prompt_tokens, seq_len)
        hidden_states = None

        for _ in range(num_steps):
            pitch_logits, dur_logits, vel_logits, instr_logits, hidden_states = self.model(
                x, style=style_tensor, instr_context=instr_tensor,
                hidden_states=hidden_states
            )

            if instr_mask is not None:
                mask = instr_mask.expand(-1, instr_logits.size(1), -1)
                instr_logits = instr_logits + mask

            pitch = sample_top_k_top_p(
                pitch_logits[:, -1, :].squeeze(0), temperature, top_k
            )
            dur = sample_top_k_top_p(
                dur_logits[:, -1, :].squeeze(0), temperature, top_k
            )
            vel = sample_top_k_top_p(
                vel_logits[:, -1, :].squeeze(0), temperature, top_k
            )
            instr = sample_top_k_top_p(
                instr_logits[:, -1, :].squeeze(0), temperature, top_k
            )

            generated.append(np.array([pitch, dur, vel, instr], dtype=np.int16))

            # Circular buffer update
            x = torch.roll(x, shifts=-1, dims=1)
            x[0, -1, 0] = pitch
            x[0, -1, 1] = dur
            x[0, -1, 2] = vel
            x[0, -1, 3] = instr

        return self._decode_output(generated, style)

    def _decode_output(self, generated: list, style: int) -> tuple:
        """Convert generated token list to MIDI and token array."""
        tokens = np.array(generated, dtype=np.int16)
        music = self.processor.decode({
            "note_level": tokens,
            "instr_level": np.unique(tokens[:, 3]),
            "song_level": np.array([
                120, 4, len(np.unique(tokens[:, 3])), tokens.shape[0], style
            ]),
        })
        return music, tokens
