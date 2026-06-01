"""
PyTorch Dataset for H-LSTM training with on-the-fly sliding windows.

Memory-efficient — stores per-song arrays instead of pre-materializing every
window. Sliding windows are computed on-the-fly in __getitem__ via a prefix-sum
index mapping (O(n) memory vs O(n·seq_len)).

Usage:
    >>> from hlstm_framework.data.dataset import MusicDataset
    >>> dataset = MusicDataset("data/encoded/encoded_tokens.pkl", seq_len=128)
    >>> len(dataset)
    48231
    >>> x, y, style, instr, control = dataset[0]
    >>> x.shape
    (128, 4)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from hlstm_framework.config import load_settings


class MusicDataset(Dataset):
    """PyTorch Dataset for H-LSTM training.

    Stores encoded songs as per-song arrays and computes sliding windows
    on-the-fly. This avoids pre-materializing all O(n·seq_len) windows.

    Args:
        token_path: Path to pickled encoded dataset (.pkl).
        seq_len: Sliding window sequence length.
        use_control_context: If True, include control context tensor.
        control_dim: Dimension of control context vector.

    Attributes:
        seq_len: Window length.
        control_dim: Control vector dimension.
        cum_lengths: Prefix sum of (song_length - seq_len) for index mapping.
    """

    def __init__(
        self,
        token_path: str,
        seq_len: Optional[int] = None,
        use_control_context: bool = True,
        control_dim: Optional[int] = None,
    ):
        cfg = load_settings()
        self.seq_len = seq_len or cfg.training.seq_len
        self.control_dim = control_dim or cfg.model.control_dim
        self.use_control_context = use_control_context

        import pickle
        with open(token_path, "rb") as f:
            data_list: List[Dict] = pickle.load(f)

        # Per-song storage
        self.songs: List[Dict] = []
        self.cum_lengths: List[int] = [0]

        for song in data_list:
            note_seq = np.array(song["note_level"], dtype=np.int64)
            if len(note_seq) <= self.seq_len:
                continue

            control_seq = song.get("control_level", np.zeros((0, 4), dtype=np.int64))
            song_seq = song.get("song_level", [120, 4, 1, 100, 0])

            control_features = self._extract_control_features(control_seq, len(note_seq))

            self.songs.append({
                "note_seq": note_seq,
                "style": int(song_seq[4]),
                "instr": int(song_seq[2]),
                "control_features": control_features,
            })
            self.cum_lengths.append(self.cum_lengths[-1] + len(note_seq) - self.seq_len)

    # --- Public API ---

    def __help__(self) -> None:
        """Print usage information for MusicDataset."""
        print("MusicDataset — PyTorch Dataset for H-LSTM training")
        print("=" * 50)
        print(f"  Songs:     {len(self.songs)}")
        print(f"  Windows:   {len(self)}")
        print(f"  Seq len:   {self.seq_len}")
        print(f"  Control:   {self.control_dim}-dim")
        print()
        print("  Returns: (x, y, style, instr_context, control_context)")
        print("    x:               (seq_len, 4)  long  — input tokens")
        print("    y:               (seq_len, 4)  long  — target tokens")
        print("    style:           scalar  long        — style_id")
        print("    instr_context:   (1,)     long       — instrument id")
        print("    control_context: (seq_len, D) float  — control features")

    def __len__(self) -> int:
        """Total number of sliding windows across all songs."""
        return self.cum_lengths[-1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get a training sample by flat index.

        Returns:
            Tuple of (x, y, style, instr_context, control_context).
            - x: Input token sequence, shape (seq_len, 4)
            - y: Target (shifted) token sequence, shape (seq_len, 4)
            - style: Style label (scalar)
            - instr_context: Instrument context (1,)
            - control_context: Control features (seq_len, control_dim)
        """
        song_idx, offset = self._resolve_index(idx)
        song = self.songs[song_idx]

        x = song["note_seq"][offset:offset + self.seq_len]
        y = np.concatenate([x[1:], x[-1:]])  # next-token prediction

        style = np.array(song["style"], dtype=np.int64)
        instr_context = np.array([song["instr"]], dtype=np.int64)
        control_ctx = song["control_features"][offset:offset + self.seq_len]

        if len(control_ctx) < self.seq_len:
            pad = np.zeros((self.seq_len - len(control_ctx), self.control_dim), dtype=np.float32)
            control_ctx = np.concatenate([control_ctx, pad])

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(style, dtype=torch.long),
            torch.tensor(instr_context, dtype=torch.long),
            torch.tensor(control_ctx, dtype=torch.float),
        )

    # --- Private helpers ---

    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        """Map flat dataset index to (song_index, offset_within_song)."""
        for song_idx in range(len(self.songs)):
            if idx < self.cum_lengths[song_idx + 1]:
                offset = idx - self.cum_lengths[song_idx]
                return song_idx, offset
        return len(self.songs) - 1, 0

    def _extract_control_features(
        self, control_seq: np.ndarray, num_notes: int
    ) -> np.ndarray:
        """Extract per-note control features from MIDI control events.

        Args:
            control_seq: Array of (instrument, time, cc_number, cc_value).
            num_notes: Number of notes in the song.

        Returns:
            Float array of shape (num_notes, control_dim).
        """
        if len(control_seq) == 0:
            return np.zeros((num_notes, self.control_dim), dtype=np.float32)

        # Tracked controller codes
        control_codes = [1, 7, 11, 64, 71, 74]

        features = np.zeros((num_notes, self.control_dim), dtype=np.float32)
        for j, cc_code in enumerate(control_codes):
            if j >= self.control_dim:
                break
            cc_events = control_seq[control_seq[:, 2] == cc_code]
            if len(cc_events) > 0:
                features[:, j] = cc_events[-1, 3] / 127.0
            else:
                if cc_code == 7:   # volume
                    features[:, j] = 0.8
                elif cc_code == 11:  # expression
                    features[:, j] = 0.7

        return features
