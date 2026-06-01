"""
Data augmentation strategies for symbolic music.

Provides controlled variations (pitch transposition, tempo scaling, velocity
shift, re-instrumentation) to expand effective dataset size.

Usage:
    >>> from hlstm_framework.data.augmentation import DataAugmenter
    >>> augmenter = DataAugmenter(prob=0.3)
    >>> variants = augmenter.apply(song, style_id)  # list of (song, style_id)
"""

import copy
import random
from typing import List, Tuple

import muspy
import numpy as np


class DataAugmenter:
    """Apply musical augmentations to create training variants.

    Each augmentation type is applied independently with a configurable
    probability, allowing for stochastic composition of transformations.

    Args:
        prob: Base probability (0-1) for each augmentation type.
        transpose_semitones: List of semitone shifts to try.
        tempo_factors: List of tempo scaling factors.
        velocity_shifts: List of velocity adjustments.
        seed: Random seed for reproducibility.

    Attributes:
        augment_count: Total augmentation operations applied.
    """

    def __init__(
        self,
        prob: float = 0.3,
        transpose_semitones: List[int] = None,
        tempo_factors: List[float] = None,
        velocity_shifts: List[int] = None,
        seed: int = 42,
    ):
        self.prob = prob
        self.transpose_semitones = transpose_semitones or [-5, -2, 2, 5]
        self.tempo_factors = tempo_factors or [0.8, 1.25]
        self.velocity_shifts = velocity_shifts or [-15, 15]
        self._rng = random.Random(seed)
        np.random.seed(seed)
        self.augment_count = 0

    def __help__(self) -> None:
        """Print usage information for DataAugmenter."""
        print("DataAugmenter — Symbolic music data augmentation")
        print("=" * 50)
        print("Augmentations (each applied independently with prob):")
        print(f"  Pitch transposition: {self.transpose_semitones} semitones")
        print(f"  Tempo scaling:       {self.tempo_factors}")
        print(f"  Velocity shift:      {self.velocity_shifts}")
        print(f"  Probability per aug: {self.prob}")
        print()
        print("Methods:")
        print("  apply(song, style_id)  → List[(song, style_id)] variants")

    # --- Individual augmentation operations ---

    @staticmethod
    def transpose(song: muspy.Music, semitones: int) -> muspy.Music:
        """Transpose all notes by a fixed number of semitones.

        Args:
            song: Input muspy.Music object (not modified).
            semitones: Number of semitones to shift. Negative = lower.

        Returns:
            New muspy.Music with transposed pitches clamped to [0, 127].
        """
        song = copy.deepcopy(song)
        for track in song.tracks:
            for note in track.notes:
                note.pitch = int(np.clip(note.pitch + semitones, 0, 127))
        return song

    @staticmethod
    def tempo_scale(song: muspy.Music, factor: float) -> muspy.Music:
        """Scale note durations and tempos by a factor.

        Args:
            song: Input muspy.Music object (not modified).
            factor: Scaling factor (<1 = faster, >1 = slower).

        Returns:
            New muspy.Music with scaled timing.
        """
        song = copy.deepcopy(song)
        for track in song.tracks:
            for note in track.notes:
                note.duration = max(1, int(note.duration * factor))
                note.start = int(note.start * factor)
        if song.tempos:
            for tempo in song.tempos:
                tempo.qpm = max(20, int(tempo.qpm / factor))
        return song

    @staticmethod
    def velocity_shift(song: muspy.Music, shift: int) -> muspy.Music:
        """Add a fixed velocity offset to all notes.

        Args:
            song: Input muspy.Music object (not modified).
            shift: Velocity adjustment (can be negative).

        Returns:
            New muspy.Music with adjusted velocities clamped to [1, 127].
        """
        song = copy.deepcopy(song)
        for track in song.tracks:
            for note in track.notes:
                note.velocity = int(np.clip(note.velocity + shift, 1, 127))
        return song

    @staticmethod
    def reinstrument(song: muspy.Music, target_family: int) -> muspy.Music:
        """Re-assign instruments to a target GM family.

        GM families: 0=piano, 1=chromatic percussion, 2=organ, 3=guitar,
        4=bass, 5=strings, 6=ensemble, 7=brass, 8=reed, 9=pipe, 10=synth,
        11=ethnic, 12=percussive, 13=sound effects.

        Args:
            song: Input muspy.Music object (not modified).
            target_family: GM family index (0-15).

        Returns:
            New muspy.Music with remapped instrument programs.
        """
        song = copy.deepcopy(song)
        for track in song.tracks:
            base = target_family * 8
            offset = random.randint(0, 7)
            track.program = min(base + offset, 127)
        return song

    # --- Composite application ---

    def apply(
        self, song: muspy.Music, style_id: int
    ) -> List[Tuple[muspy.Music, int]]:
        """Generate augmented variants of a song.

        The original song is always included. Each augmentation type is
        applied independently based on the configured probability.

        Args:
            song: The original muspy.Music object.
            style_id: The style label for the original song.

        Returns:
            List of (muspy.Music, style_id) tuples, starting with the original.
        """
        variants: List[Tuple[muspy.Music, int]] = [(song, style_id)]

        # Pitch transposition
        for semitones in self.transpose_semitones:
            if self._rng.random() < self.prob:
                variants.append((self.transpose(song, semitones), style_id))
                self.augment_count += 1

        # Tempo scaling
        for factor in self.tempo_factors:
            if self._rng.random() < self.prob:
                variants.append((self.tempo_scale(song, factor), style_id))
                self.augment_count += 1

        # Velocity shift
        for shift in self.velocity_shifts:
            if self._rng.random() < self.prob:
                variants.append((self.velocity_shift(song, shift), style_id))
                self.augment_count += 1

        return variants
