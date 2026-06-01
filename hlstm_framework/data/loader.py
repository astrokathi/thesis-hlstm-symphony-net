"""
MIDI file loader — scans directories and reads files via MusPy.

Supports two modes:
    - Subfolder-labeled: style directories (e.g., classical/, contemporary/)
    - Flat directory: all files loaded without labels (for clustering)

Usage:
    >>> from hlstm_framework.data.loader import MusicDataLoader
    >>> loader = MusicDataLoader(data_dir="/path/to/dataset")
    >>> songs = loader.load_all(sub_dirs=["classical", "contemporary"])
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import muspy
import numpy as np
from tqdm import tqdm

from hlstm_framework.config import load_settings


class MusicDataLoader:
    """Scans directories and loads MIDI files as muspy.Music objects.

    Args:
        data_dir: Root dataset directory path. Falls back to config if None.
        seed: Random seed for file shuffling.

    Attributes:
        data_dir: Resolved root directory.
        loaded_songs: List of (muspy.Music, style_id) tuples after load_all().
    """

    def __init__(self, data_dir: Optional[str] = None, seed: int = 42):
        cfg = load_settings()
        self.data_dir = Path(data_dir or cfg.data.dir).resolve()
        self.seed = seed
        self.loaded_songs: List[Tuple[muspy.Music, int]] = []
        random.seed(seed)
        np.random.seed(seed)

    def __help__(self) -> None:
        """Print usage information for MusicDataLoader."""
        print("MusicDataLoader — MIDI file scanner and loader")
        print("=" * 50)
        print("Methods:")
        print("  load_all(sub_dirs, max_per_class)  — Load from style-labeled subdirs")
        print("  load_flat(max_files)               — Load all .mid files from root (no labels)")
        print("  get_stats()                        — Print dataset statistics")
        print()
        print("Config params (prefix DATA_):")
        print("  dir              — Root dataset path")
        print("  sub_dirs         — Comma-separated style subdirectory names")
        print("  min_files_per_class — Max files per style class")
        print("  cluster_styles   — Enable K-means clustering for unlabeled files")

    def _is_valid_midi(self, path: Path) -> bool:
        """Check if a file has a valid MIDI extension."""
        return path.suffix.lower() in (".mid", ".midi", ".musicxml")

    def load_all(
        self,
        sub_dirs: Optional[List[str]] = None,
        max_per_class: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[Tuple[muspy.Music, int]]:
        """Load MIDI files from style-labeled subdirectories.

        Each subdirectory is treated as one style class, with the style_id
        assigned by its index in the sub_dirs list.

        Args:
            sub_dirs: List of subdirectory names. Defaults to config data.sub_dirs_list.
            max_per_class: Max files to load per class. Defaults to config data.min_files_per_class.
            show_progress: Show a tqdm progress bar.

        Returns:
            List of (muspy.Music, style_id) tuples.
        """
        cfg = load_settings()
        sub_dirs = sub_dirs or cfg.data.sub_dirs_list
        max_per_class = max_per_class or cfg.data.min_files_per_class

        self.loaded_songs = []

        for style_id, subdir in enumerate(sub_dirs):
            dir_path = self.data_dir / subdir
            if not dir_path.is_dir():
                print(f"[WARN] Subdirectory not found: {dir_path}")
                continue

            files = sorted([
                f for f in dir_path.iterdir()
                if f.is_file() and self._is_valid_midi(f)
            ])
            random.shuffle(files)

            if max_per_class and len(files) > max_per_class:
                files = files[:max_per_class]

            iterator = tqdm(files, desc=f"Loading [{subdir}]") if show_progress else files
            for fpath in iterator:
                try:
                    song = muspy.read_midi(str(fpath))
                    self.loaded_songs.append((song, style_id))
                except Exception as exc:
                    print(f"  [SKIP] {fpath.name}: {exc}")

        print(f"[OK] Loaded {len(self.loaded_songs)} songs from {len(sub_dirs)} style classes.")
        return self.loaded_songs

    def load_flat(
        self,
        max_files: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[muspy.Music]:
        """Load all MIDI files from the root data directory (no style labels).

        Use this when files are not organized in style subdirectories. Labels
        can later be assigned via StyleCluster.

        Args:
            max_files: Maximum files to load. None = all files.
            show_progress: Show a tqdm progress bar.

        Returns:
            List of muspy.Music objects (no style_id).
        """
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Data directory not found: {self.data_dir}")

        files = sorted([
            f for f in self.data_dir.iterdir()
            if f.is_file() and self._is_valid_midi(f)
        ])
        random.shuffle(files)

        if max_files:
            files = files[:max_files]

        songs: List[muspy.Music] = []
        iterator = tqdm(files, desc="Loading flat") if show_progress else files
        for fpath in iterator:
            try:
                songs.append(muspy.read_midi(str(fpath)))
            except Exception as exc:
                print(f"  [SKIP] {fpath.name}: {exc}")

        print(f"[OK] Loaded {len(songs)} songs (flat, unlabeled).")
        return songs

    def get_stats(self) -> Dict[str, object]:
        """Return dataset statistics."""
        if not self.loaded_songs:
            return {"status": "no songs loaded"}

        songs, styles = zip(*self.loaded_songs) if self.loaded_songs else ([], [])
        durations = [s.get_end_time() for s in songs]
        track_counts = [len(s.tracks) for s in songs]
        unique_instruments = set()
        for s in songs:
            for t in s.tracks:
                unique_instruments.add(t.program)

        return {
            "total_songs": len(songs),
            "style_classes": len(set(styles)),
            "avg_duration": float(np.mean(durations)) if durations else 0,
            "avg_tracks": float(np.mean(track_counts)) if track_counts else 0,
            "unique_instruments": len(unique_instruments),
            "total_duration_minutes": float(sum(durations)) / 60 if durations else 0,
        }
