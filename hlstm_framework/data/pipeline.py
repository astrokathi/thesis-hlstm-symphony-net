"""
End-to-end data pipeline orchestrating loading, clustering, augmentation,
encoding, and dataset creation.

Usage:
    >>> from hlstm_framework.data import MusicDataPipeline
    >>> pipeline = MusicDataPipeline()
    >>> pipeline.ingest(enable_augmentation=True)
    >>> dataset = pipeline.create_dataset()
    >>> loader = pipeline.create_dataloader(dataset, batch_size=32)
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import muspy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from hlstm_framework.config import load_settings
from hlstm_framework.data.loader import MusicDataLoader
from hlstm_framework.data.cluster import StyleCluster
from hlstm_framework.data.augmentation import DataAugmenter
from hlstm_framework.data.encoding import HEventProcessor
from hlstm_framework.data.dataset import MusicDataset


class MusicDataPipeline:
    """End-to-end data pipeline for H-LSTM training data preparation.

    Orchestrates loading → (optional clustering) → (optional augmentation)
    → encoding → pickling → PyTorch Dataset → DataLoader.

    Args:
        processor: HEventProcessor instance. Created automatically if None.
        augmenter: DataAugmenter instance. Created automatically if None.
        cluster: StyleCluster instance. Created automatically if None.

    Attributes:
        encoded_dataset: List of encoded song dictionaries (after ingest).
        processor: The event encoder/decoder.
    """

    def __init__(
        self,
        processor: Optional[HEventProcessor] = None,
        augmenter: Optional[DataAugmenter] = None,
        cluster: Optional[StyleCluster] = None,
    ):
        self.processor = processor or HEventProcessor()
        self.augmenter = augmenter or DataAugmenter()
        self.cluster = cluster
        self.encoded_dataset: List[Dict] = []
        self._stats: Dict = {}

    def __help__(self) -> None:
        """Print usage information for MusicDataPipeline."""
        print("MusicDataPipeline — End-to-end data preparation")
        print("=" * 50)
        print("Pipeline stages:")
        print("  1. ingest()        — Load, (cluster), (augment), encode, pickle")
        print("  2. create_dataset()— Build PyTorch Dataset from pickle")
        print("  3. create_dataloader() — Build DataLoader for training")
        print("  4. plot_analysis() — Dataset statistics visualization")
        print()
        print("Config: set DATA_* env vars or .env entries")

    # --- Main pipeline ---

    def ingest(
        self,
        data_dir: Optional[str] = None,
        sub_dirs: Optional[List[str]] = None,
        max_per_class: Optional[int] = None,
        enable_augmentation: Optional[bool] = None,
        use_fixed_bins: Optional[bool] = None,
        force_recluster: bool = False,
        show_progress: bool = True,
    ) -> List[Dict]:
        """Run the full ingestion pipeline: load → cluster → augment → encode → pickle.

        If DATA_CLUSTER_STYLES is True (or force_recluster), the pipeline
        will use K-means clustering on musical features to assign style labels
        instead of relying on subdirectory names.

        Args:
            data_dir: Override DATA_DIR config.
            sub_dirs: Override DATA_SUB_DIRS config.
            max_per_class: Override DATA_MIN_FILES_PER_CLASS config.
            enable_augmentation: Override DATA_ENABLE_AUGMENTATION config.
            use_fixed_bins: Override DATA_USE_FIXED_BINS config.
            force_recluster: Force re-clustering even if sub_dirs exist.
            show_progress: Show progress bars.

        Returns:
            List of encoded song dictionaries.
        """
        cfg = load_settings()
        data_dir = data_dir or cfg.data.dir
        sub_dirs = sub_dirs or cfg.data.sub_dirs_list
        max_per_class = max_per_class or cfg.data.min_files_per_class
        enable_augmentation = (
            enable_augmentation if enable_augmentation is not None
            else cfg.data.enable_augmentation
        )
        use_fixed_bins = (
            use_fixed_bins if use_fixed_bins is not None
            else cfg.data.use_fixed_bins
        )

        loader = MusicDataLoader(data_dir=data_dir)

        # Determine mode: subfolder-labeled vs flat + cluster
        use_clustering = cfg.data.cluster_styles or force_recluster
        style_label_mode = "cluster" if use_clustering else "subdir"

        if use_clustering:
            print("[PIPELINE] Flat mode + K-means clustering enabled.")
            raw_songs = loader.load_flat(show_progress=show_progress)

            # Extract features and cluster
            clusterer = self.cluster or StyleCluster(
                n_clusters=cfg.data.cluster_n_clusters
            )
            style_labels = clusterer.fit_predict(raw_songs)
            song_style_pairs: List[Tuple[muspy.Music, int]] = list(
                zip(raw_songs, style_labels)
            )
            print(f"[PIPELINE] Cluster distribution: {np.bincount(style_labels)}")
        else:
            print(f"[PIPELINE] Subdirectory style mode: {sub_dirs}")
            song_style_pairs = loader.load_all(
                sub_dirs=sub_dirs,
                max_per_class=max_per_class,
                show_progress=show_progress,
            )

        # Augment and encode
        self.encoded_dataset = []
        for song, style_id in song_style_pairs:
            if enable_augmentation:
                variants = self.augmenter.apply(song, style_id)
            else:
                variants = [(song, style_id)]

            for aug_song, aug_style in variants:
                encoded = self.processor.encode(
                    aug_song, aug_style, use_fixed_bins=use_fixed_bins
                )
                self.encoded_dataset.append(encoded)

        # Sort by style for consistent batching
        self.encoded_dataset.sort(key=lambda d: int(d["song_level"][4]))

        # Pickle
        output_path = cfg.data.encoded_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self.encoded_dataset, f)

        print(f"[PIPELINE] Saved {len(self.encoded_dataset)} encoded songs → {output_path}")
        print(f"[PIPELINE] Style label mode: {style_label_mode}")
        if enable_augmentation:
            print(f"[PIPELINE] Augmentations applied: {self.augmenter.augment_count}")

        return self.encoded_dataset

    def create_dataset(
        self, token_path: Optional[str] = None, seq_len: Optional[int] = None
    ) -> MusicDataset:
        """Create a PyTorch Dataset from the pickled encoded data.

        Args:
            token_path: Path to .pkl file. Defaults to config DATA_ENCODED_PATH.
            seq_len: Sequence window length. Defaults to config TRAIN_SEQ_LEN.

        Returns:
            MusicDataset instance.
        """
        cfg = load_settings()
        token_path = token_path or cfg.data.encoded_path
        seq_len = seq_len or cfg.training.seq_len
        return MusicDataset(
            token_path=token_path,
            seq_len=seq_len,
            use_control_context=True,
            control_dim=cfg.model.control_dim,
        )

    def create_dataloader(
        self,
        dataset: MusicDataset,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ) -> DataLoader:
        """Create a PyTorch DataLoader from a MusicDataset.

        Args:
            dataset: MusicDataset instance.
            batch_size: Batch size. Defaults to config TRAIN_BATCH_SIZE.
            shuffle: Shuffle data every epoch.
            num_workers: Worker processes. Defaults to config TRAIN_NUM_WORKERS.
            pin_memory: Pin memory. Defaults to config TRAIN_PIN_MEMORY.

        Returns:
            DataLoader instance.
        """
        cfg = load_settings()
        batch_size = batch_size or cfg.training.batch_size
        num_workers = num_workers if num_workers is not None else cfg.training.num_workers
        pin_memory = pin_memory if pin_memory is not None else cfg.training.pin_memory

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )

    def plot_analysis(self, output_path: str = "data/encoded/analysis_report.png") -> str:
        """Generate a dataset analysis plot (pitch, instruments, song lengths, etc.).

        Must be called after ingest().

        Args:
            output_path: Path to save the plot image.

        Returns:
            Path to the saved plot.
        """
        if not self.encoded_dataset:
            raise RuntimeError("No encoded data. Call ingest() first.")

        stats: Dict = {
            "song_name": [],
            "num_instruments": [],
            "song_length": [],
            "avg_pitch": [],
            "avg_velocity": [],
            "style": [],
            "instrument": [],
        }

        for enc in self.encoded_dataset:
            stats["num_instruments"].append(len(enc["instr_level"]))
            stats["song_length"].append(int(enc["song_level"][3]))
            stats["avg_pitch"].append(float(np.mean(enc["note_level"][:, 0])))
            stats["avg_velocity"].append(float(np.mean(enc["note_level"][:, 2])))
            stats["style"].append(int(enc["song_level"][4]))
            stats["instrument"].extend(enc["instr_level"].tolist())

        df_song = pd.DataFrame({
            "song_length": stats["song_length"],
            "num_instruments": stats["num_instruments"],
            "avg_pitch": stats["avg_pitch"],
            "avg_velocity": stats["avg_velocity"],
            "style": stats["style"],
        })

        df_instr = pd.DataFrame({
            "instrument": stats["instrument"],
        })

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.flatten()

        # Top instruments
        top_instr = df_instr["instrument"].value_counts().head(10)
        axes[0].barh(range(len(top_instr)), top_instr.values)
        axes[0].set_yticks(range(len(top_instr)))
        axes[0].set_yticklabels(top_instr.index)
        axes[0].set_title("Top 10 Instruments")
        axes[0].invert_yaxis()

        # Song length distribution
        axes[1].hist(df_song["song_length"], bins=20)
        axes[1].set_title("Song Length Distribution")

        # Instruments per song by style
        df_song.boxplot(column="num_instruments", by="style", ax=axes[2])
        axes[2].set_title("Instruments per Song by Style")

        # Avg pitch by style
        df_song.boxplot(column="avg_pitch", by="style", ax=axes[3])
        axes[3].set_title("Average Pitch by Style")

        plt.suptitle("Dataset Analysis")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[PIPELINE] Analysis saved → {output_path}")
        return output_path
