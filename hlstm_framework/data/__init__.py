"""
Data module — MIDI ingestion, clustering, augmentation, encoding, and dataset.

Pipeline:
    1. MusicDataLoader      — Scans directories, reads MIDI files via MusPy
    2. StyleCluster         — K-means clustering for unlabeled MIDI files
    3. DataAugmenter        — Pitch transposition, tempo scaling, velocity shift
    4. HEventProcessor      — Hierarchical event encoding / decoding
    5. MusicDataset          — PyTorch Dataset with on-the-fly sliding windows
    6. MusicDataPipeline     — End-to-end orchestration of the above
"""

from hlstm_framework.data.loader import MusicDataLoader
from hlstm_framework.data.cluster import StyleCluster
from hlstm_framework.data.augmentation import DataAugmenter
from hlstm_framework.data.encoding import HEventProcessor
from hlstm_framework.data.dataset import MusicDataset
from hlstm_framework.data.pipeline import MusicDataPipeline

__all__ = [
    "MusicDataLoader",
    "StyleCluster",
    "DataAugmenter",
    "HEventProcessor",
    "MusicDataset",
    "MusicDataPipeline",
]
