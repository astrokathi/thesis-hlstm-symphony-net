"""
Unsupervised style clustering for unlabeled MIDI datasets.

Extracts musical features (avg pitch, note density, instrument diversity, etc.)
and applies K-means to assign style labels. Designed for flat MIDI directories
where no subfolder-based style labels exist.

Usage:
    >>> from hlstm_framework.data.cluster import StyleCluster
    >>> cluster = StyleCluster(n_clusters=2)
    >>> labels = cluster.fit_predict(songs)  # list of int style_id per song
"""

from typing import List, Optional

import muspy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from hlstm_framework.config import load_settings


class StyleCluster:
    """K-means clustering to assign style labels to unlabeled MIDI files.

    Features extracted per song:
        - Average MIDI pitch
        - Average velocity
        - Note density (notes per second)
        - Pitch range (max - min pitch)
        - Number of instruments used
        - Average inter-onset interval
        - Track count

    Args:
        n_clusters: Number of style clusters (default: 2 for classical/contemporary).
        random_state: Seed for reproducible clustering.
        standardize: If True, z-score standardize features before clustering.

    Attributes:
        kmeans: Fitted KMeans model.
        scaler: Fitted StandardScaler (if standardize=True).
        feature_names: Names of extracted features.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        random_state: int = 42,
        standardize: bool = True,
    ):
        cfg = load_settings()
        self.n_clusters = n_clusters or cfg.data.cluster_n_clusters
        self.random_state = random_state
        self.standardize = standardize
        self.kmeans: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names = [
            "avg_pitch", "avg_velocity", "note_density",
            "pitch_range", "num_instruments", "avg_ioi", "num_tracks",
        ]

    def __help__(self) -> None:
        """Print usage information for StyleCluster."""
        print("StyleCluster — Unsupervised K-means style labeling")
        print("=" * 50)
        print(f"  n_clusters: {self.n_clusters}")
        print(f"  Features: {', '.join(self.feature_names)}")
        print()
        print("Methods:")
        print("  extract_features(songs)  — Feature matrix of shape (N, 7)")
        print("  fit_predict(songs)       — Fit + predict style labels")
        print("  predict(songs)           — Predict using fitted model")
        print("  get_centroids()          — Cluster centroids in feature space")

    def extract_features(self, songs: List[muspy.Music]) -> np.ndarray:
        """Extract musical features from a list of muspy.Music objects.

        Args:
            songs: List of loaded muspy.Music objects.

        Returns:
            Float array of shape (len(songs), 7) with columns:
            [avg_pitch, avg_velocity, note_density, pitch_range,
             num_instruments, avg_ioi, num_tracks]
        """
        features = []
        for song in songs:
            all_pitches = []
            all_velocities = []
            all_onsets = []
            instruments = set()

            for track in song.tracks:
                instruments.add(track.program)
                for note in track.notes:
                    all_pitches.append(note.pitch)
                    all_velocities.append(note.velocity)
                    all_onsets.append(note.time)

            total_notes = len(all_pitches)
            total_time = max(song.get_end_time(), 1)

            avg_pitch = float(np.mean(all_pitches)) if all_pitches else 60.0
            avg_vel = float(np.mean(all_velocities)) if all_velocities else 64.0
            note_density = total_notes / total_time
            pitch_range = float(max(all_pitches) - min(all_pitches)) if len(all_pitches) > 1 else 0.0
            num_instr = len(instruments)

            # Average inter-onset interval
            if len(all_onsets) > 1:
                sorted_onsets = np.sort(all_onsets)
                iois = np.diff(sorted_onsets)
                avg_ioi = float(np.mean(iois[iois > 0])) if np.any(iois > 0) else 0.0
            else:
                avg_ioi = 0.0

            features.append([
                avg_pitch, avg_vel, note_density, pitch_range,
                num_instr, avg_ioi, len(song.tracks),
            ])

        return np.array(features, dtype=np.float32)

    def fit_predict(self, songs: List[muspy.Music]) -> List[int]:
        """Fit K-means on song features and return cluster assignments.

        Args:
            songs: List of muspy.Music objects.

        Returns:
            List of integer cluster labels (style_ids) with the same length as songs.
        """
        X = self.extract_features(songs)

        if self.standardize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        labels = self.kmeans.fit_predict(X)

        print(f"[CLUSTER] Assigned {len(songs)} songs to {self.n_clusters} style clusters.")
        for i in range(self.n_clusters):
            count = int((labels == i).sum())
            print(f"  Cluster {i}: {count} songs")

        return labels.tolist()

    def predict(self, new_songs: List[muspy.Music]) -> List[int]:
        """Predict style labels for new songs using a fitted cluster model.

        Args:
            new_songs: List of muspy.Music objects.

        Returns:
            List of integer cluster labels.

        Raises:
            RuntimeError: If called before fit_predict.
        """
        if self.kmeans is None:
            raise RuntimeError("StyleCluster has not been fitted. Call fit_predict() first.")

        X = self.extract_features(new_songs)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.kmeans.predict(X).tolist()

    def get_centroids(self) -> Optional[np.ndarray]:
        """Return cluster centroids in the standardized feature space."""
        if self.kmeans is None:
            return None
        return self.kmeans.cluster_centers_
