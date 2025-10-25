import muspy
import numpy as np
from scipy.spatial.distance import pdist, squareform


class MusicGenerationMetrics:
    def __init__(self, instruments_used=None, style=None, window_size=128):
        self.instruments_used = instruments_used
        self.style = style
        self.window_size = window_size

    def self_similarity_matrix_score(self, music, window_size=16):
        """Self-Similarity Matrix Score for structural coherence"""
        window_size = self.window_size
        # Extract note sequences per track
        sequences = []
        for track in music.tracks:
            if len(track.notes) == 0:
                continue
            track_notes = []
            for note in track.notes:
                track_notes.append({
                    'pitch': note.pitch,
                    'time': note.time,
                    'duration': note.duration
                })
            sequences.append(track_notes)

        if not sequences:
            return 0.0  # No tracks with notes

        # Create pitch-time vectors for each window
        vectors = []
        max_time = music.get_end_time()

        # If no duration, return 0
        if max_time == 0:
            return 0.0

        for start in range(0, max_time, window_size):
            window_vector = []
            for track_seq in sequences:
                # Count notes in this window
                notes_in_window = sum(1 for note in track_seq
                                      if start <= note['time'] < start + window_size)
                window_vector.append(notes_in_window)
            vectors.append(window_vector)

        # Need at least 2 vectors for similarity
        if len(vectors) < 2:
            return 0.0

        try:
            vectors_array = np.array(vectors)

            # Check if we have any variation (all zeros would cause issues)
            if np.all(vectors_array == 0):
                return 0.0

            # Compute similarity matrix
            similarity_matrix = 1 - squareform(pdist(vectors_array, metric='cosine'))

            # Handle NaN values that might occur
            if np.any(np.isnan(similarity_matrix)):
                return 0.0

            # Score based on diagonal consistency
            diagonal_strength = np.mean([similarity_matrix[i, i + 1] for i in range(len(vectors) - 1)])

            # Handle potential NaN
            return float(diagonal_strength) if not np.isnan(diagonal_strength) else 0.0

        except Exception as e:
            print(f"DEBUG SSM: Error in similarity calculation: {e}")
            return 0.0

    def inter_onset_interval_variance(self, music):
        """IOI Variance for rhythmic consistency (Layer 2 validation)"""
        all_iois = []

        for track in music.tracks:
            if len(track.notes) < 2:
                continue

            # Sort notes by time
            sorted_notes = sorted(track.notes, key=lambda x: x.time)

            # Calculate IOIs
            for i in range(1, len(sorted_notes)):
                ioi = sorted_notes[i].time - sorted_notes[i - 1].time
                if ioi > 0:  # Ignore simultaneous notes
                    all_iois.append(ioi)

        if len(all_iois) < 2:
            return 0.0

        # Lower variance = more rhythmically consistent
        return np.var(all_iois)

    def instrument_usage_ratio(self, music, target_instruments=None):
        """Instrument Usage Ratio for conditional fidelity (Layer 3 validation)"""
        target_instruments = self.instruments_used
        if target_instruments is None:
            target_instruments = []

        used_instruments = set()
        total_notes = 0
        target_notes = 0

        for track in music.tracks:
            instr = track.program
            used_instruments.add(instr)
            note_count = len(track.notes)
            total_notes += note_count

            if instr in target_instruments:
                target_notes += note_count

        if total_notes == 0:
            return 0.0

        # Ratio of notes using target instruments
        iur = target_notes / total_notes

        # Penalty for using non-target instruments
        extra_instruments = len(used_instruments - set(target_instruments))
        penalty = 1.0 / (1.0 + extra_instruments)

        return iur * penalty

    def style_perplexity_score(self, music, style_model=None):
        """Cross-Entropy Perplexity-like score for style adherence"""
        # This would require a pre-trained style classifier
        # For now, use musical features as proxy
        style_model = self.style
        features = []

        # Extract style-related features without muspy.note_density
        features.append(muspy.pitch_range(music))

        # Calculate note density manually
        total_notes = sum(len(track.notes) for track in music.tracks)
        total_time = music.get_end_time()
        note_density = total_notes / max(total_time, 1)  # Avoid division by zero
        features.append(note_density)

        features.append(self.inter_onset_interval_variance(music))

        # Normalize features (you'd need to calibrate these based on your style data)
        normalized_features = [
            min(features[0] / 60.0, 1.0),  # Pitch range normalization
            min(features[1] / 0.5, 1.0),  # Note density normalization (notes per tick)
            min(features[2] / 100.0, 1.0)  # Rhythm variance normalization
        ]

        # Lower score = more stylistically consistent (like perplexity)
        style_score = np.mean(normalized_features)
        return style_score

    # Muspy has these built-in:
    def get_muspy_metrics(self, music):
        """Get available muspy metrics"""
        metrics = {}

        # Pitch-related (these should exist in muspy)
        try:
            metrics['pitch_range'] = muspy.pitch_range(music)
            metrics['pitch_entropy'] = muspy.pitch_entropy(music)
            metrics['scale_consistency'] = muspy.scale_consistency(music)
        except AttributeError as e:
            print(f"Warning: Some pitch metrics not available: {e}")

        # Rhythm-related - only use available ones
        try:
            if hasattr(muspy, 'rhythm_consistency'):
                metrics['rhythm_consistency'] = muspy.rhythm_consistency(music)
            # Skip gross_rhythm_consistency since it doesn't exist
        except AttributeError as e:
            print(f"Warning: Rhythm metrics not available: {e}")

        # Note density (calculate manually since muspy.note_density doesn't exist)
        total_notes = sum(len(track.notes) for track in music.tracks)
        total_time = max(music.get_end_time(), 1)  # Avoid division by zero
        metrics['note_density'] = total_notes / total_time

        return metrics

    def compute_all_metrics(self, music, target_instruments=None, style_model=None):
        """Compute all metrics for comprehensive evaluation"""
        metrics = {}

        # Layer 1: Structural Coherence
        metrics['ssm_score'] = self.self_similarity_matrix_score(music)

        # Layer 2: Rhythmic Consistency
        metrics['ioi_variance'] = self.inter_onset_interval_variance(music)

        # Layer 3: Conditional Fidelity
        metrics['instrument_usage_ratio'] = self.instrument_usage_ratio(
            music, target_instruments
        )

        # Style Adherence
        metrics['style_score'] = self.style_perplexity_score(music, style_model)

        # Add muspy built-in metrics
        muspy_metrics = self.get_muspy_metrics(music)
        metrics.update(muspy_metrics)

        return metrics
