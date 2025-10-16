from music21 import converter, corpus, analysis, roman
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def get_corpus_chord_distribution():
    """Extract chord root distributions from Bach chorales corpus."""
    pieces = corpus.search('bach')
    all_chords = []
    for p in pieces[:10]:  # use first 10 chorales for speed (can extend)
        s = p.parse().chordify()
        key = s.analyze("key")
        chords = [roman.romanNumeralFromChord(c, key).figure
                  for c in s.recurse().getElementsByClass("Chord")]
        all_chords.extend(chords)
    return Counter(all_chords)


def evaluate_midi(midi_file, corpus_distribution):
    # Load generated MIDI
    score = converter.parse(midi_file)

    # --- Metric 1: Key stability ---
    key = score.analyze("key")
    harmony_score = min(1.0, key.correlationCoefficient)

    # --- Metric 2: Chord progression validity vs. corpus ---
    s_chords = score.chordify()
    chords = [roman.romanNumeralFromChord(c, key).figure
              for c in s_chords.recurse().getElementsByClass("Chord")]

    gen_dist = Counter(chords)
    common_chords = sum(gen_dist[c] for c in gen_dist if c in corpus_distribution)
    progression_score = common_chords / (len(chords) + 1)

    # --- Metric 3: Rhythmic stability ---
    measures = score.parts[0].getElementsByClass("Measure")
    note_counts = [len(m.notes) for m in measures if len(m.notes) > 0]
    if len(note_counts) > 1:
        rhythm_var = np.var(note_counts)
        rhythm_score = 1 - min(rhythm_var / 10, 1)
    else:
        rhythm_score = 0.5

    # --- Metric 4: Pitch diversity ---
    pitches = [n.pitch.midi for n in score.flat.notes if n.isNote]
    if pitches:
        pitch_range = max(pitches) - min(pitches)
        pitch_diversity = min(pitch_range / 36, 1)  # normalize over 3 octaves
    else:
        pitch_diversity = 0.5

    # --- Metric 5: Repetition balance ---
    motifs = [str(n.pitch) for n in score.flat.notes if n.isNote]
    unique_ratio = len(set(motifs)) / (len(motifs) + 1)
    repetition_score = 1 - abs(unique_ratio - 0.5) * 2  # ideal ~0.5

    # Collect results
    scores = {
        "Harmony": harmony_score,
        "Progression": progression_score,
        "Rhythm": rhythm_score,
        "Pitch Diversity": pitch_diversity,
        "Repetition": repetition_score
    }

    # Final score out of 100
    overall_score = np.mean(list(scores.values())) * 100
    return scores, overall_score


def plot_results(scores, overall_score):
    labels = list(scores.keys())
    values = list(scores.values())

    # --- Bar chart ---
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="skyblue")
    plt.ylim(0, 1)
    plt.ylabel("Normalized Score (0–1)")
    plt.title(f"Music Evaluation - Final Score: {overall_score:.2f}/100")
    plt.show()

    # --- Radar chart ---
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    plt.title("Theory-Based Music Profile")
    plt.show()


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    midi_path = "gen/deep_seek_1.mid"  # drop your .mid file here
    corpus_dist = get_corpus_chord_distribution()
    scores, overall = evaluate_midi(midi_path, corpus_dist)

    print("Detailed Scores:", scores)
    print("Final Rating:", overall)
    plot_results(scores, overall)
