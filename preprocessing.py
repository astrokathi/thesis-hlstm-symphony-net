import muspy
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from h_event_processor import HEventProcessor
from collections import defaultdict
import pandas as pd
from random import shuffle
import random

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

DATA_DIR = "/Users/kathi.s/ML_Projects/thesis/data/SymphonyNet"
ENCODED_PATH = "data/encoded/encoded_tokens_25_new.pkl"
LIST_SUB_DIRECTORIES = ["classical", "contemporary"]
MIN_FILES_PER_CLASS = 25

processor = HEventProcessor()
encoded_dataset = []
stats = defaultdict(list)


def analyze_and_visualize(df_song, df_instr):
    """Perform dataset-level analysis & visualization."""
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")

    # Top instruments
    plt.subplot(2, 2, 1)
    top_instr = df_instr["instrument"].value_counts().head(10)
    sns.barplot(x=top_instr.values, y=top_instr.index)
    plt.title("Top 10 Instruments in Dataset")

    # Song length distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df_song["song_length"], bins=20)
    plt.title("Distribution of Song Lengths (time steps)")

    # Instruments per song (by style)
    plt.subplot(2, 2, 3)
    sns.boxplot(x="style", y="num_instruments", data=df_song)
    plt.title("Number of Instruments per Song")

    # Average pitch by style
    plt.subplot(2, 2, 4)
    sns.boxplot(x="style", y="avg_pitch", data=df_song)
    plt.title("Average Pitch by Style")

    plt.tight_layout()
    plt.savefig("data/encoded/analysis_report.png")
    plt.close()
    print("✅ Saved analysis visualization → analysis_report.png")

    # Console summary
    print("\nAverage Song Statistics by Style:")
    print(df_song.groupby("style")[["song_length", "num_instruments", "avg_pitch", "avg_velocity"]].mean())

    print("\nTop Instruments:")
    print(top_instr)


def preprocess_dataset():
    """Load, encode, pickle, and analyze SymphonyNet data."""
    for subdir in os.listdir(DATA_DIR):
        if subdir in LIST_SUB_DIRECTORIES:
            files_list = os.listdir(os.path.join(DATA_DIR, subdir))
            shuffle(files_list)
            for file in files_list[:min(MIN_FILES_PER_CLASS, len(files_list))]:
                if not file.endswith((".mid", ".midi", ".musicxml")):
                    continue
                song_path = os.path.join(DATA_DIR, subdir, file)
                try:
                    # Assign a style ID based on folder
                    style_id = 0 if subdir == "classical" else 1
                    song = muspy.read_midi(song_path)
                    # setting muspy attribute for the style
                    song.__setattr__("style_id", style_id)
                    encoded = processor.encode(song, style_id)
                    encoded_dataset.append(encoded)
                    # Collect stats
                    stats["song_name"].append(file)
                    stats["num_instruments"].append(len(encoded["instr_level"]))
                    stats["song_length"].append(encoded["song_level"][3])
                    stats["avg_pitch"].append(np.mean(encoded["note_level"][:, 0]))
                    stats["avg_velocity"].append(np.mean(encoded["note_level"][:, 2]))
                    stats["style"].append(style_id)
                    stats["instrument"].extend(list(encoded["instr_level"]))

                except Exception as e:
                    print(f"❌ Error processing {file}: {e}")

    # Pickle encoded dataset
    os.makedirs(os.path.dirname(ENCODED_PATH), exist_ok=True)
    with open(ENCODED_PATH, "wb") as f:
        pickle.dump(encoded_dataset, f)
    print(f"✅ Pickled encoded dataset → {ENCODED_PATH}")

    # Song-level stats
    df_song = pd.DataFrame({
        "song_name": stats["song_name"],
        "num_instruments": stats["num_instruments"],
        "song_length": stats["song_length"],
        "avg_pitch": stats["avg_pitch"],
        "avg_velocity": stats["avg_velocity"],
        "style": stats["style"]
    })

    # Instrument-level stats
    df_instr = pd.DataFrame({
        "song_name": np.repeat(stats["song_name"], [len(enc["instr_level"]) for enc in encoded_dataset]),
        "instrument": [instr for enc in encoded_dataset for instr in enc["instr_level"]]
    })

    analyze_and_visualize(df_song, df_instr)


if __name__ == "__main__":
    preprocess_dataset()
