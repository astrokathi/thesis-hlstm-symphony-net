import pickle
import numpy as np

with open("data/encoded/encoded_tokens_bkp_3_classes.pkl", "rb") as f:
    dataset = pickle.load(f)

all_notes = np.concatenate([song["note_level"] for song in dataset], axis=0)
print("max pitch:", all_notes[:,0].max())
print("max duration:", all_notes[:,1].max())
print("max velocity:", all_notes[:,2].max())
print("max instr:", all_notes[:,3].max())

