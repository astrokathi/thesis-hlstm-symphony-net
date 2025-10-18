from h_event_processor import HEventProcessor
import muspy
import numpy as np


processor = HEventProcessor()
song = muspy.read_midi("test/interstellar.mid")

d = processor.encode(song, 0)

# print(d)
events = []
styles = []
note_seq = np.array(d["note_level"], dtype=np.int64)
song_seq = d.get("song_level", [0])
events.extend(note_seq)
styles.extend([song_seq[0]] * len(note_seq))
print(styles)