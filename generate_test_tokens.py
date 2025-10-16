from h_event_processor import HEventProcessor
import muspy


processor = HEventProcessor()
song = muspy.read_midi("test/interstellar.mid")

d = processor.encode(song)

print(d['note_level'])
