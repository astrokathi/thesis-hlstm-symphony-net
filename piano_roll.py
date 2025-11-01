import muspy
import matplotlib.pyplot as plt
import numpy as np

music = muspy.read_midi("data/gen/interstellar_trio2_dreamy.mid")

multi = muspy.to_pypianoroll(music)
multi.plot(
    mode="separate",            # 'separate' for stacked, 'same' for overlayed
    track_label="program",       # show instrument names
    preset="frame",         # color preset ('frame', 'pastel', etc.)
)

plt.show()
