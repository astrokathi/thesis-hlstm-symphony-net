# Software requirements

- PyCharm - Latest version
- Pyenv
- MuseScore4
- FluidSynth font files [FluidR3_GM.sf2](https://member.keymusician.com/Member/FluidR3_GM/)

---

# Hardware requirements

- Apple M1 - preferable (It's awesome if you have a better GPU, which uses CUDA)

---

# Installation Steps

```bash
pip install -r reqirements.txt
```

---

# Model

Rename the model which is available as .txt, to .pt and use it in the `test_generate.py` file to generate music based on the heuristics set and the parameters provided.

---

# Experimentation

Run the `experimentation.py`, to get the metrics loaded to the `index.html` and open the HTML file in any browser to load the results.