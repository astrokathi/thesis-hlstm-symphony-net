# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands
- Install dependencies: `pip install -r requirements.txt` (Note: there is also a `reqirements.txt` in the root)
- Run music generation: `python test_generate.py`
- Run experiments and update metrics: `python experimentation.py`
- Train the model: `python train.py`
- Process data: `python preprocessing.py`

## Architecture & Structure
The project implements a Hierarchical LSTM (HLSTM) for music generation based on the SymphonyNet dataset.

### High-Level Flow
1. **Preprocessing**: `preprocessing.py` processes raw music data into encoded tokens stored in `data/encoded/`.
2. **Model Definition**: `model.py` defines the HLSTM architecture, which is designed to capture both local and global musical structures.
3. **Training**: `train.py` handles the training loop, utilizing the encoded tokens and saving model weights in `models/`.
4. **Generation**: `generator.py` (and `generator_v2.py`) implements the logic to generate music sequences based on a seed and model weights.
5. **Evaluation**: `evaluator.py` and `metrics.py` calculate musicality and accuracy metrics.
6. **Synthesis & Visualization**: `wav_composer.py` and `piano_roll.py` convert generated tokens into MIDI/WAV and visual representations.
7. **Experimentation**: `experimentation.py` orchestrates multiple generation runs and populates `index.html` with results for analysis.

### Key Components
- `h_event_processor.py`: Handles the event-based representation of music.
- `fetch_embeddings.py`: Manages embedding retrieval for the model.
- `visualize_metrics.py`: Creates plots for the experimental results.
