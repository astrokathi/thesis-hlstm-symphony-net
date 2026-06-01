# Execution Steps

> Complete guide to dataset ingestion, training, testing, and evaluation for the H-LSTM Symphony Net.

---

## Table of Contents

1. [Dataset Ingestion](#1-dataset-ingestion)
2. [Training the Model](#2-training-the-model)
3. [Testing / Generating Music](#3-testing--generating-music)
4. [Metrics & Comparison](#4-metrics--comparison)
5. [Appendix: Environment & Config Quick Reference](#appendix)

---

## 1. Dataset Ingestion

### 1.1 Prerequisites

- **SymphonyNet dataset** downloaded to your machine (or any folder with `.mid` files organized into subdirectories by style).
- Python environment with dependencies installed (`pip install -r requirements.txt`).

### 1.2 Directory Structure Expected

```
/path/to/SymphonyNet/
├── classical/          # style_id = 0
│   ├── song1.mid
│   ├── song2.mid
│   └── ...
└── contemporary/       # style_id = 1
    ├── song1.mid
    ├── song2.mid
    └── ...
```

The subdirectory names are configured in `.env` via `PREPROCESS_SUB_DIRS` (default: `classical,contemporary`).

### 1.3 Configure Paths

Edit `.env` in the project root (or set env vars):

```bash
# Required: point to your dataset
PREPROCESS_DATA_DIR=/path/to/SymphonyNet
PREPROCESS_SUB_DIRS=classical,contemporary
PREPROCESS_MIN_FILES=25                  # files per class to process
PREPROCESS_ENCODED_PATH=data/encoded/encoded_tokens.pkl
```

> **How many files can be ingested?**
> - `MIN_FILES` caps the number of files processed **per subdirectory** (default: 25 → 25 classical + 25 contemporary = 50 total).
> - To use the full dataset, increase this to a large number (e.g., `MIN_FILES=10000`), but note:
>   - Each file is loaded, parsed, quantized, and pickled into a single `.pkl`.
>   - With ~1000 files the `.pkl` is roughly 500MB–2GB. Ensure your system has enough RAM.
>   - The H-LSTM was originally trained with ~50 files on M1 MPS (~45 min/epoch) and up to 1750 files on rented GPUs.
> - **Data augmentation** (Sprint 5) multiplies the count: each file can produce up to ~9 variants (transpositions + tempo + velocity), giving effective dataset sizes of 450+ from 50 files.

### 1.4 Run Preprocessing

```bash
python preprocessing.py
```

This will:
1. Scan each subdirectory in `DATA_DIR`.
2. Read each `.mid` file via MusPy.
3. Encode using `HEventProcessor` → extracts `note_level`, `instr_level`, `song_level`, `control_level`.
4. Quantize durations using log-scaled bins (or fixed global bins with `use_fixed_bins=True` in code).
5. Collect dataset statistics and generate an analysis plot → `data/encoded/analysis_report.png`.
6. Save everything to the pickled file at `ENCODED_PATH`.

**Optional: Enable augmentation** (edit `preprocessing.py` → call `preprocess_dataset(enable_augmentation=True)`).

**Optional: Fixed duration bins** (call `preprocess_dataset(use_fixed_bins=True)` for corpus-level consistent binning).

### 1.5 Output

| Output | Path | Description |
|---|---|---|
| Encoded dataset | `data/encoded/encoded_tokens.pkl` | Pickled list of song dicts |
| Analysis report | `data/encoded/analysis_report.png` | Dataset statistics visualization |

---

## 2. Training the Model

### 2.1 Configure Training

Edit `.env` or set env vars:

```bash
# Model architecture
MODEL_NUM_PITCHES=128
MODEL_NUM_DURATIONS=128
MODEL_NUM_VELOCITIES=128
MODEL_NUM_INSTRUMENTS=128
MODEL_STYLE_CLASSES=2
MODEL_EMBED_DIM=256
MODEL_CONTROL_DIM=128
MODEL_HIDDEN_DIM=512
MODEL_DROPOUT=0.3
MODEL_DEVICE=mps                      # "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu"

# Training parameters
TRAIN_SEQ_LEN=128                     # sequence window length
TRAIN_BATCH_SIZE=32                   # reduce if OOM on MPS
TRAIN_NUM_EPOCHS=30                   # max epochs
TRAIN_LR=0.001
TRAIN_VAL_SPLIT=0.2                   # 20% held out for validation
TRAIN_PATIENCE=7                      # early stopping after 7 epochs without improvement
TRAIN_TOKEN_PATH=data/encoded/encoded_tokens.pkl
TRAIN_OUTPUT_DIR=models
TRAIN_LOG_DIR=runs/hlstm

# Sprint upgrades (all optional, default off)
MODEL_USE_POSITION_ENCODING=false     # bar/beat position tokens
MODEL_USE_CHECKPOINTING=false         # gradient checkpointing (saves memory)
MODEL_USE_TORCH_COMPILE=false         # torch.compile graph optimization
MODEL_USE_CROSS_ATTENTION=false       # cross-attention between LSTM layers
MODEL_USE_INSTR_ATTN_POOLING=false    # attention pooling for instr context
MODEL_USE_TOKEN_FUSION=false          # joint token fusion MLP

# Training perf upgrades
TRAIN_USE_AMP=true                    # mixed precision (1.5-2x faster)
TRAIN_NUM_WORKERS=2                   # parallel data loading
TRAIN_PIN_MEMORY=true
TRAIN_LOSS_WEIGHTS=1.0,1.0,1.0,1.0    # pitch,dur,vel,instr weights
TRAIN_LR_SCHEDULER=plateau            # "plateau", "cosine", or "none"
TRAIN_LR_PATIENCE=3
TRAIN_LR_MIN=1e-6
TRAIN_GRAD_CLIP_NORM=0.0              # 0 = disabled
```

### 2.2 Start Training

```bash
python train.py
```

This runs `train_model()` which:
1. Loads the pickled dataset.
2. Splits into train/validation (80/20 by default).
3. Creates `HEventModel` and `HLSTMTrainer`.
4. Runs epochs with training + validation loops.
5. Logs loss, perplexity, and LR to TensorBoard (`runs/hlstm`).
6. Saves checkpoints to `models/` every epoch.
7. Early stops if validation loss doesn't improve for `PATIENCE` epochs.

### 2.3 Resume from Checkpoint

Edit the `__main__` block in `train.py`:

```python
if __name__ == "__main__":
    train_model(
        token_path=Config.TOKEN_PATH,
        resume_checkpoint="models/hlstm_epoch_9.pt"
    )
```

The trainer will load the weights and continue from the next epoch.

### 2.4 Monitoring Training

```bash
tensorboard --logdir runs/hlstm
```

Visit `http://localhost:6006` to view loss curves, perplexity, and learning rate.

### 2.5 How Many Files Can Be Trained?

| Hardware | Max Files | Epoch Time | Notes |
|---|---|---|---|
| Apple M1 MPS (8GB) | 50–100 | ~45 min | Thesis baseline; batch_size=32, seq_len=128 |
| Apple M1 MPS (16GB) | 200–500 | ~30 min | Can increase batch_size or seq_len |
| NVIDIA GPU (rented, e.g. RTX 4090) | 1000–1750 | ~10 min | Full SymphonyNet subset; ~6× faster than MPS |
| Any + augmentation | 500+ effective | varies | 50 files × 9 augmentations = 450 effective songs |

**Limiting factors on MPS:**
- Memory: AutoGrad graph scales with batch_size × seq_len. Reduce either if OOM.
- The dataset `__getitem__` now computes windows on-the-fly (Sprint 2), so memory is O(n) not O(n·seq_len).
- Enable `USE_CHECKPOINTING=true` to trade compute for ~40% memory savings, allowing larger batch sizes or longer sequences.

---

## 3. Testing / Generating Music

### 3.1 Quick Generation Test

**`test_generate.py`** provides a standalone test:

```bash
python test_generate.py
```

This script:
1. Loads a prompt MIDI (`test/pirates.mid`).
2. Loads the pre-trained model checkpoint (`models/hlstm_epoch_9.pt`).
3. Generates music using the preset method.
4. Saves the output MIDI to `new_gen/mop_gen_mul_v5.mid`.

**Customize the test:**
```python
# In test_generate.py, modify:
LIST_OF_INSTRUMENTS = [52, 53, 54]       # GM program numbers
PRESET = "neutral"                        # control preset
STYLE = 0                                 # 0=classical, 1=contemporary
NUM_SAMPLES = 5                           # generations per method
FILE_NAME = "new_gen/my_generation.mid"   # output path
```

### 3.2 Four Generation Methods

All methods are accessible through `MusicGenerator`:

| Method | Function | Best For |
|---|---|---|
| **Method 1 — Vanilla** | `generator.method_1(...)` | Controlled comparisons, stable outputs |
| **Method 2 — Natural** | `generator.method_2(...)` | Creative variations, exploring learned distribution |
| **Method 3 — Force Polyphonic** | `generator.method_3(...)` | Dense textures, large ensembles |
| **Method 4 — Preset** | `generator.method_4(preset_name=...)` | Expressive/controlled music (recommended) |

**Available presets** (for Method 4):
```
expressive: [0.8, 0.7, 0.9, 0.3]    # Lots of modulation and expression
gentle:     [0.2, 0.6, 0.7, 0.1]    # Soft and delicate
bright:     [0.4, 0.8, 0.8, 0.2]    # Clear and vibrant
sustained:  [0.3, 0.7, 0.7, 0.9]    # Long notes with sustain
percussive: [0.1, 0.8, 0.6, 0.0]    # Short, punchy notes
dreamy:     [0.9, 0.5, 0.8, 0.7]    # Lots of modulation, medium sustain
neutral:    [0.0, 0.7, 0.7, 0.0]    # Default neutral values
nothing:    [0.0, 0.0, 0.0, 0.0]
```

### 3.3 Generator Parameters

```python
generator.generate(
    prompt_tokens=d['note_level'],     # primer sequence from prompt MIDI
    style=0,                            # 0=classical, 1=contemporary
    num_steps=200,                      # notes to generate (~60-120s of music)
    temperature=0.5,                    # lower = more deterministic
    top_k=20,                           # top-k sampling (0=disable)
    allowed_instruments=[40, 41, 42],   # GM program numbers to allow
    include_initial=False               # include prompt in output?
)
```

### 3.4 Running Full Experimentation

**`experimentation.py`** runs all 4 methods across multiple samples and generates comparison plots:

```bash
python experimentation.py
```

Configure via `.env`:
```bash
EXP_INPUT_FILE=test/20centuryfox.mid    # prompt MIDI
EXP_PRESET=gentle
EXP_NUM_SAMPLES=3                       # generations per method
EXP_STYLE=1                             # 0=classical, 1=contemporary
EXP_INSTRUMENTS=40,41,42,43             # allowed instrument list
```

Outputs are saved to `assets/`:
- Piano roll plots: `assets/generation_plots/{method}/sample_*_piano_roll.png`
- Velocity time series: `assets/generation_plots/{method}/sample_*_velocity.png`
- Metrics plots: `assets/metrics_plots/{method}/sample_*_metrics.png`
- MIDI files: `assets/mid/{method}/sample_*_song.mid`
- WAV files: `assets/wav/{method}/sample_*_song.wav`

### 3.5 Loading a Specific Checkpoint

In any test script, specify the checkpoint path:

```python
model = HEventModel()
model.load_state_dict(torch.load("models/hlstm_epoch_9.pt", map_location="mps"))
generator = MusicGenerator(model, processor, device="mps")
```

Checkpoints are named `hlstm_epoch_{N}.pt` (improved) or `hlstm_epoch_{N}_not_improved.pt` (no improvement).

---

## 4. Metrics & Comparison

### 4.1 Available Metrics

Computed by `MusicGenerationMetrics` (`metrics.py`):

| Metric | Class | What It Measures |
|---|---|---|
| **SSM Score** | `self_similarity_matrix_score()` | Structural coherence — motif recurrence and phrase repetition (Layer 1 validation) |
| **IOI Variance** | `inter_onset_interval_variance()` | Rhythmic consistency — lower = more stable pulse (Layer 2 validation) |
| **IUR** | `instrument_usage_ratio()` | Conditional fidelity — ratio of notes using requested instruments (Layer 2 validation) |
| **Style Score** | `style_perplexity_score()` | Style adherence — pitch range + note density + rhythm fit (coarse proxy) |
| **Pitch Range** | `get_muspy_metrics()` | Spread of MIDI pitches used |
| **Pitch Entropy** | `get_muspy_metrics()` | Diversity of pitch selection |
| **Scale Consistency** | `get_muspy_metrics()` | Adherence to a musical scale |
| **Note Density** | `get_muspy_metrics()` | Notes per quarter note / tick |

### 4.2 Computing Metrics for a Single Generation

```python
from metrics import MusicGenerationMetrics
from h_event_processor import HEventProcessor
from model import HEventModel
from generator import MusicGenerator
import torch

# Load model + generator
processor = HEventProcessor()
model = HEventModel()
model.load_state_dict(torch.load("models/hlstm_epoch_9.pt", map_location="mps"))
generator = MusicGenerator(model, processor, device="mps")

# Load prompt
import muspy
prompt = muspy.read_midi("test/pirates.mid")
d = processor.encode(prompt, 0)

# Generate
midi_music, tokens = generator.method_4(
    preset_name="gentle",
    num_steps=100,
    prompt_tokens=d['note_level'],
    style=0,
    allowed_instruments=[40, 41, 42],
    notes_per_chord=3,
    temperature=0.7,
    top_k=40,
    top_p=0.9
)

# Compute metrics
metrics_calc = MusicGenerationMetrics(
    instruments_used=[40, 41, 42],
    style=0
)
metrics = metrics_calc.compute_all_metrics(midi_music)

print(metrics)
# {
#   'ssm_score': 0.42,
#   'ioi_variance': 3.21,
#   'instrument_usage_ratio': 1.0,
#   'style_score': 0.35,
#   'pitch_range': 52,
#   'pitch_entropy': 3.8,
#   'scale_consistency': 0.72,
#   'note_density': 0.08
# }
```

### 4.3 Comparing Multiple Generations

Use `GeneratorMetrics` (`visualize_metrics.py`) to run a full comparison:

```python
from visualize_metrics import GeneratorMetrics

gm = GeneratorMetrics(
    generator=generator,
    preset_name="gentle",
    style=0,
    instruments_used=[40, 41, 42],
    prompt_tokens=d['note_level'],
    num_generations=5,          # generate 5 samples
    num_steps=100,
    control_context=[],
    notes_per_chord=3,
    temperature=0.7,
    include_initial=False,
    top_k=50,
    top_p=0.9,
    generation_method=4,        # 1=vanilla, 2=natural, 3=force poly, 4=preset
    seq_length=128
)

all_music, all_metrics = gm.compare_multiple_generations()
```

This generates:
- Individual piano roll + velocity plots per sample.
- Individual metrics bar charts per sample.
- A **radar comparison chart** overlaying all samples.
- MIDI and WAV files for each sample.

### 4.4 Validating Generation Quality (Bulk)

The commented-out function in `test_generate.py` shows how to validate across N samples:

```python
def validate_generation_quality(generator, num_samples=5, preset_name="neutral"):
    all_metrics = []
    metrics_calculator = MusicGenerationMetrics(
        instruments_used=LIST_OF_INSTRUMENTS,
        style=STYLE
    )

    for i in range(num_samples):
        midi_music, _ = generator.generate_with_preset(
            preset_name=preset_name,
            num_steps=100,
            prompt_tokens=prompt_tokens,
            style=STYLE,
            allowed_instruments=LIST_OF_INSTRUMENTS,
            notes_per_chord=2,
            temperature=0.7,
            include_initial=False,
            top_k=2,
            top_p=0.9
        )
        metrics = metrics_calculator.compute_all_metrics(midi_music)
        all_metrics.append(metrics)

    # Average across samples
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_metrics, all_metrics
```

### 4.5 Interpreting Results

| Metric | Good Range | What to Look For |
|---|---|---|
| **SSM Score** | > 0.35 | Higher = better structural coherence. Transformer usually scores ~0.45, H-LSTM ~0.35-0.40 |
| **IOI Variance** | < 10 | Lower = more rhythmically stable. H-LSTM typically lower than Transformer |
| **IUR** | > 0.9 | Fraction of notes using requested instruments. H-LSTM achieves 1.0 consistently |
| **Style Score** | < 0.5 | Lower = better style adherence. H-LSTM outperforms both baselines |
| **Pitch Entropy** | 3.5–4.5 | Too low = repetitive; too high = random |
| **Scale Consistency** | > 0.6 | Higher = more tonal coherence |

### 4.6 Finding the Best Generation

The typical workflow to find the best generation:

1. **Run `experimentation.py`** to generate samples across all 4 methods.
2. **Compare the radar plots** in `assets/metrics_plots/` — look for samples with the largest area (high SSM + high IUR + low IOI variance = best overall).
3. **Listen to the WAV files** in `assets/wav/{method}/` — qualitative listening is essential since metrics don't capture emotional or harmonic quality.
4. **Refine parameters** — adjust `temperature`, `top_k`, `notes_per_chord`, and `preset_name` based on what sounded best.
5. **Publish results** — the project includes an `index.html` for sharing comparisons on GitHub Pages.

---

## Appendix

### A. Complete `.env` Template

```bash
# === MODEL ===
MODEL_NUM_PITCHES=128
MODEL_NUM_DURATIONS=128
MODEL_NUM_VELOCITIES=128
MODEL_NUM_INSTRUMENTS=128
MODEL_STYLE_CLASSES=2
MODEL_EMBED_DIM=256
MODEL_CONTROL_DIM=128
MODEL_HIDDEN_DIM=512
MODEL_DROPOUT=0.3
MODEL_DEVICE=mps
MODEL_USE_POSITION_ENCODING=false
MODEL_USE_CHECKPOINTING=false
MODEL_USE_TORCH_COMPILE=false
MODEL_USE_CROSS_ATTENTION=false
MODEL_USE_INSTR_ATTN_POOLING=false
MODEL_USE_TOKEN_FUSION=false

# === TRAINING ===
TRAIN_SEQ_LEN=128
TRAIN_BATCH_SIZE=32
TRAIN_NUM_EPOCHS=30
TRAIN_LR=0.001
TRAIN_VAL_SPLIT=0.2
TRAIN_PATIENCE=7
TRAIN_TOKEN_PATH=data/encoded/encoded_tokens_25_new.pkl
TRAIN_OUTPUT_DIR=models
TRAIN_LOG_DIR=runs/hlstm
TRAIN_USE_AMP=true
TRAIN_NUM_WORKERS=2
TRAIN_PIN_MEMORY=true
TRAIN_LOSS_WEIGHTS=1.0,1.0,1.0,1.0
TRAIN_LR_SCHEDULER=plateau
TRAIN_LR_PATIENCE=3
TRAIN_LR_MIN=1e-6
TRAIN_GRAD_CLIP_NORM=0.0

# === PREPROCESSING ===
PREPROCESS_DATA_DIR=/Users/kathi.s/ML_Projects/thesis/SymphonyNet
PREPROCESS_ENCODED_PATH=data/encoded/encoded_tokens_25_new.pkl
PREPROCESS_SUB_DIRS=classical,contemporary
PREPROCESS_MIN_FILES=25

# === EXPERIMENTATION ===
EXP_INPUT_FILE=test/20centuryfox.mid
EXP_PRESET=gentle
EXP_NUM_SAMPLES=3
EXP_STYLE=1
EXP_INSTRUMENTS=40,41,42,43
```

### B. Key Scripts Reference

| Script | Purpose |
|---|---|
| `preprocessing.py` | Dataset ingestion, encoding, analysis |
| `train.py` | Model training with checkpointing |
| `test_generate.py` | Quick generation test |
| `experimentation.py` | Multi-method comparison with metrics |
| `generator.py` | MusicGenerator class (4 methods) |
| `generator_v2.py` | Advanced generator with motif boosting + leap constraints |
| `metrics.py` | MusicGenerationMetrics evaluation |
| `visualize_metrics.py` | Plotting and comparison utilities |
| `combine.py` | Combine two MIDI files |
| `wav_composer.py` | Convert MIDI to WAV via FluidSynth |
| `h_event_processor.py` | Event encoding/decoding pipeline |
| `config.py` | Centralized configuration from env vars |
| `model.py` | HEventModel + HLSTMTrainer |
| `tests/test_model.py` | 7 unit tests for the model |
| `tests/test_processor.py` | 4 unit tests for the processor |

### C. Useful Commands

```bash
# Preprocess with augmentation
python -c "from preprocessing import preprocess_dataset; preprocess_dataset(enable_augmentation=True)"

# Train with a specific checkpoint
python train.py   # edit __main__ to set resume_checkpoint

# Quick generation test
python test_generate.py

# Full comparison
python experimentation.py

# Run unit tests
python -m pytest tests/ -v
# or
python tests/test_model.py
python tests/test_processor.py

# View training logs
tensorboard --logdir runs/hlstm
```
