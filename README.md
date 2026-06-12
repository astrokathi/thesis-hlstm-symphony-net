# hlstm-framework

**Hierarchical LSTM Framework for Symbolic Music Generation**

A complete Python library for ingesting MIDI datasets, training a 3-layered Hierarchical LSTM, generating multi-instrument symbolic music, and evaluating results with objective metrics.

Built from the thesis: *"Generating a Symphony Using Hierarchical LSTM Networks: A Comparative Analysis on Constrained Hardware"* by Sai Sravan Kathi.

---

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Configure
cp .env.example .env
# Edit .env: set DATA_DIR to your MIDI dataset path

# 3. Ingest dataset
python -c "from hlstm_framework.data import MusicDataPipeline; MusicDataPipeline().ingest()"

# 4. Train
python execution_steps/e2e.py

# 5. Generate
python execution_steps/generate.py                    # uses latest checkpoint
python execution_steps/generate.py models/hlstm_epoch_5.pt  # custom checkpoint

# 6. Evaluate
python -c "
from hlstm_framework.evaluation import MusicMetrics
import muspy
metrics = MusicMetrics(instruments=[40,41,42])
print(metrics.compute_all(muspy.read_midi('output/generated.mid')))
"
```

---

## Overview

```
hlstm_framework/
├── __init__.py          # Package entry, __help__(), __version__
├── config/              # Pydantic settings from .env / env vars
│   └── settings.py
├── data/                # MIDI ingestion, clustering, augmentation, encoding
│   ├── loader.py        # Scan + load MIDI files
│   ├── cluster.py       # K-means style clustering (unlabeled datasets)
│   ├── augmentation.py  # Pitch transpose, tempo scale, velocity shift
│   ├── encoding.py      # Hierarchical event encoding/decoding
│   ├── dataset.py       # PyTorch Dataset (on-the-fly sliding windows)
│   └── pipeline.py      # End-to-end orchestration
├── models/              # H-LSTM architecture, trainer, factory
│   ├── layers.py        # CrossAttention, AttentionPooling, TokenFusion
│   ├── hlstm.py         # HEventModel + HLSTMTrainer
│   ├── factory.py       # Model creation from config
│   └── trainer.py       # High-level training entry point
├── generation/          # Samplers, presets, 4 generation methods
│   ├── sampler.py       # top-k, top-p, leap limiting
│   ├── presets.py       # Expressive control presets
│   └── generator.py     # MusicGenerator (4 methods)
├── evaluation/          # Metrics, visualizers, comparison
│   ├── metrics.py       # SSM, IOI, IUR, style, muspy metrics
│   └── visualizer.py    # Piano roll, bar charts, radar comparison
├── utils/               # Logging, Ray distributed training, MIDI I/O
│   ├── logging_utils.py
│   ├── ray_utils.py
│   └── midi_io.py
└── tests/               # 30+ unit tests
    ├── test_config.py
    ├── test_data.py
    ├── test_models.py
    ├── test_generation.py
    └── test_evaluation.py

execution_steps/         # Ready-to-run scripts
├── e2e.py               # End-to-end training
└── generate.py          # Generate from checkpoint
```

---

## Architecture

### 3-Layer Hierarchical LSTM

| Layer | Function | Conditioning | Thesis Reference |
|---|---|---|---|
| LSTM1 — Style | Global genre/rhythm structure | `style` embedding broadcast across time | §3.6.2 |
| LSTM2 — Instrumentation | Timbre, ensemble, instrument-specific patterns | `instr_context` via attention/mean pooling | §3.6.3 |
| LSTM3 — Control | Expression, dynamics, micro-timing | `control_context` dense vector | §3.6.4 |

### Optional Upgrades (config-gated)

| Feature | Config | Sprint | Thesis § |
|---|---|---|---|
| AMP mixed precision | `TRAIN_USE_AMP=true` | S1 | §4.2 |
| LR scheduling | `TRAIN_LR_SCHEDULER=plateau` | S1 | — |
| Multi-worker DataLoader | `TRAIN_NUM_WORKERS=2` | S1 | — |
| Generator circular buffer | (always on) | S1/S2 | — |
| On-the-fly sliding window dataset | (always on) | S2 | — |
| Bar/beat position encoding | `MODEL_USE_POSITION_ENCODING=true` | S3 | §4.12 |
| Gradient checkpointing | `MODEL_USE_CHECKPOINTING=true` | S3 | — |
| torch.compile | `MODEL_USE_TORCH_COMPILE=true` | S3 | — |
| Cross-attention (LSTM1↔LSTM2) | `MODEL_USE_CROSS_ATTENTION=true` | S4 | §6.6 |
| Attention pooling (instrument) | `MODEL_USE_INSTR_ATTN_POOLING=true` | S4 | §3.6.3 |
| Joint token fusion MLP | `MODEL_USE_TOKEN_FUSION=true` | S4 | §4.12 |
| Data augmentation | `DATA_ENABLE_AUGMENTATION=true` | S5 | — |
| Fixed global duration bins | `DATA_USE_FIXED_BINS=true` | S5 | — |
| K-means style clustering | `DATA_CLUSTER_STYLES=true` | S5 | — |
| Ray distributed training | `RAY_ENABLED=true` | — | — |

---

## Dataset Ingestion

### With Style Subdirectories

```
/path/to/dataset/
├── classical/       → style_id = 0
│   ├── song1.mid
│   └── ...
└── contemporary/    → style_id = 1
    ├── song1.mid
    └── ...
```

```python
from hlstm_framework.data import MusicDataPipeline
pipeline = MusicDataPipeline()
pipeline.ingest()  # uses DATA_DIR from .env
```

### Without Labels (K-means Clustering)

```python
# Set DATA_CLUSTER_STYLES=true in .env
pipeline.ingest()  # automatically clusters by musical features
```

Or programmatically:
```python
from hlstm_framework.data.loader import MusicDataLoader
from hlstm_framework.data.cluster import StyleCluster

loader = MusicDataLoader(data_dir="path/to/flat_midi")
songs = loader.load_flat()
clusterer = StyleCluster(n_clusters=2)
labels = clusterer.fit_predict(songs)  # [0, 1, 0, 1, ...]
```

---

## Training

### Single GPU
```python
from hlstm_framework.models.trainer import train_model
history = train_model()
```

### Ray Distributed (Multi-GPU / Multi-Node)

```bash
pip install ray[train]
```

```python
from hlstm_framework.models.trainer import train_model
history = train_model(use_ray=True)
# Uses RAY_NUM_WORKERS and RAY_USE_GPU from .env
```

### Manual Training Loop
```python
from hlstm_framework.models import HEventModel, HLSTMTrainer
from hlstm_framework.data import MusicDataset, MusicDataPipeline
from torch.utils.data import DataLoader, random_split

dataset = MusicDataset("data/encoded/encoded_tokens.pkl")
train_ds, val_ds = random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

model = HEventModel(use_cross_attention=True, use_token_fusion=True)
trainer = HLSTMTrainer(model, use_amp=True)
history = trainer.train(train_loader, val_loader, num_epochs=30)
```

---

## Generation

### Quick CLI (recommended)

```bash
python execution_steps/generate.py                          # default checkpoint
python execution_steps/generate.py models/hlstm_epoch_5.pt  # custom checkpoint
```

The script handles checkpoint loading (extracts `model_state_dict` from trainer-saved dicts), creates the prompt from training data if missing, and writes output to `output/generated.mid`.

### Programmatic API

```python
import torch
from hlstm_framework.generation import MusicGenerator
from hlstm_framework.data.encoding import HEventProcessor
from hlstm_framework.models import HEventModel
from hlstm_framework.config import load_settings
import muspy

device = load_settings().model.device

# Load checkpoint (saved by HLSTMTrainer as dict with "model_state_dict")
checkpoint = torch.load("models/hlstm_epoch_2.pt", map_location=device)
model = HEventModel()
model.load_state_dict(checkpoint["model_state_dict"])  # key difference!
model.to(device)

processor = HEventProcessor()
gen = MusicGenerator(model, processor)

prompt = muspy.read_midi("test/prompt.mid")
encoded = processor.encode(prompt, style_id=0)

# Method 1: Vanilla
midi, tokens = gen.generate(encoded["note_level"], style=0, num_steps=200)

# Method 2: Natural
midi, tokens = gen.generate_natural(encoded["note_level"], style=1, temperature=0.8)

# Method 3: Force Polyphonic
midi, tokens = gen.generate_force_polyphonic(encoded["note_level"], style=0, notes_per_chord=4)

# Method 4: Preset (recommended)
midi, tokens = gen.generate_with_preset("gentle", prompt_tokens=encoded["note_level"], style=1)

midi.write_midi("output/generated.mid")
print(f"Generated {len(tokens)} notes")
```

---

## Evaluation

```python
from hlstm_framework.evaluation import MusicMetrics, MetricsVisualizer, compare_generations
import muspy

# Single generation
metrics = MusicMetrics(instruments=[40, 41, 42])
scores = metrics.compute_all(muspy.read_midi("output/generated.mid"))
print(scores)
# {'ssm_score': 0.42, 'ioi_variance': 3.2, 'instrument_usage_ratio': 1.0, ...}

# Compare multiple generations
viz = MetricsVisualizer(output_dir="output/plots")
viz.plot_piano_roll(midi_music, "My Generation", "my_gen.png")
viz.plot_radar([scores_1, scores_2], labels=["A", "B"], filename="comparison.png")
```

---

## Configuration

All settings are managed via `.env` file or environment variables with `DATA_`, `MODEL_`, `TRAIN_`, `GEN_`, and `RAY_` prefixes. See `.env.example` for the complete reference.

```bash
# Load settings in code
from hlstm_framework.config import load_settings
cfg = load_settings()
print(cfg.model.hidden_dim)       # 512
print(cfg.training.batch_size)    # 32
```

---

## Running Tests

```bash
pip install -e ".[dev]"   # or: pip install pytest
python -m pytest tests/ -v
```

Or run individual test files:
```bash
python tests/test_models.py
python tests/test_data.py
python tests/test_generation.py
python tests/test_evaluation.py
python tests/test_config.py
```

---

## Project Links

- **Repository:** https://github.com/astrokathi/thesis-hlstm-symphony-net
- **Thesis:** *Generating a Symphony Using Hierarchical LSTM Networks: A Comparative Analysis on Constrained Hardware* (Sai Sravan Kathi, 2025)
- **Generated Samples:** https://astrokathi.github.io/thesis-hlstm-symphony-net/

---

## Citation

```bibtex
@mastersthesis{kathi2025hlstm,
  author  = {Kathi, Sai Sravan},
  title   = {Generating a Symphony Using Hierarchical LSTM Networks:
             A Comparative Analysis on Constrained Hardware},
  school  = {International University of Applied Sciences},
  year    = {2025},
}
```
