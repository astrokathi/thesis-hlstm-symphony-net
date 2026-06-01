"""
hlstm-framework — Hierarchical LSTM Framework for Symbolic Music Generation.

A complete pipeline for ingesting, training, generating, and evaluating
symbolic music with a 3-layered Hierarchical LSTM (H-LSTM) architecture.

Typical usage:
    >>> from hlstm_framework import __help__
    >>> __help__()

    >>> from hlstm_framework.data import MusicDataPipeline
    >>> pipeline = MusicDataPipeline()
    >>> pipeline.ingest()           # load + encode MIDI files
    >>> pipeline.cluster_styles()   # unsupervised style labeling (optional)

    >>> from hlstm_framework.models import HEventModel, create_trainer
    >>> model = HEventModel()
    >>> trainer = create_trainer(model)
    >>> trainer.train()             # single-GPU or Ray-distributed

    >>> from hlstm_framework.generation import MusicGenerator
    >>> gen = MusicGenerator(model, processor)
    >>> midi, tokens = gen.generate(prompt, style=0)

    >>> from hlstm_framework.evaluation import evaluate
    >>> metrics = evaluate(midi, instruments=[40, 41, 42])
"""

__version__ = "1.0.0"
__author__ = "Kathi, Sai Sravan"
__thesis__ = (
    "Generating a Symphony Using Hierarchical LSTM Networks: "
    "A Comparative Analysis on Constrained Hardware"
)

import os
import sys

# Ensure the package root is on the path for relative imports
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, os.path.dirname(_root))


def __help__(verbose: bool = True) -> None:
    """Print a comprehensive overview of the framework's capabilities.

    Args:
        verbose: If True, prints detailed module-level descriptions.
                 If False, prints a compact summary.
    """
    print("=" * 72)
    print(f"  hlstm-framework v{__version__}")
    print(f"  {__thesis__}")
    print("=" * 72)
    print()
    print("  MODULES")
    print("  ───────")
    print("  hlstm_framework.config    — Pydantic settings from .env / env vars")
    print("  hlstm_framework.data      — MIDI ingestion, clustering, augmentation, dataset")
    print("  hlstm_framework.models    — H-LSTM architecture, trainer, factory, Ray wrapper")
    print("  hlstm_framework.generation— 4 generation methods, samplers, presets")
    print("  hlstm_framework.evaluation— Metrics, visualizers, comparison tools")
    print("  hlstm_framework.utils     — Logging, Ray orchestration, MIDI I/O")
    print()
    if verbose:
        print("  QUICKSTART")
        print("  ──────────")
        print("  1. Configure:  cp .env.example .env  # edit paths")
        print("  2. Ingest:     python -c \"from hlstm_framework.data import MusicDataPipeline;"
              " MusicDataPipeline().ingest()\"")
        print("  3. Train:      python -c \"from hlstm_framework.scripts import train; train.main()\"")
        print("  4. Generate:   python -c \"from hlstm_framework.scripts import generate; generate.main()\"")
        print("  5. Evaluate:   python -c \"from hlstm_framework.scripts import evaluate; evaluate.main()\"")
        print()
        print("  ARCHITECTURE (3-layer H-LSTM)")
        print("  ─────────────────────────────")
        print("  Layer 1 — Style conditionning (global genre/rhythm)")
        print("  Layer 2 — Instrumentation conditioning (timbre / ensemble)")
        print("  Layer 3 — Control context + expressive micro-timing")
        print()
        print("  UPGRADES (config-gated, all optional)")
        print("  ────────────────────────────────────")
        print("  • Position encoding  • Cross-attention  • Token fusion")
        print("  • AMP (mixed-precision)  • Gradient checkpointing  • torch.compile")
        print("  • Attention pooling  • LR scheduling  • Data augmentation")
        print("  • Ray distributed training  • Global fixed duration bins")
        print()
    print("  Report issues: https://github.com/astrokathi/thesis-hlstm-symphony-net")
    print("=" * 72)
