"""
High-level training entry point.

Builds model, dataset, dataloaders, and trainer from config, then runs the
training loop. Optionally uses Ray for distributed training.

Usage:
    >>> from hlstm_framework.models.trainer import train_model
    >>> history = train_model()
    >>> print(history["val_loss"][-1])
"""

from typing import Dict, List, Optional

from hlstm_framework.config import load_settings
from hlstm_framework.models.factory import create_model, create_trainer
from hlstm_framework.data.dataset import MusicDataset
from hlstm_framework.utils.ray_utils import is_ray_available, RayTrainWrapper
import torch


def train_model(
    token_path: Optional[str] = None,
    use_ray: Optional[bool] = None,
) -> Dict[str, List[float]]:
    """Run the full H-LSTM training pipeline.

    Orchestrates dataset loading, model creation, optional Ray distribution,
    and the training loop.

    Args:
        token_path: Path to pickled encoded dataset. Defaults to config.
        use_ray: If True, use Ray for distributed training.
                 Defaults to config RAY_ENABLED.

    Returns:
        Training history dict with keys: train_loss, val_loss, perplexity.
    """
    cfg = load_settings()
    token_path = token_path or cfg.training.token_path
    use_ray = use_ray if use_ray is not None else cfg.ray.enabled

    # Resolve device early so DataLoader adapts to platform limits
    device = cfg.model.device
    pin_memory = cfg.training.pin_memory and device == "cuda"
    num_workers = cfg.training.num_workers if device in ("cuda", "cpu") else 0

    # --- Dataset ---
    dataset = MusicDataset(
        token_path=token_path,
        seq_len=cfg.training.seq_len,
        use_control_context=True,
        control_dim=cfg.model.control_dim,
    )

    val_size = int(len(dataset) * cfg.training.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # --- Model ---
    model = create_model()
    if cfg.model.use_torch_compile:
        model.compile_model()

    # --- Distributed training via Ray ---
    if use_ray and is_ray_available():
        print("[TRAIN] Using Ray distributed training")
        ray_wrapper = RayTrainWrapper(
            train_loader=train_loader,
            val_loader=val_loader,
            num_workers=cfg.ray.num_workers,
            use_gpu=cfg.ray.use_gpu,
            storage_path=cfg.ray.storage_path,
        )
        return ray_wrapper.run()

    # --- Local training ---
    trainer = create_trainer(model)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.training.num_epochs,
        output_dir=cfg.training.output_dir,
        log_dir=cfg.training.log_dir,
        patience=cfg.training.patience,
    )

    return history
