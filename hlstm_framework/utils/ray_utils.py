"""
Ray distributed training utilities.

Wraps H-LSTM training in Ray Train for multi-GPU / multi-node scaling.
Supports Ray's TorchTrainer with automatic checkpointing and result reporting.

Usage:
    >>> from hlstm_framework.utils.ray_utils import RayTrainWrapper
    >>> wrapper = RayTrainWrapper(train_loader, val_loader, num_workers=4, use_gpu=True)
    >>> history = wrapper.run()
"""

from typing import Dict, List, Optional


def is_ray_available() -> bool:
    """Check if Ray is installed and can be imported.

    Returns:
        True if ray[train] is available.
    """
    try:
        import ray
        from ray import train as ray_train
        ray_train  # suppress unused
        return True
    except ImportError:
        return False


class RayTrainWrapper:
    """Wrap H-LSTM training in a Ray Train distributed session.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_workers: Number of Ray training workers.
        use_gpu: If True, allocate one GPU per worker.
        storage_path: Ray result storage directory.
        checkpoint_freq: Checkpoint frequency in epochs.

    Note:
        Ray is an optional dependency. This class raises ImportError
        if ray is not installed when run() is called.
    """

    def __init__(
        self,
        train_loader,
        val_loader,
        num_workers: int = 1,
        use_gpu: bool = False,
        storage_path: str = "/tmp/ray_results",
        checkpoint_freq: int = 1,
    ):
        if not is_ray_available():
            raise ImportError(
                "Ray is not installed. Install with: pip install ray[train]"
            )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.storage_path = storage_path
        self.checkpoint_freq = checkpoint_freq

    def run(self) -> Dict[str, List[float]]:
        """Execute distributed training via Ray.

        Returns:
            Training history dict.
        """
        import ray
        from ray import train as ray_train
        from ray.train import ScalingConfig, RunConfig, CheckpointConfig
        from ray.train.torch import TorchTrainer

        from hlstm_framework.config import load_settings
        from hlstm_framework.models.factory import create_model, create_trainer
        from hlstm_framework.models.hlstm import HLSTMTrainer

        cfg = load_settings()

        def train_func():
            """Training function executed on each Ray worker."""
            import os
            import math
            import torch.distributed as dist
            from torch.utils.tensorboard import SummaryWriter

            # Get worker info
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            rank = int(os.environ.get("RANK", 0))

            # Shard dataset
            # Note: In production, use Ray Data + train_loader sharding
            train_loader = ray_train.get_dataset_shard("train")
            val_loader = ray_train.get_dataset_shard("val")

            model = create_model()
            if cfg.model.use_torch_compile:
                model.compile_model()

            trainer = create_trainer(model)

            best_val_loss = float("inf")
            epochs_no_improve = 0

            for epoch in range(1, cfg.training.num_epochs + 1):
                train_loss = 0.0
                val_loss = 0.0

                for batch in train_loader:
                    loss = trainer.train_step(*batch)
                    train_loss += loss

                avg_train_loss = train_loss / len(train_loader)
                train_ppl = math.exp(avg_train_loss) if avg_train_loss < 20 else float("inf")

                for batch in val_loader:
                    val_loss += trainer.eval_step(*batch)

                avg_val_loss = val_loss / len(val_loader)

                # LR step
                trainer.scheduler_step(avg_val_loss)

                metrics = {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_perplexity": train_ppl,
                }

                # Checkpoint
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                ray_train.report(
                    metrics=metrics,
                    checkpoint=ray_train.Checkpoint.from_dict(
                        {"model": trainer.model.state_dict()}
                    ) if epoch % cfg.ray.checkpoint_freq == 0 else None,
                )

                if epochs_no_improve >= cfg.training.patience and rank == 0:
                    print(f"[Ray] Early stopping at epoch {epoch}")

        # Build Ray TorchTrainer
        ray_trainer = TorchTrainer(
            train_func,
            scaling_config=ScalingConfig(
                num_workers=self.num_workers,
                use_gpu=self.use_gpu,
            ),
            datasets={
                "train": self.train_loader.dataset,
                "val": self.val_loader.dataset,
            },
            run_config=RunConfig(
                storage_path=self.storage_path,
                checkpoint_config=CheckpointConfig(
                    num_to_keep=3,
                    checkpoint_frequency=self.checkpoint_freq,
                ),
            ),
        )

        result = ray_trainer.fit()
        print(f"[Ray] Training complete. Best val loss: {result.metrics.get('val_loss', 'N/A')}")
        return result.metrics
