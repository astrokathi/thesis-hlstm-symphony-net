"""
Core H-LSTM model and trainer.

HEventModel:
    3-layer Hierarchical LSTM with config-gated architectural upgrades:
    - Position encoding, cross-attention, attention pooling, token fusion
    - Gradient checkpointing, torch.compile

HLSTMTrainer:
    Training loop with AMP, LR scheduling, loss weighting, gradient clipping.

Usage:
    >>> from hlstm_framework.models import HEventModel, HLSTMTrainer
    >>> model = HEventModel()
    >>> trainer = HLSTMTrainer(model)
    >>> trainer.train(train_loader, val_loader, num_epochs=30)
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from hlstm_framework.config import load_settings
from hlstm_framework.models.layers import (
    CrossAttention,
    AttentionPooling,
    TokenFusion,
)


class HEventModel(nn.Module):
    """3-layer Hierarchical LSTM for symbolic music generation.

    Processes event-level tokens [pitch, duration, velocity, instrument]
    through three stacked LSTM layers:
        1. Style conditioning (global genre/rhythm)
        2. Instrumentation conditioning (timbre/ensemble)
        3. Control context (expression/micro-timing)

    Args:
        num_pitches: Pitch vocabulary size.
        num_durations: Duration bin vocabulary size.
        num_velocities: Velocity vocabulary size.
        num_instruments: Instrument program vocabulary size.
        style_classes: Number of style classes.
        embed_dim: Embedding dimension for each feature.
        control_dim: Control context vector dimension.
        hidden_dim: LSTM hidden state dimension.
        dropout: Dropout rate.
        device: Torch device string.
        use_position_encoding: Enable bar/beat position encoding (5th channel).
        num_positions: Position vocabulary size.
        use_checkpointing: Enable gradient checkpointing.
        use_cross_attention: Enable cross-attention between LSTM1 and LSTM2.
        use_instr_attention_pooling: Attention pooling for instrument context.
        use_token_fusion: Enable joint token fusion MLP.

    Attributes:
        hidden_dim: LSTM hidden dimension.
        control_dim: Control context dimension.
    """

    def __init__(
        self,
        num_pitches: int = 128,
        num_durations: int = 128,
        num_velocities: int = 128,
        num_instruments: int = 128,
        style_classes: int = 2,
        embed_dim: int = 256,
        control_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        device: str = "auto",
        # Sprint 3
        use_position_encoding: bool = False,
        num_positions: int = 64,
        use_checkpointing: bool = False,
        # Sprint 4
        use_cross_attention: bool = False,
        use_instr_attention_pooling: bool = False,
        use_token_fusion: bool = False,
    ):
        super().__init__()

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.hidden_dim = hidden_dim
        self.control_dim = control_dim
        self.use_position_encoding = use_position_encoding
        self.use_checkpointing = use_checkpointing
        self.use_cross_attention = use_cross_attention
        self.use_token_fusion = use_token_fusion

        # --- Embeddings ---
        self.pitch_embed = nn.Embedding(num_pitches, embed_dim, device=device)
        self.duration_embed = nn.Embedding(num_durations, embed_dim, device=device)
        self.velocity_embed = nn.Embedding(num_velocities, embed_dim, device=device)
        self.instrument_embed = nn.Embedding(num_instruments, embed_dim, device=device)
        self.style_embed = nn.Embedding(style_classes, embed_dim, device=device)

        # Position encoding (Sprint 3)
        if use_position_encoding:
            self.position_embed = nn.Embedding(num_positions, embed_dim // 4, device=device)
            self.position_proj = nn.Linear(embed_dim // 4, hidden_dim, device=device)

        # --- Projections ---
        self.embed_proj = nn.Linear(embed_dim, hidden_dim, device=device)
        self.style_proj = nn.Linear(embed_dim, hidden_dim, device=device)
        self.instr_proj = nn.Linear(embed_dim, hidden_dim, device=device)
        self.control_proj = nn.Linear(control_dim, hidden_dim, device=device)

        # --- LSTM layers ---
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, device=device)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, device=device)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, device=device)

        # Cross-attention (Sprint 4)
        if use_cross_attention:
            self.cross_attn = CrossAttention(hidden_dim, num_heads=4, dropout=dropout)

        # Instrument context attention pooling (Sprint 4)
        self.use_instr_attention_pooling = use_instr_attention_pooling
        if use_instr_attention_pooling:
            self.instr_pooling = AttentionPooling(hidden_dim, num_heads=4)

        # Token fusion (Sprint 4)
        if use_token_fusion:
            self.token_fusion = TokenFusion(hidden_dim, fusion_dim=256, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        # --- Output heads ---
        self.pitch_out = nn.Linear(hidden_dim, num_pitches, device=device)
        self.duration_out = nn.Linear(hidden_dim, num_durations, device=device)
        self.velocity_out = nn.Linear(hidden_dim, num_velocities, device=device)
        self.instrument_out = nn.Linear(hidden_dim, num_instruments, device=device)

        self._compiled = False

    # --- Forward checkpoints ---

    def _lstm1_fwd(self, x, h):
        return self.lstm1(x, h)

    def _lstm2_fwd(self, x, h):
        return self.lstm2(x, h)

    def _lstm3_fwd(self, x, h):
        return self.lstm3(x, h)

    def forward(
        self,
        x: torch.Tensor,
        style: Optional[torch.Tensor] = None,
        instr_context: Optional[torch.Tensor] = None,
        control_context: Optional[torch.Tensor] = None,
        hidden_states: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass through the 3-layer hierarchical LSTM.

        Args:
            x: Input tensor (B, T, 4) — [pitch, duration, velocity, instrument].
                If position encoding is enabled and x has 5 channels, the 5th
                channel is treated as beat position.
            style: Style labels (B,) or (B, 1).
            instr_context: Instrument IDs (B, N) for conditioning.
            control_context: Control features (B, T, control_dim).
            hidden_states: Optional tuple of (h1, h2, h3) for stateful generation.

        Returns:
            Tuple of (pitch_logits, duration_logits, velocity_logits,
                      instrument_logits, (h1, h2, h3)).
        """
        batch_size, seq_len, n_features = x.size()
        device = x.device

        # --- Extract features ---
        pitch = x[:, :, 0].long()
        duration = x[:, :, 1].long()
        velocity = x[:, :, 2].long()
        instrument = x[:, :, 3].long()

        # --- Base embeddings ---
        x_embed = (
            self.pitch_embed(pitch)
            + self.duration_embed(duration)
            + self.velocity_embed(velocity)
            + self.instrument_embed(instrument)
        )
        x_embed = self.embed_proj(x_embed)

        # Position encoding (Sprint 3)
        if self.use_position_encoding and n_features >= 5:
            beat_pos = x[:, :, 4].long().clamp(0, self.position_embed.num_embeddings - 1)
            pos_emb = self.position_embed(beat_pos)
            pos_emb = self.position_proj(pos_emb)
            x_embed = x_embed + pos_emb

        # --- Conditioning ---
        style_emb: Optional[torch.Tensor] = None
        instr_emb: Optional[torch.Tensor] = None
        control_emb: Optional[torch.Tensor] = None

        # Style → LSTM1
        if style is not None:
            style_t = style.long()
            if style_t.dim() == 1:
                style_t = style_t.unsqueeze(1).expand(-1, seq_len)
            style_emb = self.style_proj(self.style_embed(style_t))

        # Instrument → LSTM2
        if instr_context is not None:
            instr_t = instr_context.long()
            if instr_t.dim() == 1:
                instr_t = instr_t.unsqueeze(0)
            instr_raw = self.instrument_embed(instr_t)  # (B, N, D)
            if self.use_instr_attention_pooling:
                instr_emb = self.instr_pooling(instr_raw)  # (B, 1, D)
            else:
                instr_emb = instr_raw.mean(dim=1, keepdim=True)  # (B, 1, D)
            instr_emb = self.instr_proj(instr_emb).expand(batch_size, seq_len, -1)

        # Control → LSTM3
        if control_context is not None:
            cc = control_context.float()
            if cc.dim() == 2:
                cc = cc.unsqueeze(1).expand(-1, seq_len, -1)
            elif cc.dim() == 3 and cc.size(1) == 1:
                cc = cc.expand(-1, seq_len, -1)
            control_emb = self.control_proj(cc)

        # --- LSTM1: Style + base ---
        lstm1_in = x_embed
        if style_emb is not None:
            lstm1_in = lstm1_in + style_emb
        h0 = None if hidden_states is None else hidden_states[0]
        if self.use_checkpointing and self.training:
            out1, h1 = checkpoint(self._lstm1_fwd, lstm1_in, h0, use_reentrant=False)
        else:
            out1, h1 = self.lstm1(lstm1_in, h0)

        # --- LSTM2: Instrument ---
        lstm2_in = out1
        if self.use_cross_attention:
            lstm2_in = self.cross_attn(out1, out1)  # self-attn on LSTM1 output
        if instr_emb is not None:
            lstm2_in = lstm2_in + instr_emb
        h1_state = None if hidden_states is None else hidden_states[1]
        if self.use_checkpointing and self.training:
            out2, h2 = checkpoint(self._lstm2_fwd, lstm2_in, h1_state, use_reentrant=False)
        else:
            out2, h2 = self.lstm2(lstm2_in, h1_state)

        # --- LSTM3: Control ---
        lstm3_in = out2
        if control_emb is not None:
            lstm3_in = lstm3_in + control_emb
        h2_state = None if hidden_states is None else hidden_states[2]
        if self.use_checkpointing and self.training:
            out3, h3 = checkpoint(self._lstm3_fwd, lstm3_in, h2_state, use_reentrant=False)
        else:
            out3, h3 = self.lstm3(lstm3_in, h2_state)

        out3 = self.dropout(out3)

        # Token fusion (Sprint 4)
        if hasattr(self, "token_fusion") and self.token_fusion is not None:
            out3 = self.token_fusion(out3)

        # --- Output ---
        return (
            self.pitch_out(out3),
            self.duration_out(out3),
            self.velocity_out(out3),
            self.instrument_out(out3),
            (h1, h2, h3),
        )

    def compile_model(self, backend: str = "inductor", mode=None):
        """Apply torch.compile for graph optimization.

        Args:
            backend: torch.compile backend ("inductor", "aot_eager", etc.).
            mode: Compilation mode (None, "reduce-overhead", etc.).

        Returns:
            Self for chaining.
        """
        try:
            self.forward = torch.compile(self.forward, backend=backend, mode=mode)
            self._compiled = True
        except Exception:
            pass  # Graceful fallback
        return self


class HLSTMTrainer:
    """Trainer for HEventModel with AMP, LR scheduling, and loss weighting.

    Args:
        model: HEventModel instance.
        lr: Learning rate.
        device: Device string. Auto-detected if None.
        use_amp: Enable mixed precision training.
        loss_weights: Per-head loss weights [pitch, dur, vel, instr].
        scheduler_type: LR scheduler type ("plateau", "cosine", "none").
        scheduler_params: Dict of scheduler kwargs.
        grad_clip_norm: Max gradient norm (0 = disabled).
        weight_decay: L2 regularization strength.

    Attributes:
        model: The HEventModel being trained.
        optimizer: torch.optim.Adam instance.
        scaler: GradScaler for AMP (if enabled).
        scheduler: LR scheduler (if enabled).
    """

    def __init__(
        self,
        model: HEventModel,
        lr: float = 0.001,
        device: Optional[str] = None,
        use_amp: bool = True,
        loss_weights: Optional[List[float]] = None,
        scheduler_type: str = "plateau",
        scheduler_params: Optional[Dict] = None,
        grad_clip_norm: float = 0.0,
        weight_decay: float = 1e-4,
    ):
        # Resolve device
        cfg = load_settings()
        if device is None:
            device = cfg.model.device
            if device == "auto":
                device = "cpu"
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"

        self.device = device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        # AMP — MPS does not support float16 for LSTM weights, so disable.
        self.use_amp = use_amp and device == "cuda"
        self.scaler = torch.amp.GradScaler(device=self.device) if self.use_amp else None
        self.grad_clip_norm = grad_clip_norm

        # Loss weights
        if loss_weights is None:
            loss_weights = [1.0, 1.0, 1.0, 1.0]
        self.loss_weights = loss_weights
        self.loss_weights_sum = sum(loss_weights)

        # LR scheduler
        sp = scheduler_params or {}
        self.scheduler_type = scheduler_type
        self.scheduler = None
        if scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5,
                patience=sp.get("patience", 3), min_lr=sp.get("min_lr", 1e-6),
            )
        elif scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=sp.get("t_max", 30),
                eta_min=sp.get("min_lr", 1e-6),
            )

    def __help__(self) -> None:
        """Print usage information for HLSTMTrainer."""
        print("HLSTMTrainer — H-LSTM training harness")
        print("=" * 50)
        print(f"  Device:    {self.device}")
        print(f"  AMP:       {self.use_amp}")
        print(f"  Scheduler: {self.scheduler_type}")
        print(f"  Loss w:    {self.loss_weights}")
        print()
        print("Methods:")
        print("  train_step(x, y, ...)      — Single training step")
        print("  eval_step(x, y, ...)       — Single evaluation step")
        print("  train(train_loader, val_loader, epochs)  — Full training loop")
        print("  save(path) / load(path)    — Checkpoint I/O")
        print("  get_lr() / scheduler_step(val_loss)  — LR management")

    def _compute_loss(
        self,
        pitch_logits: torch.Tensor,
        dur_logits: torch.Tensor,
        vel_logits: torch.Tensor,
        instr_logits: torch.Tensor,
        y_pitch: torch.Tensor,
        y_dur: torch.Tensor,
        y_vel: torch.Tensor,
        y_instr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted cross-entropy loss for all four heads."""
        pitch_loss = self.criterion(
            pitch_logits.view(-1, pitch_logits.size(-1)), y_pitch.reshape(-1)
        )
        dur_loss = self.criterion(
            dur_logits.view(-1, dur_logits.size(-1)), y_dur.reshape(-1)
        )
        vel_loss = self.criterion(
            vel_logits.view(-1, vel_logits.size(-1)), y_vel.reshape(-1)
        )
        instr_loss = self.criterion(
            instr_logits.view(-1, instr_logits.size(-1)), y_instr.reshape(-1)
        )

        w = self.loss_weights
        total = (w[0] * pitch_loss + w[1] * dur_loss + w[2] * vel_loss + w[3] * instr_loss) / self.loss_weights_sum
        details = {
            "pitch": pitch_loss.item(),
            "duration": dur_loss.item(),
            "velocity": vel_loss.item(),
            "instrument": instr_loss.item(),
        }
        return total, details

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        style: Optional[torch.Tensor] = None,
        instr_context: Optional[torch.Tensor] = None,
        control_context: Optional[torch.Tensor] = None,
    ) -> float:
        """Execute a single training step.

        Args:
            x: Input tokens (B, T, 4).
            y: Target tokens (B, T, 4) — shifted by 1.
            style: Style labels (B,).
            instr_context: Instrument IDs (B, N).
            control_context: Control features (B, T, D).

        Returns:
            Loss value (float).
        """
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)

        def _to_device(t):
            return t.to(self.device) if t is not None else None

        style = _to_device(style)
        instr_context = _to_device(instr_context)
        control_context = _to_device(control_context)

        y_pitch = y[:, :, 0].long()
        y_dur = y[:, :, 1].long()
        y_vel = y[:, :, 2].long()
        y_instr = y[:, :, 3].long()

        self.optimizer.zero_grad()

        if self.use_amp:
            with torch.amp.autocast(device_type=self.device):
                pitch_logits, dur_logits, vel_logits, instr_logits, _ = self.model(
                    x, style=style, instr_context=instr_context,
                    control_context=control_context
                )
                loss, _ = self._compute_loss(
                    pitch_logits, dur_logits, vel_logits, instr_logits,
                    y_pitch, y_dur, y_vel, y_instr,
                )
            self.scaler.scale(loss).backward()
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            pitch_logits, dur_logits, vel_logits, instr_logits, _ = self.model(
                x, style=style, instr_context=instr_context,
                control_context=control_context
            )
            loss, _ = self._compute_loss(
                pitch_logits, dur_logits, vel_logits, instr_logits,
                y_pitch, y_dur, y_vel, y_instr,
            )
            loss.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def eval_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        style: Optional[torch.Tensor] = None,
        instr_context: Optional[torch.Tensor] = None,
        control_context: Optional[torch.Tensor] = None,
    ) -> float:
        """Execute a single evaluation step without gradient computation.

        Args:
            Same as train_step.

        Returns:
            Loss value (float).
        """
        self.model.eval()
        x, y = x.to(self.device), y.to(self.device)

        def _to_device(t):
            return t.to(self.device) if t is not None else None

        style = _to_device(style)
        instr_context = _to_device(instr_context)
        control_context = _to_device(control_context)

        y_pitch = y[:, :, 0].long()
        y_dur = y[:, :, 1].long()
        y_vel = y[:, :, 2].long()
        y_instr = y[:, :, 3].long()

        if self.use_amp:
            with torch.amp.autocast(device_type=self.device):
                pitch_logits, dur_logits, vel_logits, instr_logits, _ = self.model(
                    x, style=style, instr_context=instr_context,
                    control_context=control_context
                )
                loss, _ = self._compute_loss(
                    pitch_logits, dur_logits, vel_logits, instr_logits,
                    y_pitch, y_dur, y_vel, y_instr,
                )
        else:
            pitch_logits, dur_logits, vel_logits, instr_logits, _ = self.model(
                x, style=style, instr_context=instr_context,
                control_context=control_context
            )
            loss, _ = self._compute_loss(
                pitch_logits, dur_logits, vel_logits, instr_logits,
                y_pitch, y_dur, y_vel, y_instr,
            )

        return loss.item()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30,
        output_dir: str = "models",
        log_dir: str = "runs/hlstm",
        patience: int = 7,
        resume_path: Optional[str] = None,
        start_epoch: int = 1,
        callbacks: Optional[List] = None,
    ) -> Dict[str, List[float]]:
        """Run the full training loop with validation and checkpointing.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            num_epochs: Maximum number of epochs.
            output_dir: Directory for checkpoints.
            log_dir: TensorBoard log directory.
            patience: Early stopping patience.
            resume_path: Optional checkpoint path to resume from.
            start_epoch: Starting epoch number.
            callbacks: Optional list of callable(epoch, logs) for custom hooks.

        Returns:
            History dict with keys "train_loss", "val_loss", "perplexity".
        """
        import math
        import os
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

        # Resume
        if resume_path and os.path.exists(resume_path):
            self.load(resume_path)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        history = {"train_loss": [], "val_loss": [], "perplexity": [], "val_perplexity": []}

        for epoch in range(start_epoch, num_epochs + 1):
            train_loss = 0.0
            val_loss = 0.0

            for batch in tqdm(train_loader, desc=f"[Epoch {epoch}] Train"):
                loss = self.train_step(*batch)
                train_loss += loss

            avg_train_loss = train_loss / len(train_loader)
            train_ppl = math.exp(avg_train_loss) if avg_train_loss < 20 else float("inf")

            for batch in tqdm(val_loader, desc=f"[Epoch {epoch}] Val"):
                val_loss += self.eval_step(*batch)

            avg_val_loss = val_loss / len(val_loader)
            val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")

            # LR step
            self.scheduler_step(avg_val_loss)
            current_lr = self.get_lr()

            print(
                f"Epoch {epoch:03d} | Train: {avg_train_loss:.4f} ({train_ppl:.1f} PPL) | "
                f"Val: {avg_val_loss:.4f} ({val_ppl:.1f} PPL) | LR: {current_lr:.2e}"
            )

            # Logging
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            writer.add_scalar("Perplexity/Train", train_ppl, epoch)
            writer.add_scalar("Perplexity/Validation", val_ppl, epoch)
            writer.add_scalar("LR", current_lr, epoch)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["perplexity"].append(train_ppl)
            history["val_perplexity"].append(val_ppl)

            # Checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                ckpt = os.path.join(output_dir, f"hlstm_epoch_{epoch}.pt")
                self.save(ckpt)
                print(f"  ✓ Saved {ckpt}")
            else:
                epochs_no_improve += 1
                ckpt = os.path.join(output_dir, f"hlstm_epoch_{epoch}_not_improved.pt")
                self.save(ckpt)
                print(f"  - No improvement ({epochs_no_improve}/{patience})")

                if epochs_no_improve >= patience:
                    print(f"  ⏹ Early stopping at epoch {epoch}")
                    break

            # Callbacks
            if callbacks:
                for cb in callbacks:
                    cb(epoch, {"train_loss": avg_train_loss, "val_loss": avg_val_loss})

        writer.close()
        return history

    def scheduler_step(self, val_loss: Optional[float] = None) -> None:
        """Advance the LR scheduler by one step."""
        if self.scheduler is None:
            return
        if self.scheduler_type == "plateau" and val_loss is not None:
            self.scheduler.step(val_loss)
        elif self.scheduler_type == "cosine":
            self.scheduler.step()

    def get_lr(self) -> float:
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def save(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save the .pt file.
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to the .pt checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
