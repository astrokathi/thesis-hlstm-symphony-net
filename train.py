from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pickle
import numpy as np
import os
from tqdm import tqdm
from model import HEventModel
from model import HLSTMTrainer
import math
from torch.utils.tensorboard import SummaryWriter


# -----------------------
# Dataset for hierarchical 4-feature input
# -----------------------
from config import Config

class MusicDataset(Dataset):
    """Sprint 2: On-the-fly sliding window dataset. Stores per-song arrays
    instead of pre-materializing every window — O(n) memory vs O(n·seq_len)."""

    def __init__(self, token_path=Config.TOKEN_PATH, seq_len=Config.SEQ_LEN,
                 use_control_context=True, control_dim=Config.CONTROL_DIM):
        with open(token_path, "rb") as f:
            data_list = pickle.load(f)

        self.use_control_context = use_control_context
        self.control_dim = control_dim
        self.seq_len = seq_len

        # Per-song storage: each entry is (note_seq, style, instr, control_features)
        self.songs = []
        self.cum_lengths = [0]  # prefix sum of (len(note_seq) - seq_len) per song

        for song in data_list:
            note_seq = np.array(song["note_level"], dtype=np.int64)
            if len(note_seq) <= seq_len:
                continue  # skip songs too short

            control_seq = song.get("control_level", np.zeros((0, 4), dtype=np.int64))
            song_seq = song.get("song_level", [120, 4, 1, 100, 0])

            # Pre-compute control features once per song (not per window)
            control_features = self._extract_control_features(control_seq, len(note_seq))

            self.songs.append({
                "note_seq": note_seq,
                "style": song_seq[4],
                "instr": song_seq[2],
                "control_features": control_features,
            })
            self.cum_lengths.append(self.cum_lengths[-1] + len(note_seq) - seq_len)

    def _extract_control_features(self, control_seq, num_notes):
        """Extract control features per note with dimension control_dim"""
        if len(control_seq) == 0:
            return np.zeros((num_notes, self.control_dim), dtype=np.float32)

        # Define control codes we want to track
        control_codes = [1, 7, 11, 64, 71, 74]

        # Build a full array then fill in values per note
        control_features = np.zeros((num_notes, self.control_dim), dtype=np.float32)
        for j, cc_code in enumerate(control_codes):
            if j >= self.control_dim:
                break
            cc_events = control_seq[control_seq[:, 2] == cc_code]
            if len(cc_events) > 0:
                # Use the most recent value for all notes (broadcast)
                control_features[:, j] = cc_events[-1, 3] / 127.0
            else:
                # Set reasonable defaults
                if cc_code == 7:
                    control_features[:, j] = 0.8
                elif cc_code == 11:
                    control_features[:, j] = 0.7

        return control_features

    def _get_song_and_offset(self, idx):
        """Map flat index to (song_index, offset_within_song)."""
        for song_idx in range(len(self.songs)):
            if idx < self.cum_lengths[song_idx + 1]:
                offset = idx - self.cum_lengths[song_idx]
                return song_idx, offset
        # Fallback (shouldn't happen with valid idx)
        return len(self.songs) - 1, 0

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, idx):
        song_idx, offset = self._get_song_and_offset(idx)
        song = self.songs[song_idx]

        x = song["note_seq"][offset:offset + self.seq_len]
        y = np.concatenate([x[1:], x[-1:]])  # shifted by 1

        style = np.array(song["style"], dtype=np.int64)
        instr_context = np.array([song["instr"]], dtype=np.int64)
        control_context = song["control_features"][offset:offset + self.seq_len]

        # Pad control context if needed (last window may be short)
        if len(control_context) < self.seq_len:
            pad = np.zeros((self.seq_len - len(control_context), self.control_dim), dtype=np.float32)
            control_context = np.concatenate([control_context, pad])

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(style, dtype=torch.long),
            torch.tensor(instr_context, dtype=torch.long),
            torch.tensor(control_context, dtype=torch.float)
        )


# -----------------------
# Metrics
# -----------------------
def calculate_metrics(loss):
    perplexity = math.exp(loss) if loss < 20 else float("inf")
    return {"perplexity": perplexity}


# -----------------------
# Training loop
# -----------------------
def train_model(
        token_path=Config.TOKEN_PATH,
        output_dir=Config.OUTPUT_DIR,
        log_dir=Config.LOG_DIR,
        seq_len=Config.SEQ_LEN,
        batch_size=Config.BATCH_SIZE,
        num_epochs=Config.NUM_EPOCHS,
        lr=Config.LR,
        val_split=Config.VAL_SPLIT,
        resume_checkpoint=None,
        patience=Config.PATIENCE
):
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    dataset = MusicDataset(token_path, seq_len, use_control_context=True)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # --- Sprint 1: Multi-threaded data loading ---
    num_workers = Config.NUM_WORKERS
    pin_memory = Config.PIN_MEMORY
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    model = HEventModel()
    vocab_size = (
            model.pitch_embed.num_embeddings +
            model.duration_embed.num_embeddings +
            model.velocity_embed.num_embeddings +
            model.instrument_embed.num_embeddings
    )
    print(f"[MODEL INFO]")
    print(f"  Pitch: {model.pitch_embed.num_embeddings}")
    print(f"  Duration: {model.duration_embed.num_embeddings}")
    print(f"  Velocity: {model.velocity_embed.num_embeddings}")
    print(f"  Instrument: {model.instrument_embed.num_embeddings}")
    print(f"  --> Effective combined vocab size: {vocab_size:,}")
    print(f"  DataLoader: num_workers={num_workers}, pin_memory={pin_memory}")
    print(f"  AMP: {Config.USE_AMP},  LR scheduler: {Config.LR_SCHEDULER}")
    print("=" * 60)

    # --- Sprint 1: Parse loss weights from config ---
    loss_weights = [float(w) for w in Config.LOSS_WEIGHTS.split(",")]
    trainer = HLSTMTrainer(
        model, lr=lr, device=Config.DEVICE,
        use_amp=Config.USE_AMP, loss_weights=loss_weights,
        scheduler_type=Config.LR_SCHEDULER, lr_patience=Config.LR_PATIENCE,
        lr_min=Config.LR_MIN, grad_clip_norm=Config.GRAD_CLIP_NORM
    )

    # Resume checkpoint if provided
    start_epoch = 1
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        trainer.load(resume_checkpoint)
        start_epoch = int(resume_checkpoint.split("_")[-1].split(".")[0]) + 1

    # ---- Early Stopping Variables ----
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # ---- Training ----
    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = 0.0
        val_loss = 0.0

        # -------------------
        # Training Loop
        # -------------------
        for x, y, style, instruments, controls in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
            loss = trainer.train_step(x, y, style=style, instr_context=instruments, control_context=controls)
            train_loss += loss

        avg_train_loss = train_loss / len(train_loader)
        train_metrics = calculate_metrics(avg_train_loss)

        # -------------------
        # Validation Loop
        # -------------------
        for x, y, style, instruments, controls in tqdm(val_loader, desc=f"[Epoch {epoch}] Validation"):
            val_loss += trainer.eval_step(x, y, style=style, instr_context=instruments, control_context=controls)

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = calculate_metrics(avg_val_loss)

        # Sprint 1: LR scheduler step
        trainer.scheduler_step(avg_val_loss)
        current_lr = trainer.get_lr()

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train PPL: {train_metrics['perplexity']:.2f} | Val PPL: {val_metrics['perplexity']:.2f} | "
              f"LR: {current_lr:.2e}")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Perplexity/Train", train_metrics["perplexity"], epoch)
        writer.add_scalar("Perplexity/Validation", val_metrics["perplexity"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # -------------------
        # Checkpoint & Early Stopping
        # -------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(output_dir, f"hlstm_epoch_{epoch}.pt")
            trainer.save(ckpt_path)
            print(f"✅ Validation improved. Checkpoint saved: {ckpt_path}")
        else:
            epochs_no_improve += 1
            ckpt_path = os.path.join(output_dir, f"hlstm_epoch_{epoch}_not_improved.pt")
            trainer.save(ckpt_path)
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping triggered after {patience} epochs of no improvement.")
                break

    writer.close()
    print("🎵 Training complete!")


if __name__ == "__main__":
    train_model(
        token_path=Config.TOKEN_PATH, resume_checkpoint="models/hlstm_epoch_6.pt"
    )
