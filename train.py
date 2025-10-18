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
class MusicDataset(Dataset):
    def __init__(self, token_path, seq_len=128):
        with open(token_path, "rb") as f:
            data_list = pickle.load(f)

        self.note_events = []
        self.style_levels = []
        self.instr_levels = []
        self.control_levels = []

        for song in data_list:
            note_seq = np.array(song["note_level"], dtype=np.int64)
            instr_seq = np.array(song.get("instr_level", [0]), dtype=np.int64)
            control_seq = np.array(song.get("control_level", [0]), dtype=np.int64)
            song_seq = song.get("song_level", [0])

            self.note_events.extend(note_seq)
            self.instr_levels.extend([instr_seq[0]] * len(note_seq))
            if len(control_seq) > 0:
                self.control_levels.extend([control_seq[0]] * len(note_seq))
            self.style_levels.extend([song_seq[4]] * len(note_seq))

        self.seq_len = seq_len

    def __len__(self):
        return len(self.note_events) - self.seq_len

    def __getitem__(self, idx):
        x = np.array(self.note_events[idx:idx + self.seq_len])
        y = np.array(self.note_events[idx + 1:idx + self.seq_len + 1])

        style = self.style_levels[idx]
        instr = self.instr_levels[idx]
        if len(self.control_levels) > 0:
            control = self.control_levels[idx]
            control_level = torch.tensor(control, dtype=torch.float32)
        else:
            control_level = torch.zeros((1, x.shape[0]), dtype=torch.float32)

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(style, dtype=torch.long),
            torch.tensor(instr, dtype=torch.long),
            control_level,
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
        token_path,
        output_dir="models",
        log_dir="runs/hlstm",
        seq_len=128,
        batch_size=32,
        num_epochs=30,
        lr=1e-3,
        val_split=0.2,
        resume_checkpoint=None,
        patience=3
):
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    dataset = MusicDataset(token_path, seq_len)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

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
    print("=" * 60)

    trainer = HLSTMTrainer(model, lr=lr, device="mps")

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
        for x, y, style, instruments, control_values in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
            loss = trainer.train_step(x, y, style=style, instr_context=instruments, control_context=control_values)
            train_loss += loss

        avg_train_loss = train_loss / len(train_loader)
        train_metrics = calculate_metrics(avg_train_loss)

        # -------------------
        # Validation Loop
        # -------------------
        for x, y, style, instruments, control_values in tqdm(val_loader, desc=f"[Epoch {epoch}] Validation"):
            val_loss += trainer.eval_step(x, y, style=style, instr_context=instruments, control_context=control_values)

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = calculate_metrics(avg_val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train PPL: {train_metrics['perplexity']:.2f} | Val PPL: {val_metrics['perplexity']:.2f}")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Perplexity/Train", train_metrics["perplexity"], epoch)
        writer.add_scalar("Perplexity/Validation", val_metrics["perplexity"], epoch)

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
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping triggered after {patience} epochs of no improvement.")
                break

    writer.close()
    print("🎵 Training complete!")


if __name__ == "__main__":
    train_model(
        "data/encoded/encoded_tokens_control_instrument_context.pkl"
    )