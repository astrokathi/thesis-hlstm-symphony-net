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
# Dataset for 4-feature
# -----------------------
class MusicDataset(Dataset):
    def __init__(self, token_path, seq_len=128):
        with open(token_path, "rb") as f:
            data_list = pickle.load(f)

        self.events = []
        self.styles = []
        for song in data_list:
            note_seq = np.array(song["note_level"], dtype=np.int64)
            song_seq = song.get("song_level", [0])
            self.events.extend(note_seq)
            self.styles.extend([song_seq[0]] * len(note_seq))

        self.seq_len = seq_len

    def __len__(self):
        return len(self.events) - self.seq_len

    def __getitem__(self, idx):
        x = np.array(self.events[idx:idx + self.seq_len])
        y = np.array(self.events[idx + 1:idx + self.seq_len + 1])
        style = self.styles[idx]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(style, dtype=torch.long)
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
def train_model(token_path, output_dir="models", log_dir="runs/hlstm", seq_len=128,
                batch_size=32, num_epochs=30, lr=1e-3, val_split=0.2, resume_checkpoint=None):
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

    # Resume
    start_epoch = 1
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        trainer.load(resume_checkpoint)
        start_epoch = int(resume_checkpoint.split("_")[-1].split(".")[0]) + 1

    # Training
    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = 0.0
        val_loss = 0.0

        for x, y, style in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
            # print(f"X is {x} and Y is {y}")
            loss = trainer.train_step(x, y, style)
            train_loss += loss

        avg_train_loss = train_loss / len(train_loader)
        train_metrics = calculate_metrics(avg_train_loss)

        for x, y, style in tqdm(val_loader, desc=f"[Epoch {epoch}] Validation"):
            val_loss += trainer.eval_step(x, y, style)
        avg_val_loss = val_loss / len(val_loader)
        val_metrics = calculate_metrics(avg_val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train PPL: {train_metrics['perplexity']:.2f} | Val PPL: {val_metrics['perplexity']:.2f}")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Perplexity/Train", train_metrics["perplexity"], epoch)
        writer.add_scalar("Perplexity/Validation", val_metrics["perplexity"], epoch)

        ckpt_path = os.path.join(output_dir, f"hlstm_epoch_{epoch}.pt")
        trainer.save(ckpt_path)
        print(f"✅ Checkpoint saved: {ckpt_path}")

    writer.close()
    print("🎵 Training complete!")


if __name__ == "__main__":
    train_model("data/encoded/encoded_tokens.pkl")
