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
    def __init__(self, token_path=Config.TOKEN_PATH, seq_len=Config.SEQ_LEN, use_control_context=True, control_dim=Config.CONTROL_DIM):
        with open(token_path, "rb") as f:
            data_list = pickle.load(f)

        self.note_events = []
        self.style_levels = []
        self.instr_levels = []
        self.control_levels = []
        self.use_control_context = use_control_context
        self.control_dim = control_dim
        self.seq_len = seq_len

        for song in data_list:
            note_seq = np.array(song["note_level"], dtype=np.int64)
            control_seq = song.get("control_level", np.zeros((0, 4), dtype=np.int64))
            song_seq = song.get("song_level", [120, 4, 1, 100, 0])

            # Process control changes - now returns features for entire sequence
            control_features = self._extract_control_features(control_seq, len(note_seq))

            for i in range(len(note_seq) - seq_len):
                self.note_events.append(note_seq[i:i + seq_len])
                self.style_levels.append(song_seq[4])
                self.instr_levels.append(song_seq[2])

                if self.use_control_context and len(control_features) > 0:
                    # Get control features for this sequence window
                    start_idx = i
                    end_idx = i + seq_len
                    control_window = control_features[start_idx:end_idx]

                    # If we don't have enough control features, pad with zeros
                    if len(control_window) < seq_len:
                        padding = np.zeros((seq_len - len(control_window), self.control_dim), dtype=np.float32)
                        control_window = np.concatenate([control_window, padding])

                    self.control_levels.append(control_window)
                else:
                    # Create zero tensor with shape (seq_len, control_dim)
                    self.control_levels.append(np.zeros((seq_len, self.control_dim), dtype=np.float32))

    def _extract_control_features(self, control_seq, num_notes):
        """Extract control features per note with dimension control_dim"""
        if len(control_seq) == 0:
            return np.zeros((num_notes, self.control_dim), dtype=np.float32)

        control_features = []

        # Define control codes we want to track
        # You can expand this list based on what's in your data
        control_codes = [1, 7, 11, 64, 71, 74]  # modulation, volume, expression, sustain, resonance, brightness

        for i in range(num_notes):
            control_feat = np.zeros(self.control_dim, dtype=np.float32)

            # Fill in the first few dimensions with actual control values
            for j, cc_code in enumerate(control_codes):
                if j >= self.control_dim:
                    break  # Don't exceed our feature dimension

                cc_events = control_seq[control_seq[:, 2] == cc_code]
                if len(cc_events) > 0:
                    # Use the most recent value
                    control_feat[j] = cc_events[-1, 3] / 127.0  # normalize
                else:
                    # Set reasonable defaults
                    if cc_code == 7:  # volume
                        control_feat[j] = 0.8
                    elif cc_code == 11:  # expression
                        control_feat[j] = 0.7
                    else:
                        control_feat[j] = 0.0

            # Remaining dimensions can be used for other features or left as zero
            control_features.append(control_feat)

        return np.array(control_features)

    def __len__(self):
        return len(self.note_events)

    def __getitem__(self, idx):
        x = self.note_events[idx]  # shape: (seq_len, 4)
        y = np.concatenate([self.note_events[idx][1:], self.note_events[idx][-1:]])

        style = self.style_levels[idx]
        instr_context = np.array([self.instr_levels[idx]], dtype=np.int64)
        control_context = self.control_levels[idx]  # shape: (seq_len, control_dim)

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

    trainer = HLSTMTrainer(model, lr=lr, device=Config.DEVICE)

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
