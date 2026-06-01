import torch
import torch.nn as nn
import numpy as np
from config import Config

class HEventModel(nn.Module):
    """
    Hierarchical 3-layer LSTM for symbolic music modeling.
    Multi-feature input per note: [pitch, duration, velocity, instrument_id].
    """

    def __init__(self,
                 num_pitches=Config.NUM_PITCHES,
                 num_durations=Config.NUM_DURATIONS,
                 num_velocities=Config.NUM_VELOCITIES,
                 num_instruments=Config.NUM_INSTRUMENTS,
                 style_classes=Config.STYLE_CLASSES,
                 embed_dim=Config.EMBED_DIM,
                 control_dim=Config.CONTROL_DIM,
                 hidden_dim=Config.HIDDEN_DIM,
                 dropout=Config.DROPOUT,
                 device=Config.DEVICE):
        super(HEventModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # Separate embeddings for each feature
        self.pitch_embed = nn.Embedding(num_pitches, embed_dim, device=device)
        self.duration_embed = nn.Embedding(num_durations, embed_dim, device=device)
        self.velocity_embed = nn.Embedding(num_velocities, embed_dim, device=device)
        self.instrument_embed = nn.Embedding(num_instruments, embed_dim, device=device)
        self.style_embed = nn.Embedding(style_classes, embed_dim, device=device)

        # Projection layers to match dimensions
        self.embed_proj = nn.Linear(embed_dim, hidden_dim, device=device)  # Project embeddings to hidden_dim
        self.style_proj = nn.Linear(embed_dim, hidden_dim, device=device)
        self.instr_proj = nn.Linear(embed_dim, hidden_dim, device=device)
        self.control_proj = nn.Linear(control_dim, hidden_dim, device=device)

        # 3-layer hierarchical LSTM
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, device=device)  # Changed input to hidden_dim
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, device=device)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, device=device)

        self.dropout = nn.Dropout(dropout)

        # Output layers
        self.pitch_out = nn.Linear(hidden_dim, num_pitches, device=device)
        self.duration_out = nn.Linear(hidden_dim, num_durations, device=device)
        self.velocity_out = nn.Linear(hidden_dim, num_velocities, device=device)
        self.instrument_out = nn.Linear(hidden_dim, num_instruments, device=device)

        self.max_bins = num_durations

        self.use_style_conditioning = True
        self.use_instrument_conditioning = True
        self.use_control_conditioning = True

        self.control_dim = control_dim

    def forward(self, x, style=None, instr_context=None, control_context=None, hidden_states=None):
        batch_size, seq_len, _ = x.size()
        device = x.device

        # --- Extract input components ---
        pitch = x[:, :, 0].long()
        duration = x[:, :, 1].long()
        velocity = x[:, :, 2].long()
        instrument = x[:, :, 3].long()

        # --- Base embeddings ---
        x_embed = (
                self.pitch_embed(pitch) +
                self.duration_embed(duration) +
                self.velocity_embed(velocity) +
                self.instrument_embed(instrument)
        )
        x_embed = self.embed_proj(x_embed)

        # --- Prepare conditioning contexts ---
        style_emb = None
        instr_emb = None
        control_emb = None

        # STYLE conditioning (global, high-level) - LSTM1
        if style is not None and self.use_style_conditioning:
            style = style.long()
            if style.dim() == 1:
                style = style.unsqueeze(1).expand(-1, seq_len)
            style_emb = self.style_embed(style)
            style_emb = self.style_proj(style_emb)

        # INSTRUMENT conditioning (mid-level) - LSTM2
        if instr_context is not None and self.use_instrument_conditioning:
            instr_context = instr_context.long()
            if instr_context.dim() == 1:
                instr_context = instr_context.unsqueeze(0)
            instr_emb = self.instrument_embed(instr_context).mean(dim=1, keepdim=True)
            instr_emb = self.instr_proj(instr_emb)
            instr_emb = instr_emb.expand(batch_size, seq_len, -1)

        # CONTROL conditioning - handle the shape properly
        if control_context is not None and self.use_control_conditioning:
            # control_context shape: (batch_size, seq_len, control_dim)
            if control_context.dim() == 2:
                # If it's (batch_size, control_dim), expand to sequence
                control_context = control_context.unsqueeze(1).expand(-1, seq_len, -1)
            elif control_context.dim() == 3 and control_context.size(1) == 1:
                # If it's (batch_size, 1, control_dim), expand to sequence length
                control_context = control_context.expand(-1, seq_len, -1)

            # Project control context to hidden dimension
            control_emb = self.control_proj(control_context.float())

        # --- Hierarchical LSTM flow ---
        # LSTM1 → STYLE conditioning (global characteristics)
        lstm1_in = x_embed
        if style_emb is not None:
            lstm1_in = lstm1_in + style_emb
        out1, h1 = self.lstm1(lstm1_in, None if hidden_states is None else hidden_states[0])

        # LSTM2 → INSTRUMENT conditioning (instrument-specific patterns)
        lstm2_in = out1
        if instr_emb is not None:
            lstm2_in = lstm2_in + instr_emb
        out2, h2 = self.lstm2(lstm2_in, None if hidden_states is None else hidden_states[1])

        # LSTM3 → CONTROL conditioning (expression, dynamics)
        lstm3_in = out2
        if control_emb is not None:
            lstm3_in = lstm3_in + control_emb
        out3, h3 = self.lstm3(lstm3_in, None if hidden_states is None else hidden_states[2])

        out3 = self.dropout(out3)

        # --- Output projections ---
        pitch_logits = self.pitch_out(out3)
        dur_logits = self.duration_out(out3)
        vel_logits = self.velocity_out(out3)
        instr_logits = self.instrument_out(out3)

        return pitch_logits, dur_logits, vel_logits, instr_logits, (h1, h2, h3)


class HLSTMTrainer:
    def __init__(self, model, lr=1e-3, device=None):
        self.device = device or Config.DEVICE if device is None else device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, x, y, style=None, instr_context=None, control_context=None):
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        if style is not None:
            style = style.to(self.device)
        if instr_context is not None:
            instr_context = instr_context.to(self.device)
        if control_context is not None:
            control_context = control_context.to(self.device)

        y_pitch = y[:, :, 0].long()
        y_dur = y[:, :, 1].long()
        y_vel = y[:, :, 2].long()
        y_instr = y[:, :, 3].long()

        self.optimizer.zero_grad()
        pitch_logits, dur_logits, vel_logits, instr_logits, _ = self.model(
            x, style=style, instr_context=instr_context, control_context=control_context
        )

        pitch_loss = self.criterion(pitch_logits.view(-1, pitch_logits.size(-1)), y_pitch.reshape(-1))
        dur_loss = self.criterion(dur_logits.view(-1, dur_logits.size(-1)), y_dur.reshape(-1))
        vel_loss = self.criterion(vel_logits.view(-1, vel_logits.size(-1)), y_vel.reshape(-1))
        instr_loss = self.criterion(instr_logits.view(-1, instr_logits.size(-1)), y_instr.reshape(-1))
        loss = (pitch_loss + dur_loss + vel_loss + instr_loss) / 4

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, x, y, style=None, instr_context=None, control_context=None):
        self.model.eval()
        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)
            if style is not None:
                style = style.to(self.device)
            if instr_context is not None:
                instr_context = instr_context.to(self.device)
            if control_context is not None:
                control_context = control_context.to(self.device)

            y_pitch = y[:, :, 0].long()
            y_dur = y[:, :, 1].long()
            y_vel = y[:, :, 2].long()
            y_instr = y[:, :, 3].long()

            pitch_logits, dur_logits, vel_logits, instr_logits, _ = self.model(
                x, style=style, instr_context=instr_context, control_context=control_context
            )

            pitch_loss = self.criterion(pitch_logits.view(-1, pitch_logits.size(-1)), y_pitch.reshape(-1))
            dur_loss = self.criterion(dur_logits.view(-1, dur_logits.size(-1)), y_dur.reshape(-1))
            vel_loss = self.criterion(vel_logits.view(-1, vel_logits.size(-1)), y_vel.reshape(-1))
            instr_loss = self.criterion(instr_logits.view(-1, instr_logits.size(-1)), y_instr.reshape(-1))
            loss = (pitch_loss + dur_loss + vel_loss + instr_loss) / 4

        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == "__main__":
    model = HEventModel(device='cpu')

    x = torch.randint(0, 128, (32, 128, 4))  # note-level data
    style = torch.randint(0, 2, (32, 1))  # 2 style classes
    instr_context = torch.randint(0, 128, (32, 128))  # instrument context
    control_context = torch.randn(32, 128, 128)  # continuous control data

    summary(model,
            input_data=(x, style, instr_context, control_context), col_names=["input_size","output_size","num_params","params_percent","kernel_size", "mult_adds", "trainable"])
