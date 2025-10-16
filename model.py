import torch
import torch.nn as nn
import numpy as np


class HEventModel(nn.Module):
    """
    Hierarchical 3-layer LSTM for symbolic music modeling.
    Multi-feature input per note: [pitch, duration, velocity, instrument_id].
    """

    def __init__(self,
                 num_pitches=128,
                 num_durations=128,
                 num_velocities=128,
                 num_instruments=128,
                 style_classes=256,
                 embed_dim=256,
                 hidden_dim=512,
                 dropout=0.3):
        super(HEventModel, self).__init__()

        # Separate embeddings for each feature
        self.pitch_embed = nn.Embedding(num_pitches, embed_dim)
        self.duration_embed = nn.Embedding(num_durations, embed_dim)
        self.velocity_embed = nn.Embedding(num_velocities, embed_dim)
        self.instrument_embed = nn.Embedding(num_instruments, embed_dim)
        self.style_embed = nn.Embedding(style_classes, embed_dim)

        self.use_instrument_conditioning = True
        self.use_style_conditioning = True

        # 3-layer hierarchical LSTM
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.pitch_head = nn.Linear(hidden_dim, self.pitch_embed.num_embeddings)
        self.dur_head = nn.Linear(hidden_dim, self.duration_embed.num_embeddings)
        self.vel_head = nn.Linear(hidden_dim, self.velocity_embed.num_embeddings)
        self.instr_head = nn.Linear(hidden_dim, self.instrument_embed.num_embeddings)
        self.max_bins = num_durations

    def forward(self, x, style=None, hidden_states=None):
        """
        x: (batch, seq_len, 4) -> pitch, duration, velocity, instrument_id
        style: optional style conditioning
        """
        if x.dim() == 2:  # add batch dim if missing
            x = x.unsqueeze(0)
        batch_size, seq_len, _ = x.size()

        pitch_ids = x[:, :, 0]
        dur_ids = x[:, :, 1]
        # dur_ids = np.log1p(x)
        # dur_ids = (dur_ids / dur_ids.max() * (self.max_bins - 1)).astype(int)
        vel_ids = x[:, :, 2]
        instr_ids = x[:, :, 3]

        pitch_emb = self.pitch_embed(pitch_ids)
        dur_emb = self.duration_embed(dur_ids)
        vel_emb = self.velocity_embed(vel_ids)
        instr_emb = self.instrument_embed(instr_ids)

        x_embed = pitch_emb + dur_emb + vel_emb
        if self.use_instrument_conditioning:
            x_embed += instr_emb

        if self.use_style_conditioning and style is not None:
            if style.dim() == 1:
                style_emb = self.style_embed(style).unsqueeze(1).expand(-1, seq_len, -1)
            else:
                style_emb = self.style_embed(style)
            x_embed += style_emb

        out1, h1 = self.lstm1(x_embed, None if hidden_states is None else hidden_states[0])
        out1 = self.dropout(out1)
        out2, h2 = self.lstm2(out1, None if hidden_states is None else hidden_states[1])
        out2 = self.dropout(out2)
        out3, h3 = self.lstm3(out2, None if hidden_states is None else hidden_states[2])
        out3 = self.dropout(out3)

        # Multi-head output predictions
        pitch_logits = self.pitch_head(out3)
        dur_logits = self.dur_head(out3)
        vel_logits = self.vel_head(out3)
        instr_logits = self.instr_head(out3)

        return pitch_logits, dur_logits, vel_logits, instr_logits, (h1, h2, h3)


class HLSTMTrainer:
    """
    Trainer for HEventModel (multi-feature input)
    Handles training, evaluation, loss computation.
    """

    def __init__(self, model, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()  # only for pitch prediction for now

    def train_step(self, x, y, style=None):
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        if style is not None:
            style = style.to(self.device)

        y_pitch = y[:, :, 0]
        y_dur = y[:, :, 1]
        y_vel = y[:, :, 2]
        y_instr = y[:, :, 3]

        self.optimizer.zero_grad()
        pitch_logits, dur_logits, vel_logits, instr_logits, _ = self.model(x, style=style)
        # Predict pitch only (first feature)
        pitch_loss = self.criterion(pitch_logits.view(-1, pitch_logits.size(-1)), y_pitch.reshape(-1))
        dur_loss = self.criterion(dur_logits.view(-1, dur_logits.size(-1)), y_dur.reshape(-1))
        vel_loss = self.criterion(vel_logits.view(-1, vel_logits.size(-1)), y_vel.reshape(-1))
        instr_loss = self.criterion(instr_logits.view(-1, instr_logits.size(-1)), y_instr.reshape(-1))

        loss = (pitch_loss + dur_loss + vel_loss + instr_loss) / 4

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, x, y, style=None):
        self.model.eval()
        x, y = x.to(self.device), y.to(self.device)
        if style is not None:
            style = style.to(self.device)

        y_pitch = y[:, :, 0]
        y_dur = y[:, :, 1]
        y_vel = y[:, :, 2]
        y_instr = y[:, :, 3]

        self.optimizer.zero_grad()
        pitch_logits, dur_logits, vel_logits, instr_logits, _ = self.model(x, style=style)
        # Predict pitch only (first feature)
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
