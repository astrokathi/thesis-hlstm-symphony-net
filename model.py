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
                 style_classes=2,
                 embed_dim=256,
                 control_dim=128,
                 hidden_dim=512,
                 dropout=0.3,
                 device='mps'):
        super(HEventModel, self).__init__()

        # Separate embeddings for each feature
        self.pitch_embed = nn.Embedding(num_pitches, embed_dim, device=device)
        self.duration_embed = nn.Embedding(num_durations, embed_dim, device=device)
        self.velocity_embed = nn.Embedding(num_velocities, embed_dim, device=device)
        self.instrument_embed = nn.Embedding(num_instruments, embed_dim, device=device)
        self.style_embed = nn.Embedding(style_classes, embed_dim, device=device)
        self.control_embed = nn.Linear(control_dim, embed_dim, device=device)

        self.use_instrument_conditioning = True
        self.use_style_conditioning = True

        # 3-layer hierarchical LSTM
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True, device=device)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, device=device)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, device=device)

        self.dropout = nn.Dropout(dropout)
        self.pitch_head = nn.Linear(hidden_dim, self.pitch_embed.num_embeddings, device=device)
        self.dur_head = nn.Linear(hidden_dim, self.duration_embed.num_embeddings, device=device)
        self.vel_head = nn.Linear(hidden_dim, self.velocity_embed.num_embeddings, device=device)
        self.instr_head = nn.Linear(hidden_dim, self.instrument_embed.num_embeddings, device=device)
        self.max_bins = num_durations

    def forward(self, x, style=None, instr_context=None, control_context=None, hidden_states=None):
        """
        x: (batch, seq_len, 4)
        style: (batch,) optional
        instr_context: list or tensor of instrument IDs per song
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size, seq_len, _ = x.size()

        pitch_ids = x[:, :, 0]
        dur_ids = x[:, :, 1]
        vel_ids = x[:, :, 2]
        instr_ids = x[:, :, 3]

        # Token embeddings
        pitch_emb = self.pitch_embed(pitch_ids)
        dur_emb = self.duration_embed(dur_ids)
        vel_emb = self.velocity_embed(vel_ids)
        instr_emb = self.instrument_embed(instr_ids)
        control_emb = self.control_embed(control_context)

        x_embed = pitch_emb + dur_emb + vel_emb
        if self.use_instrument_conditioning:
            x_embed += instr_emb

        # 🔸 Instrument-level context conditioning
        if instr_context is not None:
            # Convert list to tensor
            if isinstance(instr_context, list):
                instr_context = torch.tensor(instr_context, dtype=torch.long, device=x.device)

            # Ensure shape is [batch_size, num_instrs]
            if instr_context.dim() == 1:
                instr_context = instr_context.unsqueeze(1)  # [batch_size, 1]

            # Compute mean embedding across instruments
            instr_ctx_emb = self.instrument_embed(instr_context).mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]

            # Expand along sequence dimension
            instr_ctx_emb = instr_ctx_emb.expand(batch_size, seq_len, -1)  # [batch_size, seq_len, embed_dim]
            x_embed += instr_ctx_emb

        # 🔸 Style conditioning
        if self.use_style_conditioning and style is not None:
            if style.dim() == 1:
                style_emb = self.style_embed(style).unsqueeze(1).expand(-1, seq_len, -1)
            else:
                style_emb = self.style_embed(style)
            x_embed += style_emb

        # 🔸 Control-level context conditioning
        if control_context is not None:
            # control_context: [batch, seq_len, control_dim] or [batch, control_dim]
            if control_context.dim() == 2:
                # Expand to sequence length if only one vector per batch
                control_context = control_context.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, control_dim]
            # Project control features to embedding dimension
            control_context = control_context.float()
            control_emb = self.control_embed(control_context)  # [batch, seq_len, embed_dim]

            # Add to x_embed (or concatenate if you want)
        x_embed = x_embed + control_emb

        # Forward through hierarchical LSTMs
        out1, h1 = self.lstm1(x_embed, None if hidden_states is None else hidden_states[0])
        out2, h2 = self.lstm2(out1, None if hidden_states is None else hidden_states[1])
        out3, h3 = self.lstm3(out2, None if hidden_states is None else hidden_states[2])
        out3 = self.dropout(out3)

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

    def train_step(self, x, y, style=None, instr_context=None, control_context=None):
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        if style is not None:
            style = style.to(self.device)
        if instr_context is not None:
            instr_context = instr_context.to(self.device)
        if control_context is not None:
            control_context = control_context.to(self.device)

        y_pitch = y[:, :, 0]
        y_dur = y[:, :, 1]
        y_vel = y[:, :, 2]
        y_instr = y[:, :, 3]

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
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        if style is not None:
            style = style.to(self.device)
        if instr_context is not None:
            instr_context = instr_context.to(self.device)
        if control_context is not None:
            control_context = control_context.to(self.device)

        y_pitch = y[:, :, 0]
        y_dur = y[:, :, 1]
        y_vel = y[:, :, 2]
        y_instr = y[:, :, 3]

        self.optimizer.zero_grad()
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
