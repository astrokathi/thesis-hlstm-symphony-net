import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Sprint 4.1 — Lightweight cross-attention between hierarchical layers
# ---------------------------------------------------------------------------
class CrossAttention(nn.Module):
    """Scaled dot-product cross-attention: query attends to key_value."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        B, Tq, D = query.shape
        _, Tkv, _ = key_value.shape

        Q = self.q_proj(query).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, Tkv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, Tkv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, Tq, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Sprint 4.2 — Attention pooling for instrument context (replaces mean)
# ---------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """Learned query-based attention pooling over instrument embeddings."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return out  # (B, 1, D)


# ---------------------------------------------------------------------------
# Sprint 4.3 — Joint token fusion MLP (couples pitch/dur/vel/instr heads)
# ---------------------------------------------------------------------------
class TokenFusion(nn.Module):
    """Lightweight residual MLP that couples the four output heads."""

    def __init__(self, hidden_dim: int, fusion_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)  # residual connection


# ===========================================================================
# Main H-LSTM model
# ===========================================================================
class HEventModel(nn.Module):
    """
    Hierarchical 3-layer LSTM for symbolic music modeling.
    Multi-feature input per note: [pitch, duration, velocity, instrument_id].

    Sprint 3 upgrades:
      - Optional bar/beat position encoding (5th channel in x when enabled)
      - Gradient checkpointing for memory-efficient long-sequence training
      - torch.compile support via .compile_model()

    Sprint 4 upgrades:
      - Cross-attention between LSTM1 and LSTM2 (Thesis §6.6)
      - Attention pooling for instrument context (Thesis §3.6.3)
      - Joint token fusion MLP for coupled logits (Thesis §4.12)
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
                 device=Config.DEVICE,
                 use_position_encoding=False,
                 num_positions=64,
                 use_checkpointing=False,
                 use_cross_attention=False,
                 use_instr_attention_pooling=False,
                 use_token_fusion=False):
        super(HEventModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.use_position_encoding = use_position_encoding
        self.use_checkpointing = use_checkpointing
        self.use_cross_attention = use_cross_attention
        self.use_token_fusion = use_token_fusion

        # Separate embeddings for each feature
        self.pitch_embed = nn.Embedding(num_pitches, embed_dim, device=device)
        self.duration_embed = nn.Embedding(num_durations, embed_dim, device=device)
        self.velocity_embed = nn.Embedding(num_velocities, embed_dim, device=device)
        self.instrument_embed = nn.Embedding(num_instruments, embed_dim, device=device)
        self.style_embed = nn.Embedding(style_classes, embed_dim, device=device)

        # Sprint 3: Bar/beat position encoding
        if use_position_encoding:
            self.position_embed = nn.Embedding(num_positions, embed_dim // 4, device=device)
            self.position_proj = nn.Linear(embed_dim // 4, hidden_dim, device=device)
            self.num_positions = num_positions

        # Projection layers
        self.embed_proj = nn.Linear(embed_dim, hidden_dim, device=device)
        self.style_proj = nn.Linear(embed_dim, hidden_dim, device=device)
        self.instr_proj = nn.Linear(embed_dim, hidden_dim, device=device)
        self.control_proj = nn.Linear(control_dim, hidden_dim, device=device)

        # 3-layer hierarchical LSTM
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, device=device)
        self.lstm2_input_proj = None
        if use_cross_attention:
            # If cross-attention, LSTM2 input may differ from hidden_dim after attention
            self.cross_attn = CrossAttention(hidden_dim, num_heads=4, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, device=device)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, device=device)

        self.dropout = nn.Dropout(dropout)

        # Sprint 4.3: Joint token fusion
        self.token_fusion = None
        if use_token_fusion:
            self.token_fusion = TokenFusion(hidden_dim, fusion_dim=256, dropout=dropout)

        # Output layers
        self.pitch_out = nn.Linear(hidden_dim, num_pitches, device=device)
        self.duration_out = nn.Linear(hidden_dim, num_durations, device=device)
        self.velocity_out = nn.Linear(hidden_dim, num_velocities, device=device)
        self.instrument_out = nn.Linear(hidden_dim, num_instruments, device=device)

        self.max_bins = num_durations

        self.use_style_conditioning = True
        self.use_instrument_conditioning = True
        self.use_control_conditioning = True
        self.use_instr_attention_pooling = use_instr_attention_pooling

        # Sprint 4.2: Attention pooling for instrument context
        self.instr_pooling = None
        if use_instr_attention_pooling:
            self.instr_pooling = AttentionPooling(hidden_dim, num_heads=4)

        self.control_dim = control_dim
        self._compiled = False

    # ---- Forward helpers for checkpointing ----

    def _lstm1_forward(self, x, h):
        return self.lstm1(x, h)

    def _lstm2_forward(self, x, h):
        return self.lstm2(x, h)

    def _lstm3_forward(self, x, h):
        return self.lstm3(x, h)

    def forward(self, x, style=None, instr_context=None, control_context=None, hidden_states=None):
        batch_size, seq_len, n_features = x.size()
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

        # Sprint 3: Position encoding (5th channel)
        if self.use_position_encoding and n_features >= 5:
            beat_pos = x[:, :, 4].long().clamp(0, self.num_positions - 1)
            pos_emb = self.position_embed(beat_pos)
            pos_emb = self.position_proj(pos_emb)
            x_embed = x_embed + pos_emb

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
        # Sprint 4.2: optional attention pooling instead of mean
        if instr_context is not None and self.use_instrument_conditioning:
            instr_context = instr_context.long()
            if instr_context.dim() == 1:
                instr_context = instr_context.unsqueeze(0)
            instr_raw = self.instrument_embed(instr_context)  # (B, N, D)
            if self.use_instr_attention_pooling and self.instr_pooling is not None:
                instr_emb = self.instr_pooling(instr_raw)  # (B, 1, D)
            else:
                instr_emb = instr_raw.mean(dim=1, keepdim=True)  # (B, 1, D)
            instr_emb = self.instr_proj(instr_emb)
            instr_emb = instr_emb.expand(batch_size, seq_len, -1)

        # CONTROL conditioning
        if control_context is not None and self.use_control_conditioning:
            if control_context.dim() == 2:
                control_context = control_context.unsqueeze(1).expand(-1, seq_len, -1)
            elif control_context.dim() == 3 and control_context.size(1) == 1:
                control_context = control_context.expand(-1, seq_len, -1)
            control_emb = self.control_proj(control_context.float())

        # --- Hierarchical LSTM flow ---

        # LSTM1: Style + base embeddings
        lstm1_in = x_embed
        if style_emb is not None:
            lstm1_in = lstm1_in + style_emb

        h0 = None if hidden_states is None else hidden_states[0]
        if self.use_checkpointing and self.training:
            out1, h1 = checkpoint(self._lstm1_forward, lstm1_in, h0, use_reentrant=False)
        else:
            out1, h1 = self.lstm1(lstm1_in, h0)

        # LSTM2: Instrument conditioning
        # Sprint 4.1: Cross-attention from LSTM2 to LSTM1 outputs
        lstm2_in = out1
        if self.use_cross_attention:
            lstm2_in = self.cross_attn(out1, out1)  # self-attention on LSTM1 output

        if instr_emb is not None:
            lstm2_in = lstm2_in + instr_emb
        h1_state = None if hidden_states is None else hidden_states[1]
        if self.use_checkpointing and self.training:
            out2, h2 = checkpoint(self._lstm2_forward, lstm2_in, h1_state, use_reentrant=False)
        else:
            out2, h2 = self.lstm2(lstm2_in, h1_state)

        # LSTM3: Control conditioning
        lstm3_in = out2
        if control_emb is not None:
            lstm3_in = lstm3_in + control_emb
        h2_state = None if hidden_states is None else hidden_states[2]
        if self.use_checkpointing and self.training:
            out3, h3 = checkpoint(self._lstm3_forward, lstm3_in, h2_state, use_reentrant=False)
        else:
            out3, h3 = self.lstm3(lstm3_in, h2_state)

        out3 = self.dropout(out3)

        # Sprint 4.3: Joint token fusion (couples logits)
        if self.token_fusion is not None:
            out3 = self.token_fusion(out3)

        # --- Output projections ---
        pitch_logits = self.pitch_out(out3)
        dur_logits = self.duration_out(out3)
        vel_logits = self.velocity_out(out3)
        instr_logits = self.instrument_out(out3)

        return pitch_logits, dur_logits, vel_logits, instr_logits, (h1, h2, h3)

    # ---- torch.compile support ----

    def compile_model(self, backend="inductor", mode=None):
        try:
            self.forward = torch.compile(self.forward, backend=backend, mode=mode)
            self._compiled = True
            print(f"[COMPILE] Model compiled with backend='{backend}', mode={mode}")
        except Exception as e:
            print(f"[COMPILE] torch.compile failed ({e}), falling back to eager mode")
        return self

    # ---- Sprint 3: torch.compile support ----

    def compile_model(self, backend="inductor", mode=None):
        """Apply torch.compile to the forward pass for graph optimization."""
        try:
            self.forward = torch.compile(self.forward, backend=backend, mode=mode)
            self._compiled = True
            print(f"[COMPILE] Model compiled with backend='{backend}', mode={mode}")
        except Exception as e:
            print(f"[COMPILE] torch.compile failed ({e}), falling back to eager mode")
        return self


class HLSTMTrainer:
    def __init__(self, model, lr=1e-3, device=None,
                 use_amp=True, loss_weights=None,
                 scheduler_type="none", lr_patience=3, lr_min=1e-6,
                 grad_clip_norm=0.0):
        self.device = device or Config.DEVICE if device is None else device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        # --- Sprint 1: AMP ---
        self.use_amp = use_amp and self.device.type in ("cuda", "mps")
        self.scaler = torch.amp.GradScaler(device=self.device.type) if self.use_amp else None
        self.grad_clip_norm = grad_clip_norm

        # --- Sprint 1: Loss weights ---
        if loss_weights is None:
            self.loss_weights = [1.0, 1.0, 1.0, 1.0]
        else:
            self.loss_weights = loss_weights

        # --- Sprint 1: LR scheduler ---
        self.scheduler = None
        self.scheduler_type = scheduler_type
        if scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5,
                patience=lr_patience, min_lr=lr_min, verbose=True
            )
        elif scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=30, eta_min=lr_min
            )

    def _compute_loss(self, pitch_logits, dur_logits, vel_logits, instr_logits, y_pitch, y_dur, y_vel, y_instr):
        pitch_loss = self.criterion(pitch_logits.view(-1, pitch_logits.size(-1)), y_pitch.reshape(-1))
        dur_loss = self.criterion(dur_logits.view(-1, dur_logits.size(-1)), y_dur.reshape(-1))
        vel_loss = self.criterion(vel_logits.view(-1, vel_logits.size(-1)), y_vel.reshape(-1))
        instr_loss = self.criterion(instr_logits.view(-1, instr_logits.size(-1)), y_instr.reshape(-1))

        w = self.loss_weights
        loss = (w[0] * pitch_loss + w[1] * dur_loss + w[2] * vel_loss + w[3] * instr_loss) / sum(w)
        return loss, {"pitch": pitch_loss.item(), "duration": dur_loss.item(),
                       "velocity": vel_loss.item(), "instrument": instr_loss.item()}

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

        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type):
                pitch_logits, dur_logits, vel_logits, instr_logits, _ = self.model(
                    x, style=style, instr_context=instr_context, control_context=control_context
                )
                loss, _ = self._compute_loss(
                    pitch_logits, dur_logits, vel_logits, instr_logits,
                    y_pitch, y_dur, y_vel, y_instr
                )
            self.scaler.scale(loss).backward()
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            pitch_logits, dur_logits, vel_logits, instr_logits, _ = self.model(
                x, style=style, instr_context=instr_context, control_context=control_context
            )
            loss, _ = self._compute_loss(
                pitch_logits, dur_logits, vel_logits, instr_logits,
                y_pitch, y_dur, y_vel, y_instr
            )
            loss.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
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

            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type):
                    pitch_logits, dur_logits, vel_logits, instr_logits, _ = self.model(
                        x, style=style, instr_context=instr_context, control_context=control_context
                    )
                    loss, _ = self._compute_loss(
                        pitch_logits, dur_logits, vel_logits, instr_logits,
                        y_pitch, y_dur, y_vel, y_instr
                    )
            else:
                pitch_logits, dur_logits, vel_logits, instr_logits, _ = self.model(
                    x, style=style, instr_context=instr_context, control_context=control_context
                )
                loss, _ = self._compute_loss(
                    pitch_logits, dur_logits, vel_logits, instr_logits,
                    y_pitch, y_dur, y_vel, y_instr
                )

        return loss.item()

    def scheduler_step(self, val_loss):
        if self.scheduler_type == "plateau":
            self.scheduler.step(val_loss)
        elif self.scheduler_type == "cosine":
            self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

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
