"""
Shared neural network layers used by the H-LSTM architecture.

Includes:
    - CrossAttention:      Scaled dot-product cross-attention between layers
    - AttentionPooling:    Learned query-based pooling for instrument context
    - TokenFusion:         Residual MLP for joint logit coupling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Scaled dot-product cross-attention between hierarchical layers.

    Allows one LSTM layer to attend to the outputs of another, enabling
    bidirectional information flow (e.g., instrument choices influence
    phrase pacing).

    Args:
        hidden_dim: Dimension of input/output hidden states.
        num_heads: Number of attention heads. Must divide hidden_dim evenly.
        dropout: Dropout rate applied to attention weights.

    Shape:
        - query: (B, T, D)
        - key_value: (B, S, D)
        - output: (B, T, D)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> torch.Tensor:
        """Apply cross-attention.

        Args:
            query: Query tensor (e.g., LSTM2 hidden states).
            key_value: Key/Value tensor (e.g., LSTM1 outputs).

        Returns:
            Attended output of same shape as query.
        """
        B, Tq, D = query.shape
        _, Tkv, _ = key_value.shape

        Q = (
            self.q_proj(query)
            .view(B, Tq, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(key_value)
            .view(B, Tkv, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(key_value)
            .view(B, Tkv, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (
            torch.matmul(attn, V)
            .transpose(1, 2)
            .contiguous()
            .view(B, Tq, D)
        )
        return self.out_proj(out)


class AttentionPooling(nn.Module):
    """Learned-query multi-head attention pooling.

    Replaces mean-pooling of instrument embeddings with a weighted aggregation
    that preserves distributional information about ensemble composition.

    Args:
        hidden_dim: Dimension of input embeddings.
        num_heads: Number of attention heads.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a set of vectors into a single context vector.

        Args:
            x: Input of shape (B, N, D) where N = number of items.

        Returns:
            Pooled output of shape (B, 1, D).
        """
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return out


class TokenFusion(nn.Module):
    """Residual MLP that couples the four output heads (pitch, dur, vel, instr).

    Applied after LSTM3 before the output projections. The residual connection
    ensures gradient stability while allowing cross-head information sharing.

    Args:
        hidden_dim: Input/output dimension.
        fusion_dim: Bottleneck dimension for the MLP.
        dropout: Dropout rate.
    """

    def __init__(
        self, hidden_dim: int = 512, fusion_dim: int = 256, dropout: float = 0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fusion with residual connection.

        Args:
            x: Input tensor of shape (..., hidden_dim).

        Returns:
            Output tensor of same shape as input.
        """
        return x + self.net(x)
