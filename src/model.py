import math
import torch
import torch.nn as nn

#================ Implementing each component in Transformer without using off-the-shelf models ===================================#

class PositionalEncoding(nn.Module):
    '''The vanilla static PE.'''

    def __init__(self, d_model, max_len=512): # following the convention in NLP
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
       
        return x + self.pe[:, :x.size(1)]  # x: (batch, seq_len, d_model), not the default 512


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q, K, V, O projections (no bias — input is already LayerNorm'd)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        B, T, _ = x.shape

        # Project and reshape to (batch, n_heads, seq_len, d_k)
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum and reshape back
        out = attn_weights @ V  # (B, n_heads, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        return self.W_o(out), attn_weights


class FeedForward(nn.Module):
    """2-layer MLP, GELU activation."""

    def __init__(self, d_model, d_ff, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm attention + residual
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed)
        x = x + self.dropout(attn_out)

        # Pre-norm FFN + residual
        normed = self.norm2(x)
        x = x + self.dropout(self.ff(normed))

        return x, attn_weights


class TemporalAttentionPooling(nn.Module):
    """Learnable query vector attends over the sequence to produce a fixed-size output."""

    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        scores = torch.matmul(self.query, x.transpose(-2, -1))  # (1, 1, seq_len), i.e., dot-product the query with each seq
        attn_weights = torch.softmax(scores / math.sqrt(x.size(-1)), dim=-1)
        pooled = torch.matmul(attn_weights, x)  # (batch, 1, d_model)
        return pooled.squeeze(1)  # (batch, d_model)


class CO2Transformer(nn.Module):
    """Encoder-only Transformer for CO2 time-series forecasting.

    Input:  (batch, input_window, n_features)  — 90 sensors + 6 one-hot = 96
    Output: (batch, forecast_window, 6)        — CO2 at 6 sampling points
    """

    def __init__(
        self,
        n_features=96,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        forecast_window=1,
        dropout=0.2,
    ):
        super().__init__()
        
        self.forecast_window = forecast_window

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Temporal pooling + prediction head
        self.pool = TemporalAttentionPooling(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # lighter dropout in head
            nn.Linear(d_ff, forecast_window * 6),
        )

    def forward(self, x):
        # x: (batch, input_window, n_features)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.input_dropout(x)

        # Store attention weights for interpretability
        self.attn_weights = []
        for block in self.blocks:
            x, attn_w = block(x)
            self.attn_weights.append(attn_w)

        x = self.final_norm(x)
        x = self.pool(x)                    # (batch, d_model)
        x = self.head(x)                    # (batch, forecast_window * 6)
        x = torch.sigmoid(x)                # output in [0, 1] (MinMax-scaled targets)
        return x.view(-1, self.forecast_window, 6)

    def get_attention_weights(self):
        """Return attention weights from last forward pass (for analysis)."""

        return self.attn_weights
