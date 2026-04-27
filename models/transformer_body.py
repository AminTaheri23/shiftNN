"""Shared transformer body. Both TinyBitmapLM and BPELM use this so the
language-modeling architecture is identical across the encoder swap.

Input:  (B, T, d_model) sequence of token-equivalent embeddings
Output: (B, T, d_model) post-body hidden states, ready for an output head.
"""

import torch
import torch.nn as nn


class LMBody(nn.Module):
    def __init__(self, d_model=256, n_heads=4, n_layers=6, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, T, d_model). Returns (B, T, d_model) with causal self-attn applied."""
        B, T, _ = x.shape
        x = x + self.pos_emb[:, :T]
        x = self.dropout(x)
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        x = self.encoder(x, mask=causal_mask)
        return self.norm(x)
