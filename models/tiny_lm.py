"""Tiny transformer LM with word-bitmap inputs.

Architecture (shared with BPELM via LMBody — only the input encoder differs):
  word bitmaps  --[encoder]-->  word embeddings  --[LMBody]-->  logits
  (B, T, 7, 60)                 (B, T, d_model)                 (B, T, V)

Defaults sized for the RTX 2060 (6 GB VRAM).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import WordBitmapEncoder
from .transformer_body import LMBody


@dataclass
class ModelConfig:
    vocab_size: int                  # output classes for next-word prediction
    k: int = 12                       # cells per word bitmap
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    max_seq_len: int = 256
    dropout: float = 0.1
    encoder_cell_dim: int = 128
    encoder_intra_layers: int = 0     # 0 = simple linear pool over cells
    tie_head: bool = False            # tie head to encoder via lookup over vocab bitmaps

    def estimate_params(self):
        d = self.d_model
        h = self.encoder_cell_dim
        encoder = 35 * h + h + h * self.k * d + d
        if self.encoder_intra_layers > 0:
            encoder += self.encoder_intra_layers * (12 * h * h)
        per_layer = 12 * d * d
        body = self.n_layers * per_layer
        head = 0 if self.tie_head else d * self.vocab_size
        return encoder + body + head


class TinyBitmapLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = WordBitmapEncoder(
            k=cfg.k, d_model=cfg.d_model,
            cell_dim=cfg.encoder_cell_dim,
            n_intra_layers=cfg.encoder_intra_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )
        self.body = LMBody(
            d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_layers=cfg.n_layers, max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
        )
        if cfg.tie_head:
            self.head = None
        else:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def encode_words(self, bitmaps):
        """(B, T, 7, K*5) -> (B, T, d_model)"""
        B, T, H, W = bitmaps.shape
        flat = bitmaps.reshape(B * T, H, W)
        return self.encoder(flat).view(B, T, -1)

    def forward(self, bitmaps, target_ids=None, vocab_bitmaps=None):
        """
        bitmaps:        (B, T, 7, K*5) float
        target_ids:     (B, T) int64 next-word IDs; -100 for ignore
        vocab_bitmaps:  (V, 7, K*5) float — required when tie_head=True

        Returns: logits (B, T, V), loss (scalar or None).
        """
        x = self.encode_words(bitmaps)
        x = self.body(x)

        if self.cfg.tie_head:
            assert vocab_bitmaps is not None, "tie_head requires vocab_bitmaps"
            vocab_emb = self.encoder(vocab_bitmaps)  # (V, d_model)
            logits = x @ vocab_emb.T
        else:
            logits = self.head(x)

        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                target_ids.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss
