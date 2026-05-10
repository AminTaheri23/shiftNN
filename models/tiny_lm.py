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
    fusion: str = "none"              # "none" | "sum" | "concat" | "gate"

    def estimate_params(self):
        d = self.d_model
        h = self.encoder_cell_dim
        encoder = 35 * h + h + h * self.k * d + d
        if self.encoder_intra_layers > 0:
            encoder += self.encoder_intra_layers * (12 * h * h)
        per_layer = 12 * d * d
        body = self.n_layers * per_layer
        emb = d * self.vocab_size if self.fusion != "none" else 0
        fusion_extra = {"none": 0, "sum": 0, "concat": 2 * d * d, "gate": d}[self.fusion]
        head = 0 if self.tie_head else d * self.vocab_size
        return encoder + body + emb + fusion_extra + head


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

        if cfg.fusion != "none":
            self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            # GPT-2-style init — keeps tied-head logits at sane scale (≈ln V at step 0).
            nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        else:
            self.token_emb = None

        if cfg.fusion == "concat":
            self.fusion_proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        elif cfg.fusion == "gate":
            self.fusion_gate = nn.Parameter(torch.zeros(cfg.d_model))

    def encode_words(self, bitmaps):
        """(B, T, 7, K*5) -> (B, T, d_model)"""
        B, T, H, W = bitmaps.shape
        flat = bitmaps.reshape(B * T, H, W)
        return self.encoder(flat).view(B, T, -1)

    def forward(self, bitmaps, target_ids=None, vocab_bitmaps=None, token_ids=None):
        """
        bitmaps:        (B, T, 7, K*5) float
        target_ids:     (B, T) int64 next-word IDs; -100 for ignore
        vocab_bitmaps:  (V, 7, K*5) float — required when tie_head=True and fusion="none"
        token_ids:      (B, T) int64 — required when fusion != "none"

        Returns: logits (B, T, V), loss (scalar or None).
        """
        x = self.encode_words(bitmaps)

        if self.cfg.fusion != "none":
            assert token_ids is not None, f"fusion={self.cfg.fusion} requires token_ids"
            tok = self.token_emb(token_ids)             # (B, T, d)
            if self.cfg.fusion == "sum":
                x = x + tok
            elif self.cfg.fusion == "concat":
                x = self.fusion_proj(torch.cat([x, tok], dim=-1))
            elif self.cfg.fusion == "gate":
                g = torch.sigmoid(self.fusion_gate)     # (d,)
                x = g * x + (1.0 - g) * tok
            else:
                raise ValueError(f"unknown fusion: {self.cfg.fusion}")

        x = self.body(x)

        if self.cfg.tie_head:
            if self.cfg.fusion != "none":
                logits = x @ self.token_emb.weight.T
            else:
                assert vocab_bitmaps is not None, "tie_head + fusion=none requires vocab_bitmaps"
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
