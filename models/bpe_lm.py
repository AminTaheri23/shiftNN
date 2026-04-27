"""BPE baseline: standard token-embedding transformer LM.

Same shared LMBody as TinyBitmapLM, but the input "encoder" is a vanilla
nn.Embedding lookup table over BPE token IDs. This is the apples-to-apples
control we measure against to claim intelligence-density gains for the
bitmap encoding.

Use `tie_head=True` to weight-tie input embedding and output projection
(standard practice for tiny LMs and parameter-efficient).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_body import LMBody


@dataclass
class BPEConfig:
    vocab_size: int                  # BPE vocab size
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    max_seq_len: int = 256
    dropout: float = 0.1
    tie_head: bool = True            # tie token embedding to output projection

    def estimate_params(self):
        d = self.d_model
        embed = self.vocab_size * d
        per_layer = 12 * d * d
        body = self.n_layers * per_layer
        head = 0 if self.tie_head else d * self.vocab_size
        return embed + body + head


class BPELM(nn.Module):
    def __init__(self, cfg: BPEConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.trunc_normal_(self.tok_emb.weight, std=0.02)
        self.body = LMBody(
            d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_layers=cfg.n_layers, max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
        )
        if cfg.tie_head:
            self.head = None
        else:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids, target_ids=None):
        """
        input_ids:  (B, T) int64 BPE token IDs
        target_ids: (B, T) int64 next-token IDs; -100 for ignore
        """
        x = self.tok_emb(input_ids)
        x = self.body(x)
        if self.cfg.tie_head:
            logits = x @ self.tok_emb.weight.T
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
