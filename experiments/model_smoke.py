"""Smoke test for the tiny LM. Verifies forward + backward, prints VRAM use.

Run on the RTX 2060 box:
    pip install -r requirements.txt
    python3 experiments/model_smoke.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import math

import numpy as np
import torch

from models import TinyBitmapLM, ModelConfig
from pixenc.dotmatrix_font import render_word_dm, DM_ROWS, DM_COLS, DEFAULT_K
from pixenc.vocab import load_vocab
from utils import get_device, device_summary, count_params, vram_estimate_mb


def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"{device_summary()}\n")

    vocab = load_vocab(lowercase_only=True)
    print(f"Vocab size: {len(vocab)}")

    cfg = ModelConfig(
        vocab_size=len(vocab),
        k=DEFAULT_K,
        d_model=256,
        n_heads=4,
        n_layers=6,
        max_seq_len=256,
        encoder_cell_dim=128,
    )
    est = cfg.estimate_params()
    print(f"Estimated params: {est/1e6:.2f}M")
    print(f"Estimated baseline VRAM (params + Adam state, no activations): "
          f"{vram_estimate_mb(est)/1024:.2f} GB\n")

    model = TinyBitmapLM(cfg).to(device)
    print(f"Actual params:    {count_params(model)/1e6:.2f}M\n")

    # Build a small input batch from real words.
    B, T = 4, 16
    sample_words = vocab[:B * T]
    bitmaps_np = np.stack([render_word_dm(w, k=DEFAULT_K) for w in sample_words])
    bitmaps = torch.tensor(
        bitmaps_np, dtype=torch.float32, device=device
    ).view(B, T, DM_ROWS, DM_COLS * DEFAULT_K)
    targets = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    print(f"Input shape: {tuple(bitmaps.shape)}  (B={B}, T={T}, H=7, W={DM_COLS*DEFAULT_K})")

    # Forward
    logits, loss = model(bitmaps, target_ids=targets)
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Initial loss: {loss.item():.4f}")
    expected = math.log(cfg.vocab_size)
    print(f"Expected at init (~ln(V)): {expected:.4f}  "
          f"(diff {abs(loss.item() - expected):.3f})")

    # Backward
    loss.backward()
    grad_ok = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0)
    total = sum(1 for _ in model.parameters())
    print(f"Backward: {grad_ok}/{total} parameter tensors received non-zero gradients")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak VRAM (this tiny batch): {peak:.3f} GB")
        print("For real training (B=32, T=256), expect ~0.5-1.5 GB total.")
    else:
        print("\n(CPU run — VRAM stats unavailable.)")


if __name__ == "__main__":
    main()
