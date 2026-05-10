"""Smoke test for the BPE-Bitmap LM (Experiment 2).

Trains a tiny BPE on the WikiText-2 train file, renders the vocab as
bitmaps, runs forward + backward on one batch, prints VRAM. Run before
kicking off the 50k-step training.

    python3 experiments/compare_bpe_bitmap_smoke.py
"""

import os
import sys
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from models import TinyBitmapLM, ModelConfig
from pixenc.bpe import train_bpe
from pixenc.dotmatrix_font import DEFAULT_K
from training.data import (
    load_text, bpe_encode, sample_window_batch,
    render_bpe_vocab_bitmaps, WIKITEXT_TRAIN,
)
from utils import get_device, device_summary, count_params


def main():
    device = get_device()
    print(f"Device: {device_summary()}\n")

    print("== 1. Train BPE (vocab=8000) on WikiText-2 train file")
    tokenizer, bpe_path = train_bpe(WIKITEXT_TRAIN, vocab_size=8000)
    V = tokenizer.get_vocab_size()
    print(f"   BPE vocab: {V}  (saved {bpe_path})")

    print("\n== 2. Render BPE vocab as bitmaps")
    bitmaps_table = render_bpe_vocab_bitmaps(tokenizer, k=DEFAULT_K)
    bitmaps_t = torch.tensor(bitmaps_table.astype(np.float32), device=device)
    print(f"   table shape {tuple(bitmaps_t.shape)}")

    print("\n== 3. Build model (TinyBitmapLM, tie_head=True)")
    cfg = ModelConfig(
        vocab_size=V, k=DEFAULT_K, d_model=256, n_heads=4, n_layers=6,
        max_seq_len=128, tie_head=True,
    )
    model = TinyBitmapLM(cfg).to(device)
    print(f"   params: {count_params(model)/1e6:.2f}M")

    print("\n== 4. Forward + backward on one batch")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_ids = bpe_encode(load_text(WIKITEXT_TRAIN), tokenizer)
    print(f"   corpus tokens: {len(train_ids):,}")
    rng = np.random.default_rng(42)
    B, T = 16, 128
    batch = sample_window_batch(train_ids, T, B, rng)
    input_ids = batch[:, :-1]
    target_ids = torch.tensor(batch[:, 1:], dtype=torch.long, device=device)
    input_bitmaps = bitmaps_t[input_ids]
    print(f"   input_bitmaps shape: {tuple(input_bitmaps.shape)}")

    logits, loss = model(input_bitmaps, target_ids=target_ids,
                         vocab_bitmaps=bitmaps_t)
    loss.backward()
    print(f"   logits shape: {tuple(logits.shape)}")
    print(f"   initial loss: {loss.item():.3f}  "
          f"(expect ~ln V = {math.log(V):.3f})")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n== 5. VRAM peak: {peak:.3f} GB")


if __name__ == "__main__":
    main()
