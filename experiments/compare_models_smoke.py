"""Smoke test for both LMs side-by-side.

Sets up TinyBitmapLM and BPELM, trains a tiny BPE on the dictionary so we
have a real tokenizer (the dictionary isn't a great training corpus for
BPE, but it's fine for a smoke test — replace with WikiText-2 later).
Runs forward + backward on each and reports param breakdown + VRAM.

Run on the GPU box:
    pip install -r requirements.txt
    python3 experiments/compare_models_smoke.py
"""

import os
import sys
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from models import TinyBitmapLM, ModelConfig, BPELM, BPEConfig
from pixenc.dotmatrix_font import render_word_dm, DM_ROWS, DM_COLS, DEFAULT_K
from pixenc.vocab import load_vocab
from pixenc.bpe import train_bpe, stats
from utils import get_device, device_summary, count_params


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def main():
    device = get_device()
    print(f"Device: {device_summary()}\n")

    # ----- Vocab + BPE setup -----
    section("1. Vocab + BPE setup")
    vocab = load_vocab(lowercase_only=True)
    print(f"Bitmap-LM word vocab: {len(vocab)} (lowercase a-z)")

    corpus_path = "/usr/share/dict/american-english"
    bpe, bpe_path = train_bpe(corpus_path, vocab_size=8000)
    print(f"Trained BPE: {bpe.get_vocab_size()} subwords  (saved to {bpe_path})")
    print("(The dict is a word list, not a sentence corpus — replace with a real corpus later.)")

    sample = "the quick brown fox jumps over the lazy dog"
    s = stats(bpe, sample)
    print(f"BPE on sample sentence: {s['tokens']} tokens / {s['words']} words "
          f"= {s['tokens_per_word']:.2f} tok/word")

    # ----- Configs (matched body, different encoders/heads) -----
    section("2. Model configs")
    bitmap_cfg = ModelConfig(
        vocab_size=len(vocab),
        k=DEFAULT_K, d_model=256, n_heads=4, n_layers=6,
        max_seq_len=256, encoder_cell_dim=128,
        encoder_intra_layers=0,
        tie_head=False,
    )
    bpe_cfg = BPEConfig(
        vocab_size=bpe.get_vocab_size(),
        d_model=256, n_heads=4, n_layers=6,
        max_seq_len=256,
        tie_head=True,
    )
    print(f"  Bitmap LM estimated params: {bitmap_cfg.estimate_params()/1e6:.2f}M")
    print(f"  BPE LM    estimated params: {bpe_cfg.estimate_params()/1e6:.2f}M")
    print(f"  Shared body params (12*d^2 * n_layers): "
          f"{12 * 256 * 256 * 6 / 1e6:.2f}M")

    # ----- Build models -----
    section("3. Building models")
    bitmap_lm = TinyBitmapLM(bitmap_cfg).to(device)
    bpe_lm = BPELM(bpe_cfg).to(device)
    print(f"  Bitmap LM actual: {count_params(bitmap_lm)/1e6:.2f}M params")
    print(f"  BPE LM    actual: {count_params(bpe_lm)/1e6:.2f}M params")

    # ----- Bitmap LM forward + backward -----
    section("4. Bitmap LM forward + backward")
    B, T = 4, 16
    sample_words = vocab[:B * T]
    bitmaps_np = np.stack([render_word_dm(w, k=DEFAULT_K) for w in sample_words])
    bitmaps = torch.tensor(
        bitmaps_np, dtype=torch.float32, device=device
    ).view(B, T, DM_ROWS, DM_COLS * DEFAULT_K)
    targets_b = torch.randint(0, bitmap_cfg.vocab_size, (B, T), device=device)

    logits_b, loss_b = bitmap_lm(bitmaps, target_ids=targets_b)
    loss_b.backward()
    print(f"  input shape:  {tuple(bitmaps.shape)}")
    print(f"  logits shape: {tuple(logits_b.shape)}")
    print(f"  initial loss: {loss_b.item():.3f}  "
          f"(expect ~ln V = {math.log(bitmap_cfg.vocab_size):.3f})")

    # ----- BPE LM forward + backward -----
    section("5. BPE LM forward + backward")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    input_ids = torch.randint(0, bpe_cfg.vocab_size, (B, T), device=device)
    targets_t = torch.randint(0, bpe_cfg.vocab_size, (B, T), device=device)
    logits_t, loss_t = bpe_lm(input_ids, target_ids=targets_t)
    loss_t.backward()
    print(f"  input shape:  {tuple(input_ids.shape)}")
    print(f"  logits shape: {tuple(logits_t.shape)}")
    print(f"  initial loss: {loss_t.item():.3f}  "
          f"(expect ~ln V = {math.log(bpe_cfg.vocab_size):.3f})")

    # ----- VRAM summary -----
    if device.type == "cuda":
        section("6. VRAM")
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM (this tiny batch): {peak:.3f} GB")
        print(f"  For real training (B=32, T=256), expect ~0.5-1.5 GB on RTX 2060.")


if __name__ == "__main__":
    main()
