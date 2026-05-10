"""Training script for TinyBitmapLM on WikiText-2.

Run on the training box (RTX 2060):
    python3 training/train_bitmap.py

Key flags for VRAM tuning:
    --batch 16 --seq 128     # safe defaults for 6 GB
    --batch 32 --seq 256     # push it if VRAM allows
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn

from models import TinyBitmapLM, ModelConfig
from pixenc.dotmatrix_font import DEFAULT_K
from pixenc.vocab import load_vocab
from training.data import (
    load_text, tokenize_words, words_to_ids,
    render_vocab_bitmaps, sample_window_batch,
    lookup_bitmaps, eval_loss_bitmap,
    WIKITEXT_TRAIN, WIKITEXT_VALID,
)
from utils import get_device, device_summary, count_params


def cosine_lr(step, warmup_steps, total_steps, lr_max, lr_min=1e-5):
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


def parse_args():
    p = argparse.ArgumentParser(description="Train TinyBitmapLM on WikiText-2")

    # VRAM-critical
    p.add_argument("--batch",    type=int, default=16,  help="batch size")
    p.add_argument("--seq",      type=int, default=128, help="sequence length (tokens)")
    p.add_argument("--accum",    type=int, default=2,   help="gradient accumulation steps")

    # Model
    p.add_argument("--d_model",  type=int, default=256)
    p.add_argument("--n_heads",  type=int, default=4)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--cell_dim", type=int, default=128)
    p.add_argument("--intra",    type=int, default=0,   help="intra-word transformer layers")
    p.add_argument("--tie_head", action="store_true",   help="tie output head to encoder proj")
    p.add_argument("--k",        type=int, default=DEFAULT_K)

    # Training
    p.add_argument("--steps",    type=int, default=50_000)
    p.add_argument("--warmup",   type=int, default=2_000)
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--wd",       type=float, default=0.1)
    p.add_argument("--clip",     type=float, default=1.0)

    # Eval / logging
    p.add_argument("--eval_every",  type=int, default=500)
    p.add_argument("--log_every",   type=int, default=50)
    p.add_argument("--eval_batches",type=int, default=50)
    p.add_argument("--save_dir",    type=str, default="artifacts/checkpoints/bitmap")

    # Corpus
    p.add_argument("--train_file", type=str, default=WIKITEXT_TRAIN)
    p.add_argument("--valid_file", type=str, default=WIKITEXT_VALID)
    p.add_argument("--vocab_file", type=str, default=None,
                   help="vocab source — default: /usr/share/dict/american-english")

    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device_summary()}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Vocab + bitmaps ----
    print("Loading vocab...", end=" ", flush=True)
    vocab_kwargs = {"lowercase_only": True}
    if args.vocab_file:
        vocab_kwargs["path"] = args.vocab_file
    vocab = load_vocab(**vocab_kwargs)
    print(f"{len(vocab)} words")

    print("Pre-rendering bitmaps...", end=" ", flush=True)
    bitmaps_table = render_vocab_bitmaps(vocab, k=args.k)
    bitmaps_t = torch.tensor(bitmaps_table.astype(np.float32), device=device)
    print(f"table shape {bitmaps_t.shape}")

    # ---- Corpus ----
    print(f"Loading corpus from {args.train_file}...")
    if not Path(args.train_file).exists():
        print(f"ERROR: {args.train_file} not found — run: python3 training/download_wikitext2.py")
        sys.exit(1)

    train_ids = words_to_ids(tokenize_words(load_text(args.train_file)), vocab)
    valid_ids = words_to_ids(tokenize_words(load_text(args.valid_file)), vocab)
    print(f"  train: {len(train_ids):,} tokens  |  valid: {len(valid_ids):,} tokens")

    # ---- Model ----
    cfg = ModelConfig(
        vocab_size=len(vocab) + 1,  # +1 for UNK (id = len(vocab))
        k=args.k,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq,
        encoder_cell_dim=args.cell_dim,
        encoder_intra_layers=args.intra,
        tie_head=args.tie_head,
    )
    model = TinyBitmapLM(cfg).to(device)
    n_params = count_params(model)
    print(f"Model: {n_params/1e6:.2f}M params")

    # ---- Optimizer + schedule ----
    decay_params = [p for name, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for name, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": args.wd},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    # ---- Training loop ----
    rng = np.random.default_rng(42)
    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(1, args.steps + 1):
        # LR schedule
        lr = cosine_lr(step - 1, args.warmup, args.steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Accumulate gradients
        step_loss = 0.0
        for _ in range(args.accum):
            batch = sample_window_batch(train_ids, args.seq, args.batch, rng)
            input_ids = batch[:, :-1]
            target_ids = torch.tensor(batch[:, 1:], dtype=torch.long, device=device)
            input_bitmaps = bitmaps_t[input_ids]  # (B, T, 7, K*5) — float32
            _, loss = model(input_bitmaps, target_ids=target_ids)
            (loss / args.accum).backward()
            step_loss += loss.item() / args.accum

        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        optimizer.zero_grad()

        running_loss += step_loss

        # Logging
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            avg = running_loss / args.log_every
            ppl = math.exp(min(avg, 20))
            print(f"step {step:6d}/{args.steps}  loss {avg:.4f}  ppl {ppl:.1f}  "
                  f"lr {lr:.2e}  {elapsed:.0f}s")
            running_loss = 0.0
            t0 = time.time()

        # Eval
        if step % args.eval_every == 0:
            val_loss = eval_loss_bitmap(
                model, valid_ids, bitmaps_table, args.seq, args.batch,
                args.eval_batches, device, rng,
            )
            val_ppl = math.exp(min(val_loss, 20))
            print(f"  >>> VALID  loss {val_loss:.4f}  ppl {val_ppl:.1f}")

            ckpt = {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "cfg": cfg.__dict__,
                "args": vars(args),
            }
            path = save_dir / f"step_{step:06d}.pt"
            torch.save(ckpt, path)
            print(f"  >>> saved {path}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = save_dir / "best.pt"
                torch.save(ckpt, best_path)
                print(f"  >>> new best → {best_path}")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}  ppl {math.exp(min(best_val_loss,20)):.1f}")


if __name__ == "__main__":
    main()
