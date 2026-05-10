# Experiment: BPE vs Bitmap LM on WikiText-2
**Date:** 2026-05-09  
**Hardware:** NVIDIA GeForce RTX 2060 (6 GB VRAM)  
**Software:** Python 3.11.8, PyTorch 2.5.1+cu124

---

## Controlled variables

| Variable | Value |
|---|---|
| Dataset | WikiText-2 (train: ~1.69M word tokens / ~10.9M chars) |
| Architecture | Shared `LMBody`: causal transformer, pre-norm, GELU |
| `d_model` | 256 |
| `n_heads` | 4 |
| `n_layers` | 6 |
| `max_seq_len` | 128 |
| Steps | 50,000 |
| Batch size | 16 |
| Grad accum | 2 (effective batch = 32) |
| LR schedule | Cosine decay, max 3e-4, min 1e-5 |
| Warmup | 2,000 steps |
| Weight decay | 0.1 |
| Grad clip | 1.0 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Eval every | 500 steps |
| Seed | 42 |

---

## Model differences

| | BPE LM | Bitmap LM |
|---|---|---|
| Input encoding | Byte-level BPE token embeddings | Dot-matrix word bitmaps (5×7 per char, K=12 cells) |
| Vocab / token space | 8,000 BPE subwords | 63,875 words + 1 UNK |
| Input tensor shape | `(B, T)` int IDs | `(B, T, 7, 60)` float bitmaps |
| Encoder | `nn.Embedding` | `WordBitmapEncoder` (linear proj + positional emb) |
| Head | Tied to embedding (`tie_head=True`) | Separate linear head (`tie_head=False`) |
| Params | ~5.3M | ~21.5M |
| Checkpoint | `artifacts/checkpoints/bpe/best.pt` | `artifacts/checkpoints/bitmap/best.pt` |

---

## Training loss (train, sampled at key steps)

| Step | BPE loss | BPE ppl | Bitmap loss | Bitmap ppl |
|---|---|---|---|---|
| 500 | 6.975 | 1070 | 6.543 | 694 |
| 1,000 | 6.282 | 535 | 6.183 | 484 |
| 5,000 | 4.544 | 94 | 4.964 | 143 |
| 10,000 | 4.018 | 56 | 4.472 | 88 |
| 25,000 | 3.571 | 36 | 3.962 | 53 |
| 50,000 | 3.357 | 29 | 3.772 | 44 |

---

## Validation loss (all eval checkpoints, every 500 steps)

### BPE LM
| Step | Val loss | Val ppl |
|---|---|---|
| 500 | 6.922 | 1014 |
| 1,000 | 6.229 | 507 |
| 1,500 | 5.857 | 350 |
| 2,000 | 5.577 | 264 |
| 2,500 | 5.370 | 215 |
| 3,000 | 5.159 | 174 |
| 3,500 | 5.015 | 151 |
| 4,000 | 4.924 | 138 |
| 4,500 | 4.838 | 126 |
| 5,000 | 4.719 | 112 |
| 10,000 | ~4.40 | ~81 |
| 25,000 | ~4.28 | ~72 |
| **best** | **4.191** | **66.1** |
| 50,000 | 4.200 | 66.7 |

### Bitmap LM
| Step | Val loss | Val ppl |
|---|---|---|
| 500 | 6.465 | 642 |
| 1,000 | 6.159 | 473 |
| 1,500 | 5.988 | 399 |
| 2,000 | 5.802 | 331 |
| 2,500 | 5.663 | 288 |
| 3,000 | 5.579 | 265 |
| 3,500 | 5.480 | 240 |
| 4,000 | 5.403 | 222 |
| 4,500 | 5.324 | 205 |
| 5,000 | 5.292 | 199 |
| 10,000 | ~5.17 | ~176 |
| 25,000 | ~5.10 | ~164 |
| **best** | **5.095** | **163.1** |
| 50,000 | 5.157 | 174 |

---

## Final results

| Model | Best val loss | Best val ppl | Train loss @ 50k | Train ppl @ 50k | Step/s |
|---|---|---|---|---|---|
| BPE LM | 4.191 | **66.1** | 3.357 | 28.7 | ~3s |
| Bitmap LM | 5.095 | **163.1** | 3.772 | 43.5 | ~7s |

---

## Notes

- **BPE wins by 2.5× on val ppl** (66 vs 163). The smaller BPE token vocab (8k vs 64k targets) makes next-token prediction substantially easier.
- **Bitmap LM is ~2× slower per step** due to the bitmap encoder forward pass over the full vocab at each batch.
- **Both models overfit**: train ppl is well below val ppl in both cases, but more severely for BPE (28 train vs 66 val) than Bitmap (44 train vs 163 val).
- **Bug fixed in `training/train_bitmap.py`**: `vocab_size` was set to `len(vocab)` instead of `len(vocab) + 1`, causing an out-of-bounds CUDA assertion on UNK tokens (ID = `len(vocab)`). Fix: `vocab_size=len(vocab) + 1`.
