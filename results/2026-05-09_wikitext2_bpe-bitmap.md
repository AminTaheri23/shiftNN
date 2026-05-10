# Experiment 2: BPE-Bitmap LM on WikiText-2
**Date:** 2026-05-09  
**Hardware:** NVIDIA GeForce RTX 2060 (6 GB VRAM)  
**Software:** Python 3.11.8, PyTorch 2.5.1+cu124

---

## Motivation

Experiment 1 found BPE LM beating Bitmap LM 2.5× on val perplexity (66 vs 163).
Two confounds were entangled: the **input representation** (learned embedding vs
pixel bitmap) and the **target space** (8k subwords vs 64k words). Exp 2
disentangles them by keeping the BPE token stream — and thus the same 8k target
space as the BPE baseline — while swapping the input encoder for the bitmap path
(`WordBitmapEncoder`). Any remaining gap is attributable purely to input
representation.

---

## Controlled variables (held constant vs Exp 1)

| Variable | Value |
|---|---|
| Dataset | WikiText-2 (train: ~2.74M BPE tokens) |
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
| BPE tokenizer | `artifacts/bpe/bpe_8000.json` (same as Exp 1 BPE run) |

---

## Model description (Exp 2 only)

| | BPE-Bitmap LM (Exp 2) |
|---|---|
| Input encoding | Dot-matrix bitmap per BPE subword (K=12, `(B,T,7,60)`) |
| Targets | 8,000 BPE subwords (same as Exp 1 BPE baseline) |
| Encoder | `WordBitmapEncoder` (shared cell proj + positional emb) |
| Head | Tied — encoder forward over all 8k vocab bitmaps per step |
| Params | ~5.17M |
| Bitmap table | `(8000, 7, 60)` float32; 3.2% of tokens used fallback char '?' |
| Checkpoint | `artifacts/checkpoints/bpe_bitmap/best.pt` |

Normalization applied to BPE subword strings before rendering:
- `Ġ` (byte-level space marker) → literal space (blank cell, preserves word-boundary signal)
- `Ċ` (newline marker) → space
- chars absent from the dot-matrix font → `?`
- special tokens (`<unk>`, `<bos>`, `<eos>`, `<pad>`) → checkerboard UNK glyph

---

## Training loss (sampled at key steps)

| Step | BPE-Bitmap loss | BPE-Bitmap ppl |
|---|---|---|
| 500 | 7.443 | 1707 |
| 1,000 | 7.159 | 1286 |
| 5,000 | 6.044 | 421 |
| 10,000 | 5.607 | 272 |
| 25,000 | 5.181 | 178 |
| 50,000 | 4.965 | 143 |

---

## Validation loss (all eval checkpoints, every 500 steps)

| Step | Val loss | Val ppl |
|---|---|---|
| 500 | 7.3695 | 1587 |
| 1,000 | 7.1319 | 1251 |
| 1,500 | 6.9286 | 1021 |
| 2,000 | 6.7531 | 857 |
| 2,500 | 6.5646 | 710 |
| 3,000 | 6.3879 | 595 |
| 3,500 | 6.2686 | 528 |
| 4,000 | 6.1950 | 490 |
| 4,500 | 6.1393 | 464 |
| 5,000 | 6.0163 | 410 |
| 5,500 | 5.9905 | 400 |
| 6,000 | 5.9342 | 378 |
| 6,500 | 5.8860 | 360 |
| 7,000 | 5.8549 | 349 |
| 7,500 | 5.8710 | 355 |
| 8,000 | 5.8170 | 336 |
| 8,500 | 5.7828 | 325 |
| 9,000 | 5.7538 | 315 |
| 9,500 | 5.7258 | 307 |
| 10,000 | 5.6988 | 299 |
| 10,500 | 5.6790 | 293 |
| 11,000 | 5.6597 | 287 |
| 11,500 | 5.6665 | 289 |
| 12,000 | 5.6332 | 280 |
| 12,500 | 5.6542 | 286 |
| 13,000 | 5.6142 | 274 |
| 13,500 | 5.6034 | 271 |
| 14,000 | 5.5921 | 268 |
| 14,500 | 5.5997 | 270 |
| 15,000 | 5.5648 | 261 |
| 15,500 | 5.5496 | 257 |
| 16,000 | 5.5334 | 253 |
| 16,500 | 5.5248 | 251 |
| 17,000 | 5.5250 | 251 |
| 17,500 | 5.5346 | 253 |
| 18,000 | 5.5277 | 252 |
| 18,500 | 5.5125 | 248 |
| 19,000 | 5.5424 | 255 |
| 19,500 | 5.4884 | 242 |
| 20,000 | 5.4571 | 234 |
| 20,500 | 5.4577 | 235 |
| 21,000 | 5.4843 | 241 |
| 21,500 | 5.4687 | 237 |
| 22,000 | 5.4570 | 234 |
| 22,500 | 5.4484 | 232 |
| 23,000 | 5.4687 | 237 |
| 23,500 | 5.4347 | 229 |
| 24,000 | 5.4225 | 226 |
| 24,500 | 5.4175 | 225 |
| 25,000 | 5.4208 | 226 |
| 25,500 | 5.4061 | 223 |
| 26,000 | 5.4011 | 222 |
| 26,500 | 5.4285 | 228 |
| 27,000 | 5.4303 | 228 |
| 27,500 | 5.4211 | 226 |
| 28,000 | 5.4201 | 226 |
| 28,500 | 5.4157 | 225 |
| 29,000 | 5.4115 | 224 |
| 29,500 | 5.4270 | 228 |
| 30,000 | 5.4298 | 228 |
| 30,500 | 5.3587 | 212 |
| 31,000 | 5.3462 | 210 |
| 31,500 | 5.3970 | 221 |
| 32,000 | 5.3820 | 218 |
| 32,500 | 5.3993 | 221 |
| 33,000 | 5.3874 | 219 |
| 33,500 | 5.3870 | 219 |
| 34,000 | 5.4021 | 222 |
| 34,500 | 5.4013 | 222 |
| 35,000 | 5.3668 | 214 |
| 35,500 | 5.3659 | 214 |
| 36,000 | 5.3798 | 217 |
| 36,500 | 5.3800 | 217 |
| 37,000 | 5.3843 | 218 |
| 37,500 | 5.3711 | 215 |
| 38,000 | 5.3652 | 214 |
| 38,500 | 5.3891 | 219 |
| 39,000 | 5.3457 | 210 |
| 39,500 | 5.3517 | 211 |
| 40,000 | 5.3356 | 208 |
| 40,500 | 5.3728 | 216 |
| 41,000 | 5.3700 | 215 |
| 41,500 | 5.3530 | 211 |
| 42,000 | 5.3766 | 216 |
| **42,500** | **5.3173** | **204** |
| 43,000 | 5.3450 | 210 |
| 43,500 | 5.3300 | 206 |
| 44,000 | 5.3522 | 211 |
| 44,500 | 5.3352 | 208 |
| 45,000 | 5.3333 | 207 |
| 45,500 | 5.3636 | 214 |
| 46,000 | 5.3432 | 209 |
| 46,500 | 5.3479 | 210 |
| 47,000 | 5.3314 | 207 |
| 47,500 | 5.3422 | 209 |
| 48,000 | 5.3233 | 205 |
| 48,500 | 5.3301 | 207 |
| 49,000 | 5.3527 | 211 |
| 49,500 | 5.3335 | 207 |
| 50,000 | 5.3205 | 205 |

Best checkpoint: step 42,500 (val loss 5.3173, ppl **203.8**).

---

## Three-way final results

| Model | Input | Targets | Best val loss | Best val ppl | Train ppl @ 50k | Step time |
|---|---|---|---|---|---|---|
| BPE LM (Exp 1) | `nn.Embedding` | 8k BPE | 4.191 | **66.1** | 28.7 | ~3s |
| Bitmap LM (Exp 1) | Word bitmaps | 64k words | 5.095 | **163.1** | 43.5 | ~7s |
| **BPE-Bitmap LM (Exp 2)** | **Subword bitmaps** | **8k BPE** | **5.317** | **203.8** | **143.4** | **~3s** |

*Perplexity numbers for BPE LM and BPE-Bitmap LM are measured over the same BPE
tokenization of WikiText-2 validation and are directly comparable. Bitmap LM ppl
is over word tokens and is not on the same scale.*

---

## Analysis

**Controlling for target space makes the bitmap input look worse, not better.**
The BPE LM and BPE-Bitmap LM both predict over the same 8k BPE vocabulary,
isolating the effect of input representation. The gap is large: ppl 66 vs 204, a
3× difference in favour of the standard learned embedding. The bitmap encoding
is actively harmful here relative to `nn.Embedding`.

**Why the bitmap encoder struggles on BPE subwords.** A learned embedding is
just a dense lookup: it memorises one unconstrained vector per token and fits
it freely to the language modelling objective. The bitmap encoder, by contrast,
must compress the visual structure of the subword string into a fixed projection.
BPE subwords are arbitrary character sequences chosen by a frequency heuristic —
`Ġthe`, `ing`, `ation` — and their dot-matrix renderings carry no particularly
useful structural prior for predicting the next subword. The encoder is fighting
the tokenisation.

**Training loss gap confirms the issue is the encoder, not overfitting.**
At step 50k the BPE LM train ppl is 29 while BPE-Bitmap train ppl is 143 — the
bitmap model has not even fitted the training set as well. This shows the bitmap
encoder bottlenecks learning capacity, not just generalisation.

**Convergence stalls after step 30k.** Val ppl drops from 1587 → ~210 over the
first 30k steps, then plateaus with only marginal improvement (210 → 204). The
BPE LM continued improving through step 50k. The bitmap encoder's fixed
projection likely saturates capacity early.

**Implications for the hypothesis.** The original hypothesis — that bitmap
encoding preserves structure that BPE discards — may hold for word-level
tokenisation (where character visual structure is linguistically meaningful) but
fails for BPE subwords (where the subword boundaries are information-theoretic,
not morphological). The right next step (Exp 3) is to test the bitmap encoder on
a morphologically-informed segmentation (e.g. BPE trained at a smaller vocab
size, or Unigram LM) where subword strings are more likely to correspond to
meaningful character sequences.
