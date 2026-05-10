# Experiment 3: BPE-Bitmap + nn.Embedding (hybrid input) on WikiText-2
**Date:** 2026-05-10  
**Hardware:** NVIDIA GeForce RTX 2060 (6 GB VRAM)  
**Software:** Python 3.11.8, PyTorch 2.5.1+cu124

---

## Motivation

Exp 2 found that the bitmap encoder alone (ppl 204) is 3× worse than a plain
`nn.Embedding` (ppl 66) on the same 8k BPE target space. The failure is that
BPE subword strings carry no useful visual structure for the encoder to
exploit. Exp 3 tests whether **adding** a learned token embedding on top of the
bitmap encoding recovers the gap, and whether the bitmap contributes anything
once an embedding is present.

The bitmap encoder is **kept**, not replaced — this is a hybrid input ablation.
All hyperparameters and the target space are held constant vs Exp 2.

---

## Three fusion variants

| Tag | Fusion | Extra params vs Exp 2 |
|---|---|---|
| **3a** | `sum` — `x = bitmap_enc + tok_emb` | +2.05M (embedding table) |
| **3b** | `concat` — `x = W·[bitmap_enc; tok_emb]`, `W ∈ R^{d×2d}` | +2.18M (+131k proj) |
| **3c** | `gate` — `x = σ(g)·bitmap_enc + (1−σ(g))·tok_emb`, `g ∈ R^d` | +2.05M (+256 gate) |

For `3c`, `g` is a learned per-feature vector initialised to 0 (equal blend at
step 0). The trained `σ(g)` distribution tells us how much the model routes to
each stream. Tied head for all three: output head = `token_emb.weight.T`.

---

## Controlled variables (held constant vs Exp 1 and 2)

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
| Seed | 42 |
| BPE tokenizer | `artifacts/bpe/bpe_8000.json` |

---

## Final results — all experiments

| Exp | Model | Input | Params | Best val loss | Best val ppl | Notes |
|---|---|---|---|---|---|---|
| 1 | BPE LM | `nn.Embedding` only | 5.17M | 4.191 | **66.1** | baseline |
| 1 | Bitmap LM | Word bitmaps | 21.5M | 5.095 | **163.1** | word-level targets |
| 2 | BPE-Bitmap LM | Subword bitmaps | 5.17M | 5.317 | **203.8** | bitmap alone, 8k targets |
| **3a** | **BPE-Bitmap+Emb (sum)** | **bitmap + emb, sum** | **7.22M** | **4.200** | **66.7** | |
| **3b** | **BPE-Bitmap+Emb (concat)** | **bitmap + emb, concat** | **7.35M** | **4.214** | **67.6** | |
| **3c** | **BPE-Bitmap+Emb (gate)** | **bitmap + emb, gate** | **7.22M** | **4.193** | **66.2** | see gate analysis below |

*Exp 1 Bitmap LM ppl is over word tokens; all other ppl values are over the
same 8k BPE tokenisation and are directly comparable.*

---

## Gate analysis (Exp 3c)

At convergence, the per-feature gate `σ(g)` had:

| Statistic | Value |
|---|---|
| Mean | 0.327 |
| Min | 0.216 |
| Max | 0.477 |

`σ(g) > 0.5` would favour the bitmap stream; `σ(g) < 0.5` favours the token
embedding. **Every one of the 256 features settled below 0.5**, with the mean
at 0.327 — allocating only ~33% of the gate weight to the bitmap on average.
The model has learned to lean heavily on the embedding and treat the bitmap as
a minor secondary signal.

---

## Analysis

**Adding `nn.Embedding` alongside the bitmap fully recovers the Exp 1 BPE
baseline.** All three fusions converge to ppl 66–68, within noise of the
5.17M-param `nn.Embedding`-only BPE LM (ppl 66.1). The bitmap encoder
contributes no measurable lift.

**The gate gives the strongest single-number verdict.** None of the 256
features learned to route even half its weight to the bitmap side. The model
has learned to mostly use the embedding and suppress the bitmap signal — not
completely to zero, but the suppression is consistent across all features.
This is the cleanest evidence so far that bitmap structure is irrelevant for
BPE subword prediction.

**The embedding dominates; the bitmap is redundant.** When both streams are
available, the model effectively becomes a standard `nn.Embedding` LM. The
extra 2M parameters (embedding table) carry the full load; the 5.17M-param
bitmap encoder path ends up mostly idle.

**The fusion method barely matters.** Sum (66.7), concat (67.6), and gate
(66.2) are within 1.4 ppl of each other and all within 1.5 ppl of the
embedding-only baseline. No fusion gives a statistically meaningful advantage.

**Conclusion for the overall hypothesis.** Across three experiments, the
dot-matrix bitmap encoder has not improved language modelling on WikiText-2 at
any setting. The most likely explanation remains that BPE subword strings do
not have useful visual structure — bitmap encoding is a mismatch for the
tokenisation. The next natural test is morphologically-informed segmentation
(character-level or small-vocab BPE/Unigram) where subword strings correlate
more with morphemes and visual form might carry a structural prior.
