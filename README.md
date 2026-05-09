# shiftNN — Pixel-Encoded Language Model Experiments

## Goal

**Intelligence density**: maximize LLM capability per parameter and per FLOP.

BPE tokenization is the input-layer bottleneck. It collapses character and morphological structure into opaque integer IDs, discarding information the model must later re-learn at a cost in parameters and compute. This project replaces BPE with **pixel-rendered word bitmaps** — one 5×7 dot-matrix bitmap per letter-cell, K=12 cells per word — and feeds the structured image directly into a transformer.

The hypothesis: a physically-grounded, constraint-designed encoding where every pixel carries weight gives the model richer input-layer information and should improve capability at matched parameter count.

---

## Encoding design

### Why LCD / dot-matrix?

Old LCD font designers worked under a tight bit budget. That constraint forced them to choose glyphs that are maximally distinguishable — every segment carries weight. Applied to word encoding:

- **Information density per pixel**: LCD-style glyphs spend pixels on distinguishability, not aesthetics.
- **Predictable Hamming geometry**: any two distinct glyphs differ by ≥ d_min bits (d_min = 3 for uppercase+digits after hand-tuning).
- **Compositional structure preserved**: letters are visible inside the word. Typos and OOV words degrade gracefully — no information cliff at a tokenizer boundary.
- **Inductive bias**: each segment slot in a cell is a candidate linguistic feature (vowel/consonant, ascender/descender, horizontal/vertical stroke). The encoding does some of the linguistic work the model would otherwise learn from scratch.

### Why 5×7 dot-matrix over 14/16-segment?

| Font | Cell bits | d_min / bits | mean / bits | Fill % |
|---|---|---|---|---|
| 14-segment | 63 | 0.000 | 0.264 | 27.8 |
| 16-segment | 63 | 0.032 | 0.249 | 25.7 |
| **5×7 dot-matrix (tuned)** | **35** | **0.086** | **0.398** | **44.2** |

14/16-segment displays leave ~40% of the 7×9 cell unreachable by any segment combination. The 5×7 dot-matrix fills every pixel, achieving higher normalized Hamming spread in 35 bits than 16-seg achieves in 63.

### Why K=12 cells per word?

| K | Fits (%) | Collisions (%) | Bits/word |
|---|---|---|---|
| 6 | 55.9 | 16.4 | 210 |
| 8 | 78.3 | 8.6 | 280 |
| 10 | 89.7 | 4.7 | 350 |
| **12** | **95.0** | **2.1** | **420** |
| 14 | 97.4 | 1.3 | 490 |
| 16 | 98.6 | 0.8 | 560 |

K=12 is the Pareto-optimal choice: 95% of the English dictionary fits without truncation, 2.1% collision rate, 420 bits/word. Going to K=14+ adds bits but delivers diminishing coverage gains.

### Typo locality

Words that differ by a single character substitution have Hamming distances ~9× smaller than random word pairs (substitution ratio = 0.116, spread overlap = 0.04%). The encoding is naturally robust to noise — a desirable byproduct of preserving compositional structure.

---

## Architecture

```
word string
  → render_word_dm(word, k=12)           pixenc/dotmatrix_font.py
  → (7, 60) bool bitmap                 12 cells × 5 cols × 7 rows
  → WordBitmapEncoder                   models/encoder.py
      reshape → (B, K, 35) cells
      cell_embed: 35 → cell_dim
      learnable cell pos embedding
      optional intra-word transformer
      word_proj: cell_dim*K → d_model
  → (B, T, d_model)
  → LMBody (causal transformer)         models/transformer_body.py
  → linear head → vocab logits
```

Both models (bitmap LM and BPE baseline) share the same `LMBody` — a standard causal transformer with `norm_first=True`. The comparison isolates the input encoding.

### Model files

| File | Contents |
|---|---|
| `pixenc/dotmatrix_font.py` | 74-glyph hand-tuned 5×7 font (A-Z, a-z, 0-9, punctuation). `DEFAULT_K=12`. |
| `pixenc/spread_metrics.py` | Pairwise Hamming analysis utilities. |
| `pixenc/vocab.py` | Load `/usr/share/dict/american-english` → 63,875 lowercase words. |
| `pixenc/perturb.py` | Random typo generation (sub/ins/del/trans). |
| `pixenc/bpe.py` | BPE training + loading via `tokenizers` library. |
| `models/encoder.py` | `WordBitmapEncoder` — patches a word bitmap into per-cell embeddings. |
| `models/transformer_body.py` | `LMBody` — shared causal transformer body. |
| `models/tiny_lm.py` | `TinyBitmapLM` + `ModelConfig`. |
| `models/bpe_lm.py` | `BPELM` + `BPEConfig` (BPE baseline). |
| `training/data.py` | Corpus loading, window sampling, bitmap lookup, eval. |
| `utils.py` | `get_device()`, `count_params()`, `vram_estimate_mb()`. |

---

## Running the experiments

### Prerequisites

```bash
pip install -r requirements.txt
```

### Analysis (no GPU needed)

```bash
# Alphabet Hamming spread — all three font variants
python3 experiments/compare_all_fonts.py

# K-sweep: shows K=12 as optimal
python3 experiments/scan_K.py

# Word-level typo locality at K=12
python3 experiments/word_analysis.py
```

### Model smoke tests (run on RTX 2060)

```bash
# TinyBitmapLM: shape check, loss ≈ ln(V), grad, VRAM
python3 experiments/model_smoke.py

# Both LMs side-by-side
python3 experiments/compare_models_smoke.py
```

### Download corpus

```bash
python3 training/download_wikitext2.py
# → artifacts/wikitext-2/wiki.{train,valid,test}.tokens
```

### Training

```bash
# Bitmap LM (safe defaults for 6 GB VRAM)
python3 training/train_bitmap.py --batch 16 --seq 128

# BPE baseline (matched architecture)
python3 training/train_bpe.py --batch 16 --seq 128 --tie_head

# Push batch/seq up if VRAM allows
python3 training/train_bitmap.py --batch 32 --seq 256
```

Checkpoints save to `artifacts/checkpoints/bitmap/` and `artifacts/checkpoints/bpe/`. Best checkpoint is always written to `best.pt`.

---

## Hardware

- **Dev machine**: no GPU. All analysis scripts (`experiments/`) run CPU-only.
- **Training machine**: RTX 2060, 6 GB VRAM. Default config (~21M bitmap, ~6.7M BPE tied-head) needs ~340 MB params+Adam state, leaving ~5.5 GB for activations. Surface VRAM-critical flags (`--batch`, `--seq`, `--d_model`, `--n_layers`) are exposed in both training scripts.

---

## Verification gates

Before investing in long training runs, the encoding must pass these gates:

| Gate | Criterion | Status |
|---|---|---|
| **G1 — encoding space** | Hamming distribution healthy; typo-pair distance materially smaller than random | **PASSED** (ratio 0.116, 0.04% overlap) |
| **G2 — modeling** | Bitmap LM reaches within ~10% PPL of BPE baseline at matched param count | pending training |
| **G3 — intelligence density** | At matched compute, bitmap Pareto-improves BPE on PPL | pending training |
| **G4 — robustness** | Outperforms BPE on typo/OOV test set | pending training |
| **G5 — interpretability** | Per-cell linear probes hit clearly above random on ≥1 linguistic feature | pending |

---

## Next steps

1. **Resolve parameter matching** before the first fair comparison run. Current gap: TinyBitmapLM ~21M params (untied head), BPELM ~6.7M (tied head). Options: set `--tie_head` on bitmap LM, or grow BPE vocab to ~32k for ~21M parity.
2. **Run smoke tests** on the RTX 2060 to verify shapes, initial loss, and VRAM before committing to 50k-step runs.
3. **Download WikiText-2** on the training box: `python3 training/download_wikitext2.py`.
4. **Run training** with matched configs; plot validation PPL curves side-by-side.
5. **Eval suite**: held-out PPL, typo/OOV test set, intelligence-density Pareto plot (PPL vs. params × tokens-seen).
6. **Interpretability probes**: linear classifiers on per-cell features for vowel detection, prefix/suffix detection, POS hints.

---

## Future directions (parked)

- **V4 — learned LCD font**: replace the hand-designed font with a learned binary-bottleneck encoder (Gumbel-sigmoid or VQ). Train jointly with the LM plus a spread loss penalizing low Hamming distance between distinct words. Same wrapper, learned glyphs, end-to-end optimized.
- **Shift-register sentence consumer**: consume the word-bitmap stream into the LM column-by-column via a shift-register-like mechanism (1D CNN with shifted receptive fields, or SSM). Composes with V2 but is independent of the encoding choice.

---

## Prior art

- **PIXEL** (Rust et al., NeurIPS 2022) — renders text as page-level images, ViT-MAE backbone. V2 differs by word-level granularity and constraint-designed segment font.
- **Glyce** (Meng et al., NeurIPS 2019) — glyph CNN features for Chinese characters.
- **ByT5 / Canine / CharFormer** — token-free byte/character-level models. Useful baselines; share the "skip BPE" motivation but have no visual structure.
- **Coding theory** — Plotkin/Singleton bounds for formally proving d_min guarantees within a bit budget.
