# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt                    # torch, numpy, tokenizers
python3 training/download_wikitext2.py             # fetch WikiText-2 to artifacts/
```

All scripts must be run from the project root so that `pixenc`, `models`, `training`, and `utils` are importable. Each experiment file adds the root to `sys.path` itself — don't rely on that when writing new code; use the root-run convention instead.

## Running experiments

```bash
python3 experiments/model_smoke.py                 # bitmap LM: forward + backward + VRAM
python3 experiments/compare_models_smoke.py        # bitmap vs BPE LM side-by-side
python3 experiments/analyze_alphabet.py            # Hamming-distance spread across fonts
python3 experiments/scan_K.py                      # K-sweep (cells per word bitmap)
python3 experiments/compare_14_vs_16.py            # 14-cell vs 16-cell grid comparison
python3 experiments/compare_all_fonts.py           # dot-matrix vs LCD spread comparison
python3 experiments/word_analysis.py               # word-length / coverage analysis
```

There are no automated tests. The smoke tests serve as integration checks — run them after any model or encoder change.

## Architecture

The project pits two transformer LMs against each other to test whether pixel-encoded text is a better input representation than BPE subwords. The hypothesis is that BPE discards structure at the input layer; encoding characters as dot-matrix bitmaps preserves that structure and should improve information density.

### Encoding pipeline (bitmap path)

```
character → 7×5 dot-matrix bitmap (35 bits)
           ↓ render_word_dm(word, k=12)
word       → (7, 60) bool array  [K=12 cells concatenated side-by-side]
           ↓ WordBitmapEncoder
word emb   → (d_model,) float
```

`pixenc/dotmatrix_font.py` holds hand-tuned 5×7 LED glyphs for A-Z, a-z, 0-9, and punctuation. Glyph shapes were adjusted to maximise minimum pairwise Hamming distance (confusable pairs: D/O, M/N, S/5, 8/6/9, P/F/R). K=12 was chosen by the scan in `experiments/scan_K.py` — it covers 95% of the vocab, drops collision rate to 2.1%, and keeps bits/word at 420. Treat K=12 as a locked constant unless you re-run that sweep.

`models/encoder.py` (`WordBitmapEncoder`): reshapes the (7, K×5) bitmap into K letter-cells of 35 pixels each, projects each cell with a shared linear layer, adds learnable positional embeddings, optionally runs an intra-word transformer, then projects the concatenated cells to `d_model`.

### Shared transformer body

`models/transformer_body.py` (`LMBody`): standard causal transformer with learned positional embeddings, pre-norm (`norm_first=True`), and GELU activation. Both LMs use the identical body so comparisons isolate encoder differences.

### Two model variants

| Model | Input encoder | Config class | Key flag |
|---|---|---|---|
| `TinyBitmapLM` | `WordBitmapEncoder` (pixel bitmaps) | `ModelConfig` | `tie_head`: output head can be tied back through the encoder over all vocab bitmaps |
| `BPELM` | `nn.Embedding` (BPE token IDs) | `BPEConfig` | `tie_head=True` by default (standard for small LMs) |

Both are in `models/` and exported from `models/__init__.py`.

### Data path

`training/data.py` provides two parallel pipelines that consume the same WikiText-2 corpus:
- **Bitmap path**: whitespace-tokenize → word IDs → `render_vocab_bitmaps` pre-renders all words to a `(V+1, 7, K×5)` table; `lookup_bitmaps` indexes into it at batch time.
- **BPE path**: byte-level BPE encode → token IDs.

`sample_window_batch` is shared — it samples random contiguous windows of length T+1 from a 1D ID stream, suitable for next-token prediction.

Corpus artifacts live in `artifacts/` (not committed). BPE tokenizer saves to `artifacts/bpe/bpe_{vocab_size}.json`.

### Font comparison

`pixenc/lcd_font.py` provides a 14-segment LCD font as an alternative representation. `pixenc/spread_metrics.py` computes pairwise Hamming distances and summary statistics used in the `experiments/analyze_alphabet.py` and `experiments/compare_all_fonts.py` scripts to justify which font gives better inter-glyph separation.

## Hardware context

Development machine has no GPU. Training target is an RTX 2060 (6 GB VRAM). Default `ModelConfig` hyperparameters (`d_model=256`, `n_layers=6`, `B=32`, `T=256`) are sized to fit within ~2 GB. Always use `utils.get_device()` (auto-selects CUDA/CPU) and check `utils.vram_estimate_mb()` before adding capacity.
