"""Corpus loading and batching utilities for both bitmap and BPE training.

The bitmap and BPE LMs train on the same corpus but with different
tokenizations:
  - bitmap:  whitespace-tokenize, lowercase, map word -> vocab id, render bitmap.
  - BPE:     byte-level BPE encode -> token ids.

`stream_to_tensor` returns a 1D int64 array of token IDs that both
training scripts can sample contiguous windows from.
"""

import re
from pathlib import Path

import numpy as np

from pixenc.dotmatrix_font import render_word_dm, DM_ROWS, DM_COLS, DEFAULT_K


WIKITEXT_TRAIN = "artifacts/wikitext-2/wiki.train.tokens"
WIKITEXT_VALID = "artifacts/wikitext-2/wiki.valid.tokens"
WIKITEXT_TEST  = "artifacts/wikitext-2/wiki.test.tokens"

_WORD_RE = re.compile(r"[a-z]+")


def load_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


# ---- Word-level (bitmap LM path) ----

def tokenize_words(text):
    """Lowercase + extract a-z runs."""
    return _WORD_RE.findall(text.lower())


def words_to_ids(words, vocab):
    """Map words to vocab IDs; OOV -> len(vocab) (the UNK id)."""
    word2id = {w: i for i, w in enumerate(vocab)}
    unk = len(vocab)
    return np.array([word2id.get(w, unk) for w in words], dtype=np.int64)


def render_vocab_bitmaps(vocab, k=DEFAULT_K, unk_pattern="checker"):
    """Pre-render all vocab words plus an UNK glyph.

    Returns: (V+1, 7, K*5) bool array where row V is the UNK bitmap.
    """
    V = len(vocab)
    table = np.zeros((V + 1, DM_ROWS, DM_COLS * k), dtype=bool)
    for i, w in enumerate(vocab):
        table[i] = render_word_dm(w, k=k)
    if unk_pattern == "checker":
        rr, cc = np.meshgrid(np.arange(DM_ROWS), np.arange(DM_COLS * k), indexing="ij")
        table[V] = ((rr + cc) % 2 == 0)
    elif unk_pattern == "blank":
        pass
    elif unk_pattern == "filled":
        table[V] = True
    return table


# ---- BPE-level (BPE LM path) ----

def bpe_encode(text, tokenizer):
    """Byte-level BPE encode -> 1D int64 array of token ids."""
    enc = tokenizer.encode(text)
    return np.array(enc.ids, dtype=np.int64)


# ---- Window sampler (shared) ----

def sample_window_batch(ids, T, B, rng):
    """Sample B random windows of length T+1 from a 1D id stream.

    Returns (B, T+1) int64 array. The first T positions are inputs;
    positions 1..T are next-token targets.
    """
    n = len(ids)
    if n < T + 1:
        raise ValueError(f"corpus has only {n} ids but window needs {T+1}")
    starts = rng.integers(0, n - T, size=B)
    out = np.stack([ids[s:s + T + 1] for s in starts], axis=0)
    return out


def lookup_bitmaps(ids, bitmaps_table):
    """ids: (B, T) int -> (B, T, 7, K*5) bool."""
    return bitmaps_table[ids]


# ---- Eval utilities ----

def eval_loss_bitmap(model, ids, bitmaps_table, T, B, n_batches, device, rng):
    """Average loss over n_batches random windows. For bitmap LM."""
    import torch
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for _ in range(n_batches):
            batch = sample_window_batch(ids, T, B, rng)
            input_ids = batch[:, :-1]
            target_ids = torch.tensor(batch[:, 1:], dtype=torch.long, device=device)
            input_bitmaps = torch.tensor(
                lookup_bitmaps(input_ids, bitmaps_table).astype(np.float32),
                device=device,
            )
            _, loss = model(input_bitmaps, target_ids=target_ids)
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def eval_loss_bpe(model, ids, T, B, n_batches, device, rng):
    """Average loss over n_batches random windows. For BPE LM."""
    import torch
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for _ in range(n_batches):
            batch = sample_window_batch(ids, T, B, rng)
            input_ids = torch.tensor(batch[:, :-1], dtype=torch.long, device=device)
            target_ids = torch.tensor(batch[:, 1:], dtype=torch.long, device=device)
            _, loss = model(input_ids, target_ids=target_ids)
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)
