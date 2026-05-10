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


# ---- BPE-Bitmap (Experiment 2) ----

# Special tokens declared in pixenc/bpe.py:31. Rendered as the UNK glyph.
_BPE_SPECIAL_TOKENS = {"<unk>", "<bos>", "<eos>", "<pad>"}

# Per-char fallback for BPE chars not in the dot-matrix font (any non-ASCII
# byte the byte-level pre-tokenizer remapped to a printable Unicode codepoint,
# e.g. 'Ġ' for space, 'Ċ' for newline). '?' exists in the font.
_BPE_CHAR_FALLBACK = "?"


def _normalize_bpe_token(tok_str):
    """Map a raw BPE subword string to a renderable string.

    Byte-level BPE represents a leading space as 'Ġ' (U+0120). We replace it
    with a literal space so the bitmap encoder sees a blank cell at position 0
    — preserving the word-boundary signal. Other non-renderable chars are
    replaced with '?'.

    Returns: (normalized_str, is_special, n_fallback_chars)
    """
    from pixenc.dotmatrix_font import alphabet_dm

    if tok_str in _BPE_SPECIAL_TOKENS:
        return "", True, 0

    out_chars = []
    n_fallback = 0
    for ch in tok_str:
        if ch == "Ġ":      # 'Ġ' — byte-level space marker
            out_chars.append(" ")
        elif ch == "Ċ":    # 'Ċ' — byte-level newline marker
            out_chars.append(" ")
        elif ch in alphabet_dm:
            out_chars.append(ch)
        elif ch.upper() in alphabet_dm:
            out_chars.append(ch.upper())
        else:
            out_chars.append(_BPE_CHAR_FALLBACK)
            n_fallback += 1
    return "".join(out_chars), False, n_fallback


def render_bpe_vocab_bitmaps(tokenizer, k=DEFAULT_K, unk_pattern="checker",
                             verbose=True):
    """Pre-render every BPE subword as a (7, K*5) bitmap.

    Returns: (V, 7, K*5) bool array, indexed directly by BPE token id.

    Notes:
      - Special tokens ('<unk>', '<bos>', '<eos>', '<pad>') get the UNK glyph.
      - 'Ġ' (leading-space marker) renders as a literal space (blank cell)
        at the start of the bitmap, preserving word-boundary info.
      - Any other non-renderable char is substituted with '?'.
    """
    V = tokenizer.get_vocab_size()
    table = np.zeros((V, DM_ROWS, DM_COLS * k), dtype=bool)

    if unk_pattern == "checker":
        rr, cc = np.meshgrid(np.arange(DM_ROWS), np.arange(DM_COLS * k),
                             indexing="ij")
        unk_glyph = ((rr + cc) % 2 == 0)
    elif unk_pattern == "blank":
        unk_glyph = np.zeros((DM_ROWS, DM_COLS * k), dtype=bool)
    elif unk_pattern == "filled":
        unk_glyph = np.ones((DM_ROWS, DM_COLS * k), dtype=bool)
    else:
        raise ValueError(f"unknown unk_pattern: {unk_pattern}")

    n_special = 0
    n_with_fallback = 0
    total_fallback_chars = 0
    for i in range(V):
        tok_str = tokenizer.id_to_token(i)
        norm, is_special, n_fb = _normalize_bpe_token(tok_str)
        if is_special:
            table[i] = unk_glyph
            n_special += 1
            continue
        if n_fb > 0:
            n_with_fallback += 1
            total_fallback_chars += n_fb
        table[i] = render_word_dm(norm, k=k)

    if verbose:
        print(f"  BPE bitmap render: V={V}, K={k}")
        print(f"    special tokens (UNK glyph): {n_special}")
        print(f"    tokens w/ ≥1 fallback char: {n_with_fallback} "
              f"({100*n_with_fallback/V:.1f}%)  "
              f"total fallback chars: {total_fallback_chars}")
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

def eval_loss_bitmap(model, ids, bitmaps_table, T, B, n_batches, device, rng,
                     vocab_bitmaps=None):
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
            kw = {"vocab_bitmaps": vocab_bitmaps} if vocab_bitmaps is not None else {}
            _, loss = model(input_bitmaps, target_ids=target_ids, **kw)
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
