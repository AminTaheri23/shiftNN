"""Sweep K (cells per word) to understand the truncation/budget tradeoff."""

import os
import sys
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pixenc.dotmatrix_font import render_word_dm, DM_BITS
from pixenc.vocab import load_vocab
from pixenc.perturb import perturb


def render_flat(word, k):
    return render_word_dm(word, k=k).flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def metrics_for(K, vocab, rng, sample_pairs=5000, typo_sample=1000):
    bits = DM_BITS * K
    bitmaps = {w: render_flat(w, K) for w in vocab}

    # Collisions
    seen = set()
    coll = 0
    for w, bm in bitmaps.items():
        key = bm.tobytes()
        if key in seen:
            coll += 1
        else:
            seen.add(key)

    # Random pairs
    words = list(bitmaps.keys())
    rd = np.array([hamming(bitmaps[a], bitmaps[b])
                   for a, b in (rng.sample(words, 2) for _ in range(sample_pairs))])

    # Typo: substitution only (insertion/deletion are alignment-shifted; sub is the cleanest signal).
    sample = rng.sample(vocab, min(typo_sample, len(vocab)))
    sub = []
    for w in sample:
        tw = perturb(w, "sub", rng=rng)
        if tw != w:
            sub.append(hamming(bitmaps[w], render_flat(tw, K)))
    sub = np.array(sub)

    fits = sum(1 for w in vocab if len(w) <= K)

    return {
        "K": K, "bits": bits,
        "fits_pct": 100 * fits / len(vocab),
        "collision_pct": 100 * coll / len(vocab),
        "random_mean": rd.mean(),
        "random_norm": rd.mean() / bits,
        "sub_mean": sub.mean(),
        "sub_norm": sub.mean() / bits,
        "ratio": sub.mean() / rd.mean(),
    }


def main():
    print("=== Scanning K (cells per word) ===\n")
    rng = random.Random(42)
    vocab = load_vocab(lowercase_only=True)
    print(f"Vocab: {len(vocab)} lowercase a-z words\n")

    print(f"{'K':>3} {'bits':>5} {'fits %':>7} {'colls %':>8} "
          f"{'rand μ':>8} {'rand/b':>8} {'sub μ':>7} {'sub/b':>8} {'ratio':>7}")
    print("-" * 75)
    for K in [6, 8, 10, 12, 14, 16, 20]:
        m = metrics_for(K, vocab, rng)
        print(f"{m['K']:>3} {m['bits']:>5} {m['fits_pct']:>7.1f} "
              f"{m['collision_pct']:>8.2f} "
              f"{m['random_mean']:>8.1f} {m['random_norm']:>8.4f} "
              f"{m['sub_mean']:>7.2f} {m['sub_norm']:>8.4f} "
              f"{m['ratio']:>7.3f}")


if __name__ == "__main__":
    main()
