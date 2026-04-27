"""Word-level Hamming spread and typo-distance analysis on the dot-matrix font.

Run from project root:  python3 experiments/word_analysis.py
"""

import os
import sys
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pixenc.dotmatrix_font import render_word_dm, DM_BITS
from pixenc.vocab import load_vocab, length_histogram
from pixenc.perturb import perturb


def render_flat(word, k):
    return render_word_dm(word, k=k).flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def summarize(arr, label, total_bits):
    a = np.asarray(arr)
    print(f"  {label}:")
    print(f"    n:      {len(a)}")
    print(f"    min:    {int(a.min())}")
    print(f"    median: {int(np.median(a))}")
    print(f"    mean:   {a.mean():.2f}")
    print(f"    max:    {int(a.max())}")
    print(f"    std:    {a.std():.2f}")
    print(f"    mean / total_bits: {a.mean()/total_bits:.4f}")


def main(K=8, sample_pairs=10000, typo_sample=2000, seed=42):
    bits_per_word = DM_BITS * K
    print(f"=== Word-level analysis (K={K} cells, {bits_per_word} bits/word) ===\n")

    rng = random.Random(seed)
    vocab = load_vocab(lowercase_only=True)
    print(f"Total vocab (lowercase a-z only): {len(vocab)}\n")

    # Length distribution
    lh = length_histogram(vocab)
    cum = 0
    print(f"Length distribution (cumulative %):")
    for L in sorted(lh):
        cum += lh[L]
        marker = " <-- K cap" if L == K else ""
        print(f"  len {L:>2}: {lh[L]:>5}  cum {100*cum/len(vocab):>5.1f}%{marker}")

    fits = sum(1 for w in vocab if len(w) <= K)
    print(f"\nWords fitting in K={K} without truncation: {fits} ({100*fits/len(vocab):.1f}%)")
    print(f"Longer words are truncated for now (lossy; revisit cap-and-split policy).\n")

    # Render vocab
    print("Rendering full vocab...")
    bitmaps = {}
    for w in vocab:
        bitmaps[w] = render_flat(w, K)
    print(f"  {len(bitmaps)} words rendered.\n")

    # Random-pair spread baseline
    print(f"=== Random pair Hamming distances ({sample_pairs} pairs) ===")
    words = list(bitmaps.keys())
    random_dists = []
    for _ in range(sample_pairs):
        a, b = rng.sample(words, 2)
        random_dists.append(hamming(bitmaps[a], bitmaps[b]))
    summarize(random_dists, "random pair", bits_per_word)
    print()

    # Typo-pair distances
    print(f"=== Typo-pair Hamming distances ===")
    typo_words = rng.sample(vocab, min(typo_sample, len(vocab)))
    by_kind = {"sub": [], "ins": [], "del": [], "trans": []}
    for word in typo_words:
        for kind in ["sub", "ins", "del", "trans"]:
            tw = perturb(word, kind, rng=rng)
            if tw == word or not all(c in "abcdefghijklmnopqrstuvwxyz" for c in tw):
                continue
            d = hamming(bitmaps[word], render_flat(tw, K))
            by_kind[kind].append(d)

    all_typo = []
    for kind, ds in by_kind.items():
        all_typo.extend(ds)

    summarize(all_typo, "all typos pooled", bits_per_word)
    print()
    print(f"  By kind:")
    for kind in ["sub", "ins", "del", "trans"]:
        ds = by_kind[kind]
        if ds:
            arr = np.array(ds)
            print(f"    {kind:>5}: n={len(arr):>5}  mean={arr.mean():>5.2f}  "
                  f"median={int(np.median(arr)):>3}  norm={arr.mean()/bits_per_word:.4f}")
    print()

    # Headline ratio
    rd = np.array(random_dists)
    td = np.array(all_typo)
    print(f"=== Headline: typo / random ratio ===")
    print(f"  mean random pair: {rd.mean():.2f}")
    print(f"  mean typo pair:   {td.mean():.2f}")
    print(f"  ratio:            {td.mean()/rd.mean():.3f}")
    print(f"  (lower = better; 0 = perfect typo locality, 1 = no typo signal at all)")
    print()

    # Vocab-level collisions: distinct words rendering to identical bitmaps
    print(f"=== Bitmap collisions across vocab ===")
    seen = {}
    collisions = []
    for w, bm in bitmaps.items():
        key = bm.tobytes()
        if key in seen:
            collisions.append((seen[key], w))
        else:
            seen[key] = w
    print(f"  Distinct bitmaps:   {len(seen)} / {len(bitmaps)} words")
    print(f"  Collision groups:   {len(collisions)}")
    if collisions:
        print(f"  First 15 examples:")
        for a, b in collisions[:15]:
            same_truncated = a[:K] == b[:K]
            tag = "  (truncation collision)" if same_truncated else ""
            print(f"    {a!r:>20} == {b!r}{tag}")
    print()

    # Spread overlap: how many random pairs are closer than the median typo?
    median_typo = int(np.median(td))
    closer_than_median_typo = int((rd <= median_typo).sum())
    print(f"=== Spread overlap ===")
    print(f"  Median typo distance: {median_typo}")
    print(f"  Random pairs at <= that distance: {closer_than_median_typo} / {len(rd)} "
          f"({100*closer_than_median_typo/len(rd):.2f}%)")
    print(f"  (lower = clearer separation between 'typo neighborhood' and 'random pair')")


if __name__ == "__main__":
    main()
