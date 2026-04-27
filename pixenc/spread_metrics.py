"""Hamming-distance spread metrics for an LCD font alphabet."""

import numpy as np


def hamming_distance(a, b):
    return int(np.count_nonzero(a.flatten() != b.flatten()))


def pairwise_hamming(glyphs):
    """All unordered pairwise distances. Returns dict {(c1, c2): dist}."""
    items = list(glyphs.items())
    out = {}
    for i in range(len(items)):
        c1, g1 = items[i]
        for j in range(i + 1, len(items)):
            c2, g2 = items[j]
            out[(c1, c2)] = hamming_distance(g1, g2)
    return out


def spread_summary(pairwise):
    distances = np.array(list(pairwise.values()))
    return {
        "min": int(distances.min()),
        "max": int(distances.max()),
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "std": float(distances.std()),
        "num_pairs": len(distances),
    }


def closest_pairs(pairwise, n=10):
    return sorted(pairwise.items(), key=lambda x: x[1])[:n]


def histogram(pairwise, bin_width=2):
    distances = np.array(list(pairwise.values()))
    lo = int(distances.min())
    hi = int(distances.max())
    edges = np.arange(lo, hi + bin_width + 1, bin_width)
    counts, _ = np.histogram(distances, bins=edges)
    return list(zip(edges[:-1].tolist(), edges[1:].tolist(), counts.tolist()))
