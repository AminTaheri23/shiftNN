"""Head-to-head comparison of 14- vs 16-segment LCD fonts in the same 7x9 cell.

Run from project root:  python3 experiments/compare_14_vs_16.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pixenc.lcd_font import (
    alphabet_14, render_char_14,
    alphabet_16, render_char_16,
    CELL_BITS,
)
from pixenc.spread_metrics import (
    pairwise_hamming, spread_summary, closest_pairs, histogram,
)


def render_alphabet(alphabet, render_fn):
    return {ch: render_fn(ch) for ch in alphabet if ch != " "}


def stats_for(glyphs):
    pw = pairwise_hamming(glyphs)
    s = spread_summary(pw)
    fills = [int(g.sum()) for g in glyphs.values()]
    s["mean_fill"] = sum(fills) / len(fills)
    s["fill_pct"] = 100 * s["mean_fill"] / CELL_BITS
    s["pairs_le_4"] = sum(1 for d in pw.values() if d <= 4)
    s["pairs_le_2"] = sum(1 for d in pw.values() if d <= 2)
    s["pairs_eq_0"] = sum(1 for d in pw.values() if d == 0)
    return pw, s


def print_stats(label, s):
    print(f"--- {label} ---")
    keys = ["min", "median", "mean", "max", "std", "num_pairs",
            "pairs_eq_0", "pairs_le_2", "pairs_le_4", "fill_pct"]
    for k in keys:
        v = s[k]
        if isinstance(v, float):
            print(f"  {k:>12}: {v:.2f}")
        else:
            print(f"  {k:>12}: {v}")
    print()


def print_collision_diff(label, pw, top_n=8):
    print(f"--- {label}: closest {top_n} pairs ---")
    for (c1, c2), d in closest_pairs(pw, n=top_n):
        print(f"  {c1} <-> {c2}: distance {d}")
    print()


def side_by_side_render(ch, w14, w16):
    """Print 14- and 16-seg renderings of a single char side by side."""
    print(f"  char: {ch}    14-seg               16-seg")
    for r in range(9):
        l14 = "".join("#" if px else "." for px in w14[r])
        l16 = "".join("#" if px else "." for px in w16[r])
        print(f"             {l14}              {l16}")
    print()


def main():
    g14 = render_alphabet(alphabet_14, render_char_14)
    g16 = render_alphabet(alphabet_16, render_char_16)

    print(f"=== Cell: 7 x 9 = {CELL_BITS} bits ===")
    print(f"=== Glyphs: {len(g14)} chars (uppercase A-Z + 0-9) ===\n")

    pw14, s14 = stats_for(g14)
    pw16, s16 = stats_for(g16)

    print_stats("14-segment", s14)
    print_stats("16-segment", s16)

    print("--- Delta (16 minus 14) ---")
    for k in ["min", "median", "mean", "max", "std",
              "pairs_eq_0", "pairs_le_2", "pairs_le_4", "fill_pct"]:
        d = s16[k] - s14[k]
        sign = "+" if d > 0 else ""
        if isinstance(d, float):
            print(f"  {k:>12}: {sign}{d:.2f}")
        else:
            print(f"  {k:>12}: {sign}{d}")
    print()

    print_collision_diff("14-segment", pw14)
    print_collision_diff("16-segment", pw16)

    print("=== Targeted side-by-side: previously-colliding pairs ===\n")
    for ch in ["S", "5", "B", "D", "G", "6"]:
        side_by_side_render(ch, g14[ch], g16[ch])

    print("=== Spread histograms (bin width 2) ===")
    h14 = histogram(pw14, bin_width=2)
    h16 = histogram(pw16, bin_width=2)

    edges = sorted(set([(lo, hi) for lo, hi, _ in h14] + [(lo, hi) for lo, hi, _ in h16]))
    by_edge_14 = {(lo, hi): c for lo, hi, c in h14}
    by_edge_16 = {(lo, hi): c for lo, hi, c in h16}
    print(f"  {'bin':>10}  {'14-seg':>8}  {'16-seg':>8}")
    for lo, hi in edges:
        c14 = by_edge_14.get((lo, hi), 0)
        c16 = by_edge_16.get((lo, hi), 0)
        print(f"  [{lo:2d},{hi:2d})  {c14:>8}  {c16:>8}")


if __name__ == "__main__":
    main()
