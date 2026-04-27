"""Alphabet-wide Hamming spread analysis for the 14-segment LCD font.

Run from the project root:  python3 experiments/analyze_alphabet.py
"""

import os
import sys

# Make the project root importable regardless of cwd.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pixenc.lcd_font import alphabet_14, render_char_14, CELL_BITS
from pixenc.spread_metrics import (
    pairwise_hamming,
    spread_summary,
    closest_pairs,
    histogram,
)


def print_gallery(glyphs, cols_per_row=6):
    chars = sorted(glyphs.keys())
    for i in range(0, len(chars), cols_per_row):
        row_chars = chars[i:i + cols_per_row]
        for r in range(9):
            line_parts = []
            for ch in row_chars:
                bitmap = glyphs[ch]
                line_parts.append("".join("#" if px else "." for px in bitmap[r]))
            print("    ".join(line_parts))
        print("    ".join(f"   {ch}   " for ch in row_chars))
        print()


def main():
    chars = [ch for ch in alphabet_14 if ch != " "]
    glyphs = {ch: render_char_14(ch) for ch in chars}

    print(f"=== 14-segment font, 7x9 cell ({CELL_BITS} bits per glyph) ===")
    print(f"Glyphs rendered: {len(glyphs)} (uppercase A-Z + 0-9)")
    print()

    print("=== Glyph gallery ===")
    print_gallery(glyphs)

    print("=== Pairwise Hamming spread ===")
    pw = pairwise_hamming(glyphs)
    s = spread_summary(pw)
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k:>10}: {v:.2f}")
        else:
            print(f"  {k:>10}: {v}")
    print()

    print("=== Distance histogram (bin width 2) ===")
    for lo, hi, count in histogram(pw, bin_width=2):
        bar = "#" * min(count, 60)
        print(f"  [{lo:2d}, {hi:2d}): {count:4d}  {bar}")
    print()

    print("=== Top 12 closest (most confusable) pairs ===")
    for (c1, c2), d in closest_pairs(pw, n=12):
        print(f"  {c1} <-> {c2}: distance {d}")
    print()

    print("=== Bits-per-glyph distribution ===")
    bits = sorted([(ch, int(g.sum())) for ch, g in glyphs.items()], key=lambda x: x[1])
    print(f"  min  {bits[0][0]}: {bits[0][1]} bits")
    print(f"  max  {bits[-1][0]}: {bits[-1][1]} bits")
    mean_bits = sum(b for _, b in bits) / len(bits)
    print(f"  mean: {mean_bits:.1f} / {CELL_BITS} bits ({100*mean_bits/CELL_BITS:.1f}% fill)")


if __name__ == "__main__":
    main()
