"""Compare 14-seg, 16-seg, and 5x7 dot-matrix fonts.

Two reports:
  (1) Apples-to-apples on A-Z + 0-9 (the only set the segment fonts cover).
  (2) Full dot-matrix alphabet including lowercase + punctuation.

Run from project root:  python3 experiments/compare_all_fonts.py
"""

import os
import sys
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pixenc.lcd_font import (
    alphabet_14, render_char_14,
    alphabet_16, render_char_16,
    CELL_BITS as SEG_BITS,
)
from pixenc.dotmatrix_font import (
    alphabet_dm, render_char_dm,
    DM_BITS, DM_ROWS, DM_COLS,
    UPPER_DIGITS, LOWER, PUNCT,
)
from pixenc.spread_metrics import (
    pairwise_hamming, spread_summary, closest_pairs,
)


def render_alphabet(alphabet, render_fn, keep=None):
    if keep is None:
        return {ch: render_fn(ch) for ch in alphabet if ch != " "}
    return {ch: render_fn(ch) for ch in keep}


def stats_for(glyphs, cell_bits):
    pw = pairwise_hamming(glyphs)
    s = spread_summary(pw)
    fills = [int(g.sum()) for g in glyphs.values()]
    s["cell_bits"] = cell_bits
    s["mean_fill"] = sum(fills) / len(fills)
    s["fill_pct"] = 100 * s["mean_fill"] / cell_bits
    s["min_norm"] = s["min"] / cell_bits
    s["mean_norm"] = s["mean"] / cell_bits
    s["pairs_eq_0"] = sum(1 for d in pw.values() if d == 0)
    s["pairs_le_2"] = sum(1 for d in pw.values() if d <= 2)
    s["pairs_le_4"] = sum(1 for d in pw.values() if d <= 4)
    return pw, s


def print_table(rows, headers):
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(x)) for x in col) for col in cols]
    fmt_row = lambda r: "  ".join(str(x).rjust(w) for x, w in zip(r, widths))
    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))


def stats_row(label, s):
    return [
        label,
        s["cell_bits"], len([1]) if False else s.get("n_glyphs", "-"),
        s["min"], f"{s['min_norm']:.3f}",
        f"{s['mean']:.2f}", f"{s['mean_norm']:.3f}",
        f"{s['std']:.2f}", f"{s['fill_pct']:.1f}",
        s["pairs_eq_0"], s["pairs_le_2"], s["pairs_le_4"],
    ]


def add_n(s, n):
    s["n_glyphs"] = n
    return s


def main():
    # Apples-to-apples: A-Z + 0-9 across all three fonts.
    seg_chars = [ch for ch in alphabet_14 if ch != " "]
    g14 = render_alphabet(alphabet_14, render_char_14, keep=seg_chars)
    g16 = render_alphabet(alphabet_16, render_char_16, keep=seg_chars)
    gdm_basic = render_alphabet(alphabet_dm, render_char_dm, keep=UPPER_DIGITS)

    pw14, s14 = stats_for(g14, SEG_BITS)
    pw16, s16 = stats_for(g16, SEG_BITS)
    pwdm_b, sdm_b = stats_for(gdm_basic, DM_BITS)
    add_n(s14, len(g14)); add_n(s16, len(g16)); add_n(sdm_b, len(gdm_basic))

    print("=" * 78)
    print("REPORT 1: Apples-to-apples on A-Z + 0-9")
    print("=" * 78)
    headers = ["font", "cell bits", "n", "d_min", "d_min/bits",
               "mean", "mean/bits", "std", "fill %", "p=0", "p<=2", "p<=4"]
    print_table([
        stats_row("14-seg", s14),
        stats_row("16-seg", s16),
        stats_row("5x7 dotmat (tuned)", sdm_b),
    ], headers)
    print()

    print("--- 5x7 dot-matrix (tuned): closest 12 pairs ---")
    for (c1, c2), d in closest_pairs(pwdm_b, n=12):
        print(f"  {c1!r:>4} <-> {c2!r:>4}: distance {d:>2}  ({100*d/DM_BITS:.1f}% of cell)")
    print()

    print("--- Verifying targeted fixes (was distance 2 in untuned dot-matrix) ---")
    for c1, c2 in [("D", "O"), ("M", "N"), ("S", "5"), ("B", "8"), ("F", "P"), ("P", "R")]:
        d = pwdm_b[(c1, c2)] if (c1, c2) in pwdm_b else pwdm_b[(c2, c1)]
        print(f"  {c1} vs {c2}: {d}")
    print()

    # ---- Full alphabet report ----
    gdm_full = render_alphabet(alphabet_dm, render_char_dm)
    pwdm_f, sdm_f = stats_for(gdm_full, DM_BITS)
    add_n(sdm_f, len(gdm_full))

    print("=" * 78)
    print("REPORT 2: Full dot-matrix alphabet (upper + lower + digits + punct)")
    print("=" * 78)
    print(f"  total glyphs:    {len(gdm_full)}")
    print(f"  uppercase:       {sum(1 for c in gdm_full if c.isalpha() and c.isupper())}")
    print(f"  lowercase:       {sum(1 for c in gdm_full if c.isalpha() and c.islower())}")
    print(f"  digits:          {sum(1 for c in gdm_full if c.isdigit())}")
    print(f"  punctuation:     {sum(1 for c in gdm_full if not c.isalnum())}")
    print()
    print_table([stats_row("5x7 dotmat", sdm_f)], headers)
    print()

    print("--- Full alphabet: closest 20 pairs ---")
    for (c1, c2), d in closest_pairs(pwdm_f, n=20):
        print(f"  {c1!r:>4} <-> {c2!r:>4}: distance {d:>2}  ({100*d/DM_BITS:.1f}% of cell)")
    print()

    # Pairs of uppercase vs lowercase (case-confusable) — interesting subset.
    lower_set = set(LOWER)
    upper_set = set(c for c in alphabet_dm if c.isalpha() and c.isupper())
    case_pairs = []
    for (c1, c2), d in pwdm_f.items():
        s1, s2 = c1, c2
        if (s1 in upper_set and s2 == s1.lower()) or (s2 in upper_set and s1 == s2.lower()):
            case_pairs.append(((s1, s2), d))
    case_pairs.sort(key=lambda x: x[1])
    print("--- Same-letter case pairs (X vs x) ---")
    for (c1, c2), d in case_pairs[:10]:
        print(f"  {c1!r:>4} <-> {c2!r:>4}: distance {d:>2}  ({100*d/DM_BITS:.1f}% of cell)")
    print()

    print("--- Glyph gallery: tuned letters (D, M, S, P, 8, 0) ---\n")
    for ch in ["D", "M", "S", "P", "8", "0", "O", "N", "5"]:
        bitmap = gdm_full[ch] if ch in gdm_full else None
        if bitmap is None:
            continue
        print(f"  {ch}:")
        for row in bitmap:
            print("    " + "".join("#" if px else "." for px in row))
        print()


if __name__ == "__main__":
    main()
