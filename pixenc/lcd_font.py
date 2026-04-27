"""14-segment LCD-style font rendered in a 7-col x 9-row cell.

Each glyph is built by OR-ing a subset of 14 segment bitmaps. The cell is
7 columns wide and 9 rows tall (63 bits per cell). Standard 14-segment
display geometry, generalized to a pixel grid so each segment is cleanly
addressable.
"""

import numpy as np


CELL_ROWS = 9
CELL_COLS = 7
CELL_BITS = CELL_ROWS * CELL_COLS


def _empty_mask():
    return np.zeros((CELL_ROWS, CELL_COLS), dtype=bool)


def _line_mask(points):
    m = _empty_mask()
    for r, c in points:
        m[r, c] = True
    return m


# 14 segment masks. Standard segment names from real 14-seg displays.
segments_14 = {
    "a":  _line_mask([(0, c) for c in range(1, 6)]),       # top horizontal
    "b":  _line_mask([(r, 6) for r in range(1, 4)]),       # top-right vertical
    "c":  _line_mask([(r, 6) for r in range(5, 8)]),       # bottom-right vertical
    "d":  _line_mask([(8, c) for c in range(1, 6)]),       # bottom horizontal
    "e":  _line_mask([(r, 0) for r in range(5, 8)]),       # bottom-left vertical
    "f":  _line_mask([(r, 0) for r in range(1, 4)]),       # top-left vertical
    "g1": _line_mask([(4, 1), (4, 2)]),                    # middle-left horizontal
    "g2": _line_mask([(4, 4), (4, 5)]),                    # middle-right horizontal
    "h":  _line_mask([(1, 1), (2, 2), (3, 3)]),            # top-left diagonal
    "i":  _line_mask([(r, 3) for r in range(1, 4)]),       # top-middle vertical
    "j":  _line_mask([(1, 5), (2, 4), (3, 3)]),            # top-right diagonal
    "k":  _line_mask([(5, 3), (6, 2), (7, 1)]),            # bottom-left diagonal
    "l":  _line_mask([(r, 3) for r in range(5, 8)]),       # bottom-middle vertical
    "m":  _line_mask([(5, 3), (6, 4), (7, 5)]),            # bottom-right diagonal
}


# Glyph table: character -> set of active segments.
# Standard 14-segment display assignments (uppercase + digits).
alphabet_14 = {
    "A": {"a", "b", "c", "e", "f", "g1", "g2"},
    "B": {"a", "b", "c", "d", "i", "l", "g2"},
    "C": {"a", "d", "e", "f"},
    "D": {"a", "b", "c", "d", "i", "l"},
    "E": {"a", "d", "e", "f", "g1", "g2"},
    "F": {"a", "e", "f", "g1", "g2"},
    "G": {"a", "c", "d", "e", "f", "g2"},
    "H": {"b", "c", "e", "f", "g1", "g2"},
    "I": {"a", "d", "i", "l"},
    "J": {"b", "c", "d", "e"},
    "K": {"e", "f", "g1", "j", "m"},
    "L": {"d", "e", "f"},
    "M": {"b", "c", "e", "f", "h", "j"},
    "N": {"b", "c", "e", "f", "h", "m"},
    "O": {"a", "b", "c", "d", "e", "f"},
    "P": {"a", "b", "e", "f", "g1", "g2"},
    "Q": {"a", "b", "c", "d", "e", "f", "m"},
    "R": {"a", "b", "e", "f", "g1", "g2", "m"},
    "S": {"a", "c", "d", "f", "g1", "g2"},
    "T": {"a", "i", "l"},
    "U": {"b", "c", "d", "e", "f"},
    "V": {"e", "f", "j", "k"},
    "W": {"b", "c", "e", "f", "k", "m"},
    "X": {"h", "j", "k", "m"},
    "Y": {"h", "j", "l"},
    "Z": {"a", "d", "j", "k"},
    "0": {"a", "b", "c", "d", "e", "f", "j", "m"},
    "1": {"b", "c"},
    "2": {"a", "b", "d", "e", "g1", "g2"},
    "3": {"a", "b", "c", "d", "g2"},
    "4": {"b", "c", "f", "g1", "g2"},
    "5": {"a", "c", "d", "f", "g1", "g2"},
    "6": {"a", "c", "d", "e", "f", "g1", "g2"},
    "7": {"a", "b", "c"},
    "8": {"a", "b", "c", "d", "e", "f", "g1", "g2"},
    "9": {"a", "b", "c", "d", "f", "g1", "g2"},
    " ": set(),
}


def render_char_14(ch):
    """Render a single character to a (CELL_ROWS, CELL_COLS) bool bitmap."""
    key = ch.upper()
    if key not in alphabet_14:
        raise KeyError(f"Character {ch!r} not in 14-segment alphabet")
    out = _empty_mask()
    for seg in alphabet_14[key]:
        out |= segments_14[seg]
    return out


def render_word_14(word, k=8):
    """Render a word into a (CELL_ROWS, CELL_COLS * k) bitmap.

    Right-pads with blank cells. Words longer than k are truncated; a
    cap-and-split policy will be added later.
    """
    out = np.zeros((CELL_ROWS, CELL_COLS * k), dtype=bool)
    for idx, ch in enumerate(word[:k]):
        out[:, idx * CELL_COLS:(idx + 1) * CELL_COLS] = render_char_14(ch)
    return out


# ---------------------------------------------------------------------------
# 16-segment font: same 7x9 cell, with `a` and `d` each split into two halves.
# ---------------------------------------------------------------------------

# 16 segment masks. Difference from 14-seg: `a` -> `a1` + `a2`, `d` -> `d1` + `d2`.
segments_16 = {
    "a1": _line_mask([(0, 1), (0, 2)]),                    # top-left horizontal
    "a2": _line_mask([(0, 4), (0, 5)]),                    # top-right horizontal
    "b":  segments_14["b"],
    "c":  segments_14["c"],
    "d1": _line_mask([(8, 1), (8, 2)]),                    # bottom-left horizontal
    "d2": _line_mask([(8, 4), (8, 5)]),                    # bottom-right horizontal
    "e":  segments_14["e"],
    "f":  segments_14["f"],
    "g1": segments_14["g1"],
    "g2": segments_14["g2"],
    "h":  segments_14["h"],
    "i":  segments_14["i"],
    "j":  segments_14["j"],
    "k":  segments_14["k"],
    "l":  segments_14["l"],
    "m":  segments_14["m"],
}


def _expand_split(segs):
    """Translate a 14-seg glyph to 16-seg by replacing `a` and `d` with halves."""
    out = set()
    for s in segs:
        if s == "a":
            out.update({"a1", "a2"})
        elif s == "d":
            out.update({"d1", "d2"})
        else:
            out.add(s)
    return out


# Auto-translate every letter, then override the ones where asymmetric
# horizontals can buy us distinguishability.
alphabet_16 = {ch: _expand_split(segs) for ch, segs in alphabet_14.items()}

# Asymmetric overrides — this is where 16-seg earns its extra bits.
# S: cursive S using only top-left + bottom-right horizontals (breaks the S=5 tie).
alphabet_16["S"] = {"a1", "f", "g1", "g2", "c", "d2"}
# Z: traditional Z with full top, full bottom, and the two anti-diagonals.
# (Already correct from auto-translate, kept for clarity.)
alphabet_16["Z"] = {"a1", "a2", "d1", "d2", "j", "k"}


def render_char_16(ch):
    key = ch.upper()
    if key not in alphabet_16:
        raise KeyError(f"Character {ch!r} not in 16-segment alphabet")
    out = _empty_mask()
    for seg in alphabet_16[key]:
        out |= segments_16[seg]
    return out


def render_word_16(word, k=8):
    out = np.zeros((CELL_ROWS, CELL_COLS * k), dtype=bool)
    for idx, ch in enumerate(word[:k]):
        out[:, idx * CELL_COLS:(idx + 1) * CELL_COLS] = render_char_16(ch)
    return out


def ascii_render(bitmap, on="#", off="."):
    """Pretty-print a bitmap as ASCII for debugging."""
    return "\n".join("".join(on if px else off for px in row) for row in bitmap)
