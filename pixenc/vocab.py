"""Vocab loader. Reads /usr/share/dict/american-english by default."""

import re
from pathlib import Path

DEFAULT_PATH = "/usr/share/dict/american-english"
_LOWER_RE = re.compile(r"^[a-z]+$")


def load_vocab(path=DEFAULT_PATH, lowercase_only=True, max_chars=None):
    """Return a list of words from a newline-separated dictionary file."""
    p = Path(path)
    out = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        w = line.strip()
        if not w:
            continue
        if lowercase_only and not _LOWER_RE.match(w):
            continue
        if max_chars is not None and len(w) > max_chars:
            continue
        out.append(w)
    return out


def length_histogram(words):
    h = {}
    for w in words:
        h[len(w)] = h.get(len(w), 0) + 1
    return h
