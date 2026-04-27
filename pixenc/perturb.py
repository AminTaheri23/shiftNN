"""Word perturbations for typo robustness analysis.

The four classic Damerau-Levenshtein operations: substitute, insert, delete,
transpose. Each returns a new string differing from the input by one operation.
"""

import random
import string

ALPHA = string.ascii_lowercase


def perturb(word, kind, rng=None):
    """Apply one perturbation. `kind` in {sub, ins, del, trans}.

    Returns the original word unchanged if the operation isn't applicable
    (e.g., delete on a 1-letter word).
    """
    rng = rng or random
    if kind == "sub" and word:
        i = rng.randrange(len(word))
        choices = ALPHA.replace(word[i], "")
        return word[:i] + rng.choice(choices) + word[i + 1:]
    if kind == "ins":
        i = rng.randrange(len(word) + 1)
        return word[:i] + rng.choice(ALPHA) + word[i:]
    if kind == "del" and len(word) > 1:
        i = rng.randrange(len(word))
        return word[:i] + word[i + 1:]
    if kind == "trans" and len(word) >= 2:
        i = rng.randrange(len(word) - 1)
        if word[i] == word[i + 1]:
            return word
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word
