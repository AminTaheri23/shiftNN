"""BPE tokenizer setup using HuggingFace `tokenizers`.

Two modes:
  - train_bpe(...): learn a fresh BPE vocab from a corpus (matched-corpus baseline)
  - load_pretrained(): use GPT-2's pretrained BPE (industry-standard baseline)

For the cleanest comparison against the bitmap LM, train a fresh BPE on
the same corpus the bitmap LM trains on. That way the only thing varying
between the two runs is the input encoding.
"""

from pathlib import Path
import os


def train_bpe(corpus_path, vocab_size=8000, save_dir="artifacts/bpe", min_frequency=2):
    """Train a byte-level BPE tokenizer on `corpus_path` (a .txt file).

    Returns the trained tokenizer object.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<unk>", "<bos>", "<eos>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train([corpus_path], trainer)
    out_path = os.path.join(save_dir, f"bpe_{vocab_size}.json")
    tok.save(out_path)
    return tok, out_path


def load_bpe(path):
    """Load a saved tokenizer from disk."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def load_pretrained_gpt2():
    """Load the GPT-2 byte-level BPE (vocab_size = 50257)."""
    from tokenizers import Tokenizer
    return Tokenizer.from_pretrained("gpt2")


def encode_to_ids(tokenizer, text, add_special=False):
    """Tokenize a string to a list of token ids."""
    enc = tokenizer.encode(text)
    return enc.ids


def stats(tokenizer, sample_text):
    """Quick sanity report: vocab size + tokens-per-word ratio on sample text."""
    ids = encode_to_ids(tokenizer, sample_text)
    n_tokens = len(ids)
    n_words = len(sample_text.split())
    return {
        "vocab_size": tokenizer.get_vocab_size(),
        "tokens": n_tokens,
        "words": n_words,
        "tokens_per_word": n_tokens / max(n_words, 1),
    }
