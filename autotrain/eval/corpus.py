"""Deterministic synthetic corpora for M1.5.

Two corpora are frozen under ``evals/`` and verified by checksum on every
evaluation:

* ``structured_eval.bin`` -- a first-order Markov chain: with probability p
  the next token is ``(current + 1) % vocab``, otherwise it is drawn
  uniformly. Its unigram distribution is still uniform, so any improvement
  over ``ln(vocab)`` proves the model learned *structure*, not token frequency.
* ``random_control.bin`` -- i.i.d. uniform tokens. Theoretical optimum is
  exactly ``ln(vocab)``. It is a negative control: if a model appears to
  "learn" here, the measurement is wrong.

These are measurement instruments, not the M2 data pipeline. ``make-corpus``
uses the same generator with a disjoint seed to produce training tokens.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def make_markov_tokens(n: int, vocab: int, p: float, seed: int) -> np.ndarray:
    """Deterministic first-order Markov token stream (uint16)."""
    if n < 2 or vocab < 2 or vocab > 65536 or not 0.0 <= p <= 1.0:
        raise ValueError("invalid corpus parameters")
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.uint16)
    prev = int(rng.integers(0, vocab))
    out[0] = prev
    take_chain = rng.random(n - 1) < p
    noise = rng.integers(0, vocab, size=n - 1, dtype=np.uint16)
    for i in range(1, n):
        if take_chain[i - 1]:
            prev = (prev + 1) % vocab
        else:
            prev = int(noise[i - 1])
        out[i] = prev
    return out


def make_uniform_tokens(n: int, vocab: int, seed: int) -> np.ndarray:
    """Deterministic i.i.d. uniform token stream (uint16)."""
    if n < 1 or vocab < 2 or vocab > 65536:
        raise ValueError("invalid corpus parameters")
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab, size=n, dtype=np.uint16)


def write_tokens(path: str | Path, tokens: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(tokens, dtype="<u2")
    with open(p, "wb") as f:
        f.write(arr.tobytes())


def read_tokens(path: str | Path, max_tokens: int | None = None) -> np.ndarray:
    arr = np.fromfile(str(path), dtype="<u2")
    return arr[:max_tokens] if max_tokens is not None else arr


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
