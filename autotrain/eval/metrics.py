"""Objective, judgment-free metrics for M1.5.

No LLM-as-a-judge, no subjective quality score. Every number here is derived
from cross entropy against a synthetic distribution with a computable
optimum, plus a deterministic greedy-generation check.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from .evalspec import GEN_PROMPT_LEN, GEN_TOKENS, MARKOV_P

if TYPE_CHECKING:
    from ..trainer.model import WanxiangModel


def uniform_baseline_nats(vocab_size: int) -> float:
    """Cross entropy of the best possible model on i.i.d. uniform tokens."""
    return math.log(vocab_size)


def markov_optimal_nats(vocab_size: int, p: float = MARKOV_P) -> float:
    """Cross entropy of the true first-order Markov distribution.

    The generator emits ``(cur+1) % V`` with probability p, and otherwise
    draws uniformly from all V tokens. So the successor distribution is:
      target token: p + (1-p)/V
      each other token: (1-p)/V
    """
    p_hit = p + (1.0 - p) / vocab_size
    p_other = (1.0 - p) / vocab_size
    # p == 1 makes the chain deterministic: the noise terms have probability
    # zero and 0*log(0) must contribute 0, not a math-domain error.
    noise_term = (
        (vocab_size - 1) * p_other * math.log(p_other)
        if p_other > 0.0
        else 0.0
    )
    return -(p_hit * math.log(p_hit) + noise_term)


def nats_to_bits(nats: float) -> float:
    return nats / math.log(2.0)


def safe_perplexity(nats: float) -> float:
    return math.exp(min(nats, 700.0))


def structure_gain_ratio(nats: float, baseline: float, optimal: float) -> float:
    """Fraction of the learnable structure actually captured (0..1+)."""
    if baseline <= optimal:
        return 0.0
    return (baseline - nats) / (baseline - optimal)


@torch.no_grad()
def measure_nats(
    model: "WanxiangModel",
    tokens: np.ndarray,
    seq_len: int,
    batch: int,
    windows: int,
    device: str,
) -> float:
    """Mean token cross entropy over fixed, non-overlapping windows."""
    need = windows * batch * seq_len + 1
    if len(tokens) < need:
        raise ValueError(f"corpus too short: need {need} tokens, have {len(tokens)}")
    if seq_len > int(model.max_seq_len):
        raise ValueError(f"seq_len {seq_len} exceeds model max_seq_len {model.max_seq_len}")

    flat = np.asarray(tokens[:need], dtype=np.int64)
    x_all = torch.from_numpy(flat[:-1].reshape(windows * batch, seq_len))
    y_all = torch.from_numpy(flat[1:].reshape(windows * batch, seq_len))
    losses: list[float] = []
    for i in range(windows):
        x = x_all[i * batch : (i + 1) * batch].to(device)
        y = y_all[i * batch : (i + 1) * batch].to(device)
        _logits, loss = model(x, y)
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


@torch.no_grad()
def greedy_transition_accuracy(
    model: "WanxiangModel", tokens: np.ndarray, vocab_size: int
) -> float:
    """Share of greedy-generated steps that follow the Markov successor rule.

    A model that learned the rule keeps emitting ``+1`` chains, so this tends
    to 1.0. An untrained model lands there with probability ~1/V.
    """
    from ..export.sampler import greedy_generate

    prompt = [int(t) for t in tokens[:GEN_PROMPT_LEN]]
    if not prompt:
        raise ValueError("empty prompt for greedy check")
    generated = greedy_generate(model, prompt, GEN_TOKENS)
    prev = prompt[-1]
    hits = 0
    for token in generated:
        if token == (prev + 1) % vocab_size:
            hits += 1
        prev = token
    return hits / len(generated)
