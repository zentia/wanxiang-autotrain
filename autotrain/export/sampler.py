"""Minimal greedy sampler for M1.5 bundles.

No KV cache, no batching, no stochastic sampling: one sequence, one token at
a time, full-sequence forward each step. This exists so a trained checkpoint
can actually be observed before M2/M3 are built. It is an observation tool,
not a production inference path.
"""

from __future__ import annotations

from typing import Iterable

import torch


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    prompt_ids: Iterable[int],
    max_new_tokens: int,
    *,
    eos_token_id: int | None = None,
) -> list[int]:
    """Deterministically extend a prompt; return only the new token ids."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    ids = [int(t) for t in prompt_ids]
    if not ids:
        raise ValueError("prompt_ids must not be empty")

    device = next(model.parameters()).device
    cap = int(getattr(model, "max_seq_len", len(ids) + max_new_tokens))
    window = ids[-cap:]
    vocab = int(model.cfg.vocab_size)
    generated: list[int] = []

    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            idx = torch.tensor([window], dtype=torch.long, device=device)
            logits, _loss = model(idx)
            next_id = int(logits[0, -1].argmax().item())
            if not 0 <= next_id < vocab:
                raise RuntimeError(f"sampler produced out-of-range token id {next_id}")
            generated.append(next_id)
            if eos_token_id is not None and next_id == eos_token_id:
                break
            window = (window + [next_id])[-cap:]
    finally:
        if was_training:
            model.train()
    return generated
