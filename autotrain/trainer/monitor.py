"""Training monitors: loss spike detection and early-stop tracking.

LossSpikeDetector keeps a sliding window of recent losses and flags a
spike when the current loss exceeds window mean + sigma * std. The
trainer responds by rolling back to the last checkpoint and lowering LR.

EarlyStopTracker marks convergence when val loss has not improved for
`patience` consecutive evaluations.
"""

from __future__ import annotations

from collections import deque

from ..config import RecoverConfig


class LossSpikeDetector:
    def __init__(self, cfg: RecoverConfig):
        self.sigma = cfg.loss_spike_sigma
        self.window: deque[float] = deque(maxlen=cfg.spike_window)
        self.min_history = max(10, cfg.spike_window // 2)

    def update(self, loss: float) -> bool:
        """Record a loss; return True if it is a spike."""
        is_spike = False
        if len(self.window) >= self.min_history:
            mean = sum(self.window) / len(self.window)
            var = sum((x - mean) ** 2 for x in self.window) / len(self.window)
            std = max(var**0.5, 1e-8)
            if loss > mean + self.sigma * std:
                is_spike = True
        self.window.append(loss)
        return is_spike

    def reset(self) -> None:
        self.window.clear()


class EarlyStopTracker:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.bad_rounds = 0

    def update(self, val_loss: float) -> bool:
        """Record a val loss; return True if training should stop early."""
        if val_loss < self.best - 1e-5:
            self.best = val_loss
            self.bad_rounds = 0
        else:
            self.bad_rounds += 1
        return self.bad_rounds >= self.patience

    @property
    def converged_early(self) -> bool:
        return self.bad_rounds >= self.patience
