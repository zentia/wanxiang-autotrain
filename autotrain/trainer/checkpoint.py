"""Checkpoint save/load with full resumability.

A checkpoint contains everything needed to continue training bit-for-bit:
model, optimizer, scheduler, data cursor, RNG states (python/numpy/cuda),
step count, and the config hash. Resume refuses a config-hash mismatch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import ExperimentConfig


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing or incompatible."""


@dataclass
class Checkpoint:
    step: int
    config_hash: str
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict
    data_cursor: int
    rng_states: dict
    metrics: dict


def capture_rng_states() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_states(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"].cpu() if isinstance(states["torch"], torch.Tensor) else states["torch"])
    if states.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def save_checkpoint(
    path: str | Path,
    step: int,
    cfg: ExperimentConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_state: dict,
    data_cursor: int,
    metrics: dict,
) -> Path:
    """Atomically write a checkpoint (tmp file + rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(
        step=step,
        config_hash=cfg.config_hash,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler_state,
        data_cursor=data_cursor,
        rng_states=capture_rng_states(),
        metrics=metrics,
    )
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(p)
    return p


def load_checkpoint(
    path: str | Path,
    cfg: ExperimentConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict_config: bool = True,
) -> Checkpoint:
    """Load a checkpoint into model/optimizer, restoring RNG and cursor.

    Raises CheckpointError on missing file or config-hash mismatch.
    """
    p = Path(path)
    if not p.is_file():
        raise CheckpointError(f"checkpoint not found: {p}")
    ckpt: Checkpoint = torch.load(p, map_location="cpu", weights_only=False)
    if strict_config and ckpt.config_hash != cfg.config_hash:
        raise CheckpointError(
            f"config hash mismatch: checkpoint was written with "
            f"{ckpt.config_hash[:12]}..., current config is {cfg.config_hash[:12]}...; "
            f"refusing to resume a dirty experiment"
        )
    model.load_state_dict(ckpt.model_state)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt.optimizer_state)
    restore_rng_states(ckpt.rng_states)
    return ckpt


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    """Return the checkpoint with the highest step in a directory, or None."""
    d = Path(ckpt_dir)
    if not d.is_dir():
        return None
    candidates = sorted(
        d.glob("ckpt_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return candidates[-1] if candidates else None
