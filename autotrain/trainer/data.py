"""Token data loading with resumable cursor.

Reads flat binary token files (uint16), yields fixed-length batches.
The cursor (absolute token offset) is part of the training state so a
resumed run continues exactly where it stopped, never re-reading and
never skipping data.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    """Flat uint16 token stream with train/val split by fraction of file."""

    def __init__(self, token_file: str | Path, val_fraction: float = 0.01):
        self.path = Path(token_file)
        if not self.path.is_file():
            raise FileNotFoundError(f"token file not found: {self.path}")
        n_tokens = self.path.stat().st_size // 2  # uint16
        if n_tokens < 1024:
            raise ValueError(f"token file too small: {n_tokens} tokens")
        self.n_total = n_tokens
        self.n_val = max(1, int(n_tokens * val_fraction))
        self.n_train = n_tokens - self.n_val

    def _read(self, start: int, count: int) -> np.ndarray:
        with open(self.path, "rb") as f:
            f.seek(start * 2)
            raw = f.read(count * 2)
        if len(raw) < count * 2:
            raise EOFError("unexpected end of token file")
        return np.frombuffer(raw, dtype=np.uint16)

    def get_train_batch(self, cursor: int, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return (inputs, targets, next_cursor).

        Inputs are tokens [cursor, cursor + batch*seq_len), targets are the
        same shifted by one. Wraps around to 0 when reaching the end of the
        training region.
        """
        need = batch_size * seq_len + 1
        if cursor + need > self.n_train:
            cursor = 0
        buf = self._read(cursor, need).astype(np.int64)
        x = torch.from_numpy(buf[:-1].reshape(batch_size, seq_len))
        y = torch.from_numpy(buf[1:].reshape(batch_size, seq_len))
        next_cursor = cursor + batch_size * seq_len
        return x, y, next_cursor

    def get_val_batch(self, batch_size: int, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic validation batch from the tail region."""
        need = batch_size * seq_len + 1
        start = self.n_train + (offset % max(1, self.n_val - need))
        if start + need > self.n_total:
            start = self.n_total - need
        buf = self._read(start, need).astype(np.int64)
        x = torch.from_numpy(buf[:-1].reshape(batch_size, seq_len))
        y = torch.from_numpy(buf[1:].reshape(batch_size, seq_len))
        return x, y

    @staticmethod
    def write_token_file(path: str | Path, tokens: np.ndarray) -> None:
        """Helper for tests and data pipeline: write uint16 token file."""
        arr = np.asarray(tokens, dtype=np.uint16)
        with open(path, "wb") as f:
            f.write(arr.tobytes())

    @staticmethod
    def token_count(path: str | Path) -> int:
        return Path(path).stat().st_size // struct.calcsize("H")
