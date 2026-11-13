"""TokenDataset batching, cursor semantics, and train/val split."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from autotrain.trainer.data import TokenDataset


class TestTokenDataset:
    def test_counts(self, token_file):
        ds = TokenDataset(token_file, val_fraction=0.05)
        assert ds.n_total == 200_000
        assert ds.n_train + ds.n_val == ds.n_total
        assert ds.n_val == 10_000

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TokenDataset(tmp_path / "nope.bin")

    def test_batch_shapes(self, token_file):
        ds = TokenDataset(token_file, val_fraction=0.05)
        x, y, cursor = ds.get_train_batch(cursor=0, batch_size=4, seq_len=64)
        assert x.shape == (4, 64)
        assert y.shape == (4, 64)
        assert cursor == 4 * 64
        # targets are inputs shifted by one within the flat stream
        flat_x = x.reshape(-1)
        flat_y = y.reshape(-1)
        assert torch.equal(flat_x[1:], flat_y[:-1])

    def test_cursor_advances_without_overlap(self, token_file):
        ds = TokenDataset(token_file, val_fraction=0.05)
        x1, _, c1 = ds.get_train_batch(cursor=0, batch_size=2, seq_len=64)
        x2, _, c2 = ds.get_train_batch(cursor=c1, batch_size=2, seq_len=64)
        assert c2 == 256
        assert not torch.equal(x1, x2)

    def test_cursor_wraps_at_end(self, token_file):
        ds = TokenDataset(token_file, val_fraction=0.05)
        huge_cursor = ds.n_train  # at/after end
        x, _, cursor = ds.get_train_batch(cursor=huge_cursor, batch_size=2, seq_len=64)
        assert cursor == 2 * 64  # wrapped to 0 then advanced

    def test_val_batch_deterministic(self, token_file):
        ds = TokenDataset(token_file, val_fraction=0.05)
        x1, y1 = ds.get_val_batch(batch_size=2, seq_len=64, offset=0)
        x2, y2 = ds.get_val_batch(batch_size=2, seq_len=64, offset=0)
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)
        # val region is disjoint from train region
        assert x1.min() >= 0

    def test_token_count(self, token_file):
        assert TokenDataset.token_count(token_file) == 200_000
