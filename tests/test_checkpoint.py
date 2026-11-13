"""Checkpoint round-trip, config-hash guard, and latest_checkpoint."""

from __future__ import annotations

import pytest
import torch

from autotrain.config import load_config
from autotrain.trainer.checkpoint import (
    CheckpointError,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from autotrain.trainer.model import WanxiangModel

from conftest import make_raw_config, write_config


def _build(tmp_path, raw):
    cfg = load_config(write_config(tmp_path, raw))
    model = WanxiangModel(cfg.model, max_seq_len=cfg.data.seq_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return cfg, model, optimizer


class TestCheckpointRoundTrip:
    def test_save_and_load_restores_model(self, tmp_path, tiny_config):
        _, model, optimizer = _build(tmp_path, make_raw_config())
        # perturb weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.01)
        ref_out = model(torch.randint(0, 512, (2, 32)))

        path = save_checkpoint(
            tmp_path / "ck" / "ckpt_step_10.pt",
            step=10, cfg=tiny_config, model=model, optimizer=optimizer,
            scheduler_state={"step_num": 10}, data_cursor=640, metrics={"loss": 3.2},
        )
        assert path.is_file()

        model2 = WanxiangModel(tiny_config.model, max_seq_len=tiny_config.data.seq_len)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        ckpt = load_checkpoint(path, tiny_config, model2, optimizer2)
        assert ckpt.step == 10
        assert ckpt.data_cursor == 640
        assert ckpt.metrics["loss"] == 3.2

        out2 = model2(torch.randint(0, 512, (2, 32)))
        # weights identical -> same structure; compare a weight tensor directly
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
            assert k1 == k2
            assert torch.equal(v1, v2)

    def test_config_hash_mismatch_rejected(self, tmp_path, tiny_config):
        _, model, optimizer = _build(tmp_path, make_raw_config())
        path = save_checkpoint(
            tmp_path / "ck" / "ckpt_step_5.pt",
            step=5, cfg=tiny_config, model=model, optimizer=optimizer,
            scheduler_state={}, data_cursor=0, metrics={},
        )
        other = load_config(write_config(tmp_path, make_raw_config(train={"lr_peak": 9e-3})))
        model2 = WanxiangModel(other.model, max_seq_len=other.data.seq_len)
        with pytest.raises(CheckpointError, match="config hash mismatch"):
            load_checkpoint(path, other, model2)

    def test_strict_config_can_be_relaxed(self, tmp_path, tiny_config):
        _, model, optimizer = _build(tmp_path, make_raw_config())
        path = save_checkpoint(
            tmp_path / "ck" / "ckpt_step_5.pt",
            step=5, cfg=tiny_config, model=model, optimizer=optimizer,
            scheduler_state={}, data_cursor=0, metrics={},
        )
        other = load_config(write_config(tmp_path, make_raw_config(train={"lr_peak": 9e-3})))
        model2 = WanxiangModel(other.model, max_seq_len=other.data.seq_len)
        ckpt = load_checkpoint(path, other, model2, strict_config=False)
        assert ckpt.step == 5

    def test_missing_checkpoint_raises(self, tmp_path, tiny_config):
        model = WanxiangModel(tiny_config.model, max_seq_len=tiny_config.data.seq_len)
        with pytest.raises(CheckpointError, match="not found"):
            load_checkpoint(tmp_path / "none.pt", tiny_config, model)


class TestLatestCheckpoint:
    def test_returns_highest_step(self, tmp_path, tiny_config):
        _, model, optimizer = _build(tmp_path, make_raw_config())
        for step in (10, 5, 20, 15):
            save_checkpoint(
                tmp_path / "ck" / f"ckpt_step_{step}.pt",
                step=step, cfg=tiny_config, model=model, optimizer=optimizer,
                scheduler_state={}, data_cursor=0, metrics={},
            )
        latest = latest_checkpoint(tmp_path / "ck")
        assert latest is not None
        assert latest.stem == "ckpt_step_20"

    def test_empty_dir_returns_none(self, tmp_path):
        assert latest_checkpoint(tmp_path / "empty") is None


class TestAtomicWrite:
    def test_no_tmp_files_left(self, tmp_path, tiny_config):
        _, model, optimizer = _build(tmp_path, make_raw_config())
        save_checkpoint(
            tmp_path / "ck" / "ckpt_step_1.pt",
            step=1, cfg=tiny_config, model=model, optimizer=optimizer,
            scheduler_state={}, data_cursor=0, metrics={},
        )
        files = list((tmp_path / "ck").iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".pt"
