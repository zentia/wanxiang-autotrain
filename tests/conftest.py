"""Shared fixtures: tiny config and synthetic token file."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from autotrain.config import ExperimentConfig


def make_raw_config(**overrides) -> dict:
    raw = {
        "experiment": "test-4m",
        "seed": 42,
        "data": {"mix": "mix_test", "seq_len": 64, "val_fraction": 0.05},
        "model": {"layers": 2, "dim": 64, "heads": 4, "vocab_size": 512},
        "train": {
            "global_batch_tokens": 4096,
            "lr_peak": 1e-3,
            "warmup_steps": 5,
            "max_steps": 20,
            "precision": "fp32",
            "ckpt_every_steps": 5,
            "val_every_steps": 5,
        },
        "recover": {"max_retries": 2, "loss_spike_sigma": 3.0, "spike_window": 20, "early_stop_patience": 3},
        "eval": {"harness_tasks": [], "milestones": [1.0]},
    }
    for section, values in overrides.items():
        if isinstance(values, dict):
            raw[section].update(values)
        else:
            raw[section] = values
    return raw


def write_config(tmp_path, raw: dict) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "config.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f)
    return str(p)


@pytest.fixture
def tiny_config(tmp_path) -> ExperimentConfig:
    from autotrain.config import load_config

    return load_config(write_config(tmp_path, make_raw_config()))


@pytest.fixture
def token_file(tmp_path) -> str:
    rng = np.random.default_rng(1234)
    tokens = rng.integers(0, 512, size=200_000, dtype=np.uint16)
    p = tmp_path / "tokens.bin"
    from autotrain.trainer.data import TokenDataset

    TokenDataset.write_token_file(p, tokens)
    return str(p)
