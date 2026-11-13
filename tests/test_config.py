"""Config loading, validation, hashing, and derived values."""

from __future__ import annotations

import pytest
import yaml

from autotrain.config import ConfigError, load_config

from conftest import make_raw_config, write_config


class TestLoadConfig:
    def test_valid_config_loads(self, tmp_path):
        cfg = load_config(write_config(tmp_path, make_raw_config()))
        assert cfg.name == "test-4m"
        assert cfg.model.layers == 2
        assert cfg.train.lr_peak == 1e-3

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_missing_key_raises(self, tmp_path):
        raw = make_raw_config()
        del raw["model"]
        with pytest.raises(ConfigError, match="missing config key"):
            load_config(write_config(tmp_path, raw))

    def test_invalid_dim_heads_raises(self, tmp_path):
        raw = make_raw_config(model={"dim": 64, "heads": 5})
        with pytest.raises(ConfigError, match="divisible"):
            load_config(write_config(tmp_path, raw))

    def test_invalid_precision_raises(self, tmp_path):
        raw = make_raw_config(train={"precision": "fp16"})
        with pytest.raises(ConfigError, match="precision"):
            load_config(write_config(tmp_path, raw))

    def test_invalid_lr_raises(self, tmp_path):
        raw = make_raw_config(train={"lr_peak": 0.0})
        with pytest.raises(ConfigError, match="lr_peak"):
            load_config(write_config(tmp_path, raw))

    def test_non_mapping_root_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping"):
            load_config(p)


class TestConfigHash:
    def test_hash_is_deterministic(self, tmp_path):
        cfg1 = load_config(write_config(tmp_path, make_raw_config()))
        cfg2 = load_config(write_config(tmp_path, make_raw_config()))
        assert cfg1.config_hash == cfg2.config_hash

    def test_hash_changes_with_content(self, tmp_path):
        cfg1 = load_config(write_config(tmp_path, make_raw_config()))
        cfg2 = load_config(write_config(tmp_path, make_raw_config(train={"lr_peak": 2e-3})))
        assert cfg1.config_hash != cfg2.config_hash

    def test_hash_ignores_max_steps_budget(self, tmp_path):
        """max_steps is a budget knob: extending it must not invalidate resume."""
        cfg1 = load_config(write_config(tmp_path, make_raw_config(train={"max_steps": 20})))
        cfg2 = load_config(write_config(tmp_path, make_raw_config(train={"max_steps": 100})))
        assert cfg1.config_hash == cfg2.config_hash

    def test_hash_independent_of_key_order(self, tmp_path):
        raw = make_raw_config()
        reordered = {
            "eval": raw["eval"],
            "train": raw["train"],
            "recover": raw["recover"],
            "model": raw["model"],
            "data": raw["data"],
            "seed": raw["seed"],
            "experiment": raw["experiment"],
        }
        cfg1 = load_config(write_config(tmp_path, raw))
        p2 = tmp_path / "reordered.yaml"
        with open(p2, "w", encoding="utf-8") as f:
            yaml.safe_dump(reordered, f)
        cfg2 = load_config(p2)
        assert cfg1.config_hash == cfg2.config_hash


class TestDerived:
    def test_grad_accum_at_least_one(self, tmp_path):
        cfg = load_config(write_config(tmp_path, make_raw_config()))
        d = cfg.derived()
        assert d["grad_accum_steps"] >= 1
        assert d["micro_batch_size"] >= 1
        assert d["effective_batch_tokens"] >= cfg.data.seq_len
