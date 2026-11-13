"""Experiment configuration loading, validation, and hashing.

The YAML config is the single source of truth for every experiment.
The config hash guards checkpoint resume: a checkpoint may only be
resumed by a byte-identical config, preventing dirty experiments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment config is invalid."""


@dataclass(frozen=True)
class DataConfig:
    mix: str
    seq_len: int
    val_fraction: float = 0.01

    def validate(self) -> None:
        if not self.mix:
            raise ConfigError("data.mix must be a non-empty string")
        if self.seq_len < 32:
            raise ConfigError("data.seq_len must be >= 32")
        if not 0.0 < self.val_fraction < 0.5:
            raise ConfigError("data.val_fraction must be in (0, 0.5)")


@dataclass(frozen=True)
class ModelConfig:
    layers: int
    dim: int
    heads: int
    vocab_size: int

    def validate(self) -> None:
        if self.layers < 1:
            raise ConfigError("model.layers must be >= 1")
        if self.dim < 32:
            raise ConfigError("model.dim must be >= 32")
        if self.dim % self.heads != 0:
            raise ConfigError("model.dim must be divisible by model.heads")
        if self.vocab_size < 256:
            raise ConfigError("model.vocab_size must be >= 256")


@dataclass(frozen=True)
class TrainConfig:
    global_batch_tokens: int
    lr_peak: float
    warmup_steps: int
    precision: str
    ckpt_every_steps: int
    val_every_steps: int
    max_steps: int = 100_000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)

    def validate(self) -> None:
        if self.global_batch_tokens < 1024:
            raise ConfigError("train.global_batch_tokens must be >= 1024")
        if self.lr_peak <= 0 or self.lr_peak > 1:
            raise ConfigError("train.lr_peak must be in (0, 1]")
        if self.warmup_steps < 0:
            raise ConfigError("train.warmup_steps must be >= 0")
        if self.precision not in ("bf16", "fp32"):
            raise ConfigError("train.precision must be 'bf16' or 'fp32'")
        if self.ckpt_every_steps < 1 or self.val_every_steps < 1:
            raise ConfigError("train checkpoint/val intervals must be >= 1")
        if self.max_steps < 1:
            raise ConfigError("train.max_steps must be >= 1")


@dataclass(frozen=True)
class RecoverConfig:
    max_retries: int = 3
    loss_spike_sigma: float = 3.0
    spike_window: int = 50
    early_stop_patience: int = 10

    def validate(self) -> None:
        if self.max_retries < 0:
            raise ConfigError("recover.max_retries must be >= 0")
        if self.loss_spike_sigma <= 0:
            raise ConfigError("recover.loss_spike_sigma must be > 0")
        if self.spike_window < 10:
            raise ConfigError("recover.spike_window must be >= 10")
        if self.early_stop_patience < 1:
            raise ConfigError("recover.early_stop_patience must be >= 1")


@dataclass(frozen=True)
class EvalConfig:
    harness_tasks: tuple[str, ...] = ()
    milestones: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

    def validate(self) -> None:
        for m in self.milestones:
            if not 0.0 < m <= 1.0:
                raise ConfigError("eval.milestones must be in (0, 1]")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    recover: RecoverConfig
    eval: EvalConfig
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("experiment name must be non-empty")
        if self.seed < 0:
            raise ConfigError("seed must be >= 0")
        self.data.validate()
        self.model.validate()
        self.train.validate()
        self.recover.validate()
        self.eval.validate()

    @property
    def config_hash(self) -> str:
        """Deterministic hash of the semantic config (raw dict, sorted keys).

        Volatile keys are excluded: train.max_steps is a budget knob —
        extending it to continue a finished run is a legitimate resume,
        so it must not invalidate the config hash.
        """
        import copy

        canonical_raw = copy.deepcopy(self.raw)
        if isinstance(canonical_raw.get("train"), dict):
            canonical_raw["train"].pop("max_steps", None)
        canonical = json.dumps(canonical_raw, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def derived(self) -> dict[str, Any]:
        """Values derived from config, used by the trainer.

        micro_batch_size is chosen so that micro_batch * seq_len * grad_accum
        approximates global_batch_tokens. grad_accum is always >= 1.
        """
        tokens_per_sample = self.data.seq_len
        target_samples = max(1, self.train.global_batch_tokens // tokens_per_sample)
        micro_batch = max(1, min(target_samples, 32))
        grad_accum = max(1, target_samples // micro_batch)
        return {
            "micro_batch_size": micro_batch,
            "grad_accum_steps": grad_accum,
            "effective_batch_tokens": micro_batch * grad_accum * tokens_per_sample,
        }


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML config."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    try:
        cfg = ExperimentConfig(
            name=str(raw["experiment"]),
            seed=int(raw["seed"]),
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            train=TrainConfig(**raw["train"]),
            recover=RecoverConfig(**raw.get("recover", {})),
            eval=EvalConfig(
                harness_tasks=tuple(raw.get("eval", {}).get("harness_tasks", [])),
                milestones=tuple(raw.get("eval", {}).get("milestones", (0.25, 0.5, 0.75, 1.0))),
            ),
            raw=raw,
        )
    except KeyError as e:
        raise ConfigError(f"missing config key: {e}") from e
    except TypeError as e:
        raise ConfigError(f"invalid config field: {e}") from e

    cfg.validate()
    return cfg
