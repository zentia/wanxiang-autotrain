"""Trainer Runner: the M1 training execution engine.

Wraps the training loop with:
- warmup + cosine LR schedule (state saved in checkpoint)
- gradient accumulation to reach global_batch_tokens
- periodic checkpoint + validation
- loss spike detection -> rollback to last checkpoint with lower LR
- early stop on stalled val loss
- metric logging to ExperimentStore
- crash-safe resume via latest checkpoint + config hash guard
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import torch

from ..config import ExperimentConfig
from ..store.experiment_store import ExperimentStore
from .checkpoint import CheckpointError, latest_checkpoint, load_checkpoint, save_checkpoint
from .data import TokenDataset
from .model import WanxiangModel
from .monitor import EarlyStopTracker, LossSpikeDetector

log = logging.getLogger(__name__)


class WarmupCosineScheduler:
    """Linear warmup to lr_peak, then cosine decay to 10% of peak."""

    def __init__(self, optimizer: torch.optim.Optimizer, lr_peak: float, warmup_steps: int, max_steps: int):
        self.optimizer = optimizer
        self.lr_peak = lr_peak
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        lr = self._lr_at(self.step_num)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr_peak * (step + 1) / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        return self.lr_peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    def state_dict(self) -> dict:
        return {"step_num": self.step_num}

    def load_state_dict(self, state: dict) -> None:
        self.step_num = state["step_num"]


class TrainerRunner:
    def __init__(
        self,
        cfg: ExperimentConfig,
        token_file: str | Path,
        output_dir: str | Path,
        store: ExperimentStore | None = None,
        device: str | None = None,
    ):
        cfg.validate()
        self.cfg = cfg
        self.token_file = Path(token_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.store = store
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_id: int | None = None

        derived = cfg.derived()
        self.micro_batch = derived["micro_batch_size"]
        self.grad_accum = derived["grad_accum_steps"]

        self.model = WanxiangModel(cfg.model, max_seq_len=cfg.data.seq_len).to(self.device)
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if (param.ndim < 2 or "norm" in name) else decay).append(param)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.train.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.train.lr_peak,
            betas=tuple(cfg.train.betas),
        )
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, cfg.train.lr_peak, cfg.train.warmup_steps, cfg.train.max_steps
        )
        self.dataset = TokenDataset(self.token_file, val_fraction=cfg.data.val_fraction)
        self.spike_detector = LossSpikeDetector(cfg.recover)
        self.early_stop = EarlyStopTracker(cfg.recover.early_stop_patience)
        self.step = 0
        self.data_cursor = 0
        self.autocast_dtype = torch.bfloat16 if cfg.train.precision == "bf16" else torch.float32

    # ------------------------------------------------------------------

    def _log_metrics(self, kv: dict[str, float]) -> None:
        if self.store is not None and self.experiment_id is not None:
            self.store.log_metrics(self.experiment_id, self.step, kv)

    def _validate(self, n_batches: int = 4) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for i in range(n_batches):
                x, y = self.dataset.get_val_batch(self.micro_batch, self.cfg.data.seq_len, offset=i)
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device, dtype=self.autocast_dtype, enabled=self.cfg.train.precision == "bf16"):
                    _, loss = self.model(x, y)
                losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)

    def _train_step(self) -> float:
        """One optimizer step (grad_accum micro-batches). Returns loss."""
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(self.grad_accum):
            x, y, self.data_cursor = self.dataset.get_train_batch(
                self.data_cursor, self.micro_batch, self.cfg.data.seq_len
            )
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device, dtype=self.autocast_dtype, enabled=self.cfg.train.precision == "bf16"):
                _, loss = self.model(x, y)
            (loss / self.grad_accum).backward()
            total_loss += loss.item() / self.grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        self._last_grad_norm = float(grad_norm)
        return total_loss

    def _rollback_with_lower_lr(self) -> None:
        ckpt_path = latest_checkpoint(self.ckpt_dir)
        if ckpt_path is None:
            log.warning("loss spike but no checkpoint to roll back to; continuing")
            return
        ckpt = load_checkpoint(ckpt_path, self.cfg, self.model, self.optimizer)
        self.step = ckpt.step
        self.data_cursor = ckpt.data_cursor
        self.spike_detector.reset()
        for group in self.optimizer.param_groups:
            group["lr"] *= 0.5
        log.warning("rolled back to step %d, lr lowered to %.2e", self.step, group["lr"])

    # ------------------------------------------------------------------

    def resume(self) -> bool:
        """Try to resume from the latest checkpoint. Returns True if resumed."""
        ckpt_path = latest_checkpoint(self.ckpt_dir)
        if ckpt_path is None:
            return False
        try:
            ckpt = load_checkpoint(ckpt_path, self.cfg, self.model, self.optimizer)
        except CheckpointError as e:
            log.error("cannot resume: %s", e)
            raise
        self.scheduler.load_state_dict(ckpt.scheduler_state)
        self.step = ckpt.step
        self.data_cursor = ckpt.data_cursor
        log.info("resumed from %s at step %d", ckpt_path, self.step)
        return True

    def train(self) -> dict[str, Any]:
        """Main training loop. Returns a summary dict."""
        if self.store is not None:
            self.experiment_id = self.store.create_experiment(
                self.cfg.name, self.cfg.config_hash, self.cfg.raw
            )
        resumed = self.resume()
        if not resumed:
            self.step = 0
            self.data_cursor = 0
        self.model.train()
        start_time = time.time()
        tokens_per_step = self.micro_batch * self.grad_accum * self.cfg.data.seq_len
        status = "RUNNING"

        while self.step < self.cfg.train.max_steps:
            loss = self._train_step()

            if self.spike_detector.update(loss):
                self._log_metrics({"loss_spike": 1.0})
                self._rollback_with_lower_lr()
                continue

            if self.step % self.cfg.train.val_every_steps == 0 or self.step == 1:
                val_loss = self._validate()
                self._log_metrics({"loss": loss, "val_loss": val_loss, "lr": self.optimizer.param_groups[0]["lr"], "grad_norm": self._last_grad_norm})
                log.info("step %d | loss %.4f | val %.4f | lr %.2e", self.step, loss, val_loss, self.optimizer.param_groups[0]["lr"])
                if self.early_stop.update(val_loss):
                    status = "CONVERGED_EARLY"
                    log.info("early stop: val loss stalled for %d evaluations", self.early_stop.patience)
                    break
            else:
                self._log_metrics({"loss": loss, "lr": self.optimizer.param_groups[0]["lr"]})

            if self.step % self.cfg.train.ckpt_every_steps == 0:
                save_checkpoint(
                    self.ckpt_dir / f"ckpt_step_{self.step}.pt",
                    self.step, self.cfg, self.model, self.optimizer,
                    self.scheduler.state_dict(), self.data_cursor,
                    metrics={"loss": loss},
                )

        final_val = self._validate()
        save_checkpoint(
            self.ckpt_dir / f"ckpt_step_{self.step}.pt",
            self.step, self.cfg, self.model, self.optimizer,
            self.scheduler.state_dict(), self.data_cursor,
            metrics={"val_loss": final_val},
        )
        if status == "RUNNING":
            status = "DONE"
        if self.store is not None and self.experiment_id is not None:
            self.store.set_status(self.experiment_id, status)

        elapsed = time.time() - start_time
        summary = {
            "status": status,
            "steps": self.step,
            "final_val_loss": final_val,
            "tokens_seen": self.step * tokens_per_step,
            "elapsed_sec": elapsed,
            "tokens_per_sec": self.step * tokens_per_step / max(elapsed, 1e-6),
            "checkpoint": str(self.ckpt_dir / f"ckpt_step_{self.step}.pt"),
        }
        log.info("training finished: %s", summary)
        return summary
