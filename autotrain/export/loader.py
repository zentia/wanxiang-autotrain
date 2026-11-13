"""M1.5 bundle loader: bundle -> WanxiangModel, with full contract checks.

The loader is deliberately independent of runner/checkpoint/data: an
evaluator can reconstruct a model from a bundle without importing the
training loop at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from ..config import ModelConfig
from ..trainer.model import WanxiangModel
from .contract import (
    MANIFEST_FILE,
    TIED_ALIASES,
    WEIGHTS_FILE,
    ContractError,
    canonical_json,
    checkpoint_state,
    sha256_bytes,
    sha256_file,
    tensor_bytes,
    validate_bundle_files,
)


@dataclass(frozen=True)
class BundleInfo:
    bundle_dir: Path
    config: dict[str, Any]
    manifest: dict[str, Any]
    step: int
    vocab_size: int
    max_seq_len: int
    dtype: str
    device: str
    manifest_file_hash: str
    config_hash: str


def default_device() -> str:
    """cuda -> mps -> cpu. Unlike M1's runner, M1.5 supports Apple Silicon."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_bundle(
    bundle_dir: str | Path, *, device: str | None = None, verify: bool = True
) -> tuple[WanxiangModel, BundleInfo]:
    """Load and verify a bundle; return ``(model, info)``.

    Verification covers file checksums, config schema, tensor inventory
    (name/shape/dtype/per-tensor sha256) and the tied-weight alias.
    """
    d = Path(bundle_dir)
    config, _tokenizer, manifest = validate_bundle_files(d)
    model_doc = config["model"]

    model_cfg = ModelConfig(
        layers=model_doc["layers"],
        dim=model_doc["dim"],
        heads=model_doc["heads"],
        vocab_size=model_doc["vocab_size"],
    )
    model = WanxiangModel(model_cfg, max_seq_len=model_doc["max_seq_len"])

    state = load_file(str(d / WEIGHTS_FILE))
    inventory = {rec["name"]: rec for rec in manifest.get("tensors", [])}
    if set(state) != set(inventory):
        raise ContractError(
            f"tensor inventory mismatch: missing={sorted(set(inventory) - set(state))} "
            f"extra={sorted(set(state) - set(inventory))}"
        )
    if verify:
        for name, tensor in state.items():
            rec = inventory[name]
            if list(tensor.shape) != rec["shape"]:
                raise ContractError(f"shape mismatch for {name}")
            if str(tensor.dtype).replace("torch.", "") != rec["dtype"]:
                raise ContractError(f"dtype mismatch for {name}")
            if sha256_bytes(tensor_bytes(tensor)) != rec["sha256"]:
                raise ContractError(f"tensor checksum mismatch for {name}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or set(missing) != set(TIED_ALIASES):
        raise ContractError(
            f"unexpected state_dict delta: missing={missing} unexpected={unexpected}"
        )

    chosen = device or default_device()
    model = model.to(chosen).eval()
    info = BundleInfo(
        bundle_dir=d,
        config=config,
        manifest=manifest,
        step=int(manifest.get("step", -1)),
        vocab_size=int(model_doc["vocab_size"]),
        max_seq_len=int(model_doc["max_seq_len"]),
        dtype=config["dtype"],
        device=chosen,
        manifest_file_hash=sha256_file(d / MANIFEST_FILE),
        config_hash=sha256_bytes(canonical_json(config).encode("utf-8")),
    )
    return model, info


def validate_bundle(bundle_dir: str | Path, checkpoint_path: str | Path) -> dict[str, Any]:
    """Compare a bundle bitwise against its source M1 checkpoint."""
    d = Path(bundle_dir)
    _config, _tokenizer, manifest = validate_bundle_files(d)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint_state(ckpt)
    loaded = load_file(str(d / WEIGHTS_FILE))

    for name, original in state.items():
        key = TIED_ALIASES.get(name, name)
        if key not in loaded:
            raise ContractError(f"bundle is missing {key} for checkpoint entry {name}")
        if not torch.equal(loaded[key], original.detach().to("cpu")):
            raise ContractError(f"bundle differs from checkpoint at {name}")
    return {
        "status": "OK",
        "format_version": manifest.get("format_version"),
        "step": manifest.get("step"),
        "tensors_compared": len(state),
    }
