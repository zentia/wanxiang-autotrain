"""M1.5 exporter: M1 checkpoint -> self-describing model bundle.

The export is all-or-nothing: everything is written into a sibling temp
directory, verified (file checksums + bitwise tensor round-trip), and only
then atomically renamed into place. A failed export never leaves a partial
bundle that a downstream evaluator could mistake for a valid one.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from ..config import ExperimentConfig, load_config
from .contract import (
    CHECKSUMMED_FILES,
    CONFIG_FILE,
    FORMAT_VERSION,
    MANIFEST_FILE,
    SUPPORTED_DTYPES,
    TIED_ALIASES,
    TOKENIZER_FILE,
    WEIGHTS_FILE,
    ContractError,
    build_config,
    build_tokenizer,
    canonical_json,
    checkpoint_state,
    sha256_bytes,
    sha256_file,
    tensor_bytes,
    validate_bundle_files,
    write_json,
)


def _export_manifest(
    tmp: Path,
    checkpoint_path: Path,
    cfg: ExperimentConfig,
    ckpt: Any,
    dtype: str,
    tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    inventory = [
        {
            "name": name,
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "sha256": sha256_bytes(tensor_bytes(t)),
        }
        for name, t in tensors.items()
    ]
    return {
        "format_version": FORMAT_VERSION,
        "source_checkpoint": checkpoint_path.name,
        "source_config_hash": cfg.config_hash,
        "step": int(getattr(ckpt, "step", -1)),
        "source_metrics": dict(getattr(ckpt, "metrics", {}) or {}),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "dtype": dtype,
        "tied_aliases": dict(TIED_ALIASES),
        "tensors": inventory,
        "files": {name: sha256_file(tmp / name) for name in CHECKSUMMED_FILES},
    }


def _verify_roundtrip(tmp: Path, state: dict[str, torch.Tensor]) -> None:
    """Every exported tensor must be bitwise identical to the checkpoint."""
    loaded = load_file(str(tmp / WEIGHTS_FILE))
    expected_names = set(state) - set(TIED_ALIASES)
    if set(loaded) != expected_names:
        raise ContractError(
            f"exported tensor names differ: missing={sorted(expected_names - set(loaded))} "
            f"extra={sorted(set(loaded) - expected_names)}"
        )
    for name, original in state.items():
        key = TIED_ALIASES.get(name, name)
        if not torch.equal(loaded[key], original.detach().to("cpu")):
            raise ContractError(f"bitwise round-trip failed for {name}")


def export_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Export an M1 ``.pt`` checkpoint into an M1.5 bundle.

    Returns the MANIFEST dict. Raises ContractError on any violation; no
    partial bundle is published on failure.
    """
    cfg = load_config(config_path)
    if isinstance(checkpoint_path, (str, Path)):
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise ContractError(f"checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        # In-memory Checkpoint, used by export_initial for the untrained baseline.
        ckpt_path = Path("<untrained>")
        ckpt = checkpoint_path
    state = checkpoint_state(ckpt)
    if not isinstance(state, dict) or not state:
        raise ContractError("checkpoint model_state is empty or not a mapping")

    for alias, target in TIED_ALIASES.items():
        if alias in state and target in state:
            if not torch.equal(
                state[alias].detach().to("cpu"), state[target].detach().to("cpu")
            ):
                raise ContractError(
                    f"tied weight {alias} does not equal {target}; refusing to export"
                )

    tensors = {
        name: tensor.detach().to("cpu").contiguous()
        for name, tensor in sorted(state.items())
        if name not in TIED_ALIASES
    }
    if not tensors:
        raise ContractError("no exportable tensors")
    dtype = str(next(iter(tensors.values())).dtype).replace("torch.", "")
    if dtype not in SUPPORTED_DTYPES:
        raise ContractError(f"unsupported weight dtype: {dtype}")

    out = Path(out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".m15tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    try:
        write_json(tmp / CONFIG_FILE, build_config(cfg.model, cfg.data.seq_len, dtype))
        write_json(tmp / TOKENIZER_FILE, build_tokenizer(cfg.model.vocab_size))
        save_file(tensors, str(tmp / WEIGHTS_FILE))
        manifest = _export_manifest(tmp, ckpt_path, cfg, ckpt, dtype, tensors)
        write_json(tmp / MANIFEST_FILE, manifest)

        _verify_roundtrip(tmp, state)
        validate_bundle_files(tmp)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    if out.exists():
        if not force:
            shutil.rmtree(tmp, ignore_errors=True)
            raise ContractError(
                f"bundle already exists: {out} (pass force=True to replace)"
            )
        shutil.rmtree(out)
    tmp.replace(out)
    return manifest


def export_initial(
    config_path: str | Path,
    out_dir: str | Path,
    *,
    seed: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Export an untrained, deterministically initialized baseline bundle.

    M1 only writes checkpoints during/after training, so there is no "before"
    artifact to compare against. This produces step-0 weights with the same
    contract, making the training A/B measurable from the very first run.
    """
    from ..trainer.checkpoint import Checkpoint
    from ..trainer.model import WanxiangModel

    cfg = load_config(config_path)
    # Seeding for a reproducible baseline must not disturb the caller's RNG:
    # exporting a "before" artifact should not change the "after" run.
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(int(cfg.seed if seed is None else seed))
        model = WanxiangModel(cfg.model, max_seq_len=cfg.data.seq_len)
        ckpt = Checkpoint(
            step=0,
            config_hash=cfg.config_hash,
            model_state=model.state_dict(),
            optimizer_state={},
            scheduler_state={"step_num": 0},
            data_cursor=0,
            rng_states={},
            metrics={},
        )
    finally:
        torch.set_rng_state(rng_state)
    return export_checkpoint(ckpt, config_path, out_dir, force=force)


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Stable digest of the semantic part of a MANIFEST (no timestamp/path)."""
    semantic = {
        k: manifest.get(k)
        for k in ("format_version", "step", "dtype", "tensors", "tied_aliases")
    }
    return sha256_bytes(canonical_json(semantic).encode("utf-8"))
