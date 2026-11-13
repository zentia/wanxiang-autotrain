"""M1.5 model bundle contract, version ``wanxiang-m15-1``.

A bundle is a self-describing directory. Any consumer can load the trained
weights without importing the training loop::

    <bundle>/
      config.json        architecture, structure, tokenizer metadata, dtype
      model.safetensors  state_dict weights, minus tied aliases
      tokenizer.json     minimal token-id specification
      MANIFEST.json      provenance, step, tensor inventory, file checksums

Read-only boundaries:

* the training code never writes into an exported bundle;
* the evaluator never writes into ``evals/``;
* MANIFEST checksums are verified on every load, so a bundle cannot be
  silently edited after export.

This module is the contract. It intentionally has no torch dependency so it
can be used by tooling that only needs to inspect or validate a bundle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FORMAT_VERSION = "wanxiang-m15-1"
ARCHITECTURE = "WanxiangModel"

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
TOKENIZER_FILE = "tokenizer.json"
MANIFEST_FILE = "MANIFEST.json"

# Files whose checksums are recorded in MANIFEST.json. MANIFEST.json itself is
# excluded to avoid a self-referential hash.
CHECKSUMMED_FILES = (CONFIG_FILE, WEIGHTS_FILE, TOKENIZER_FILE)
BUNDLE_FILES = (CONFIG_FILE, WEIGHTS_FILE, TOKENIZER_FILE, MANIFEST_FILE)

# WanxiangModel ties lm_head.weight to tok_emb.weight. safetensors refuses
# shared storage, so the alias is omitted on export and restored on load.
TIED_ALIASES: dict[str, str] = {"lm_head.weight": "tok_emb.weight"}

MODEL_KEYS = ("layers", "dim", "heads", "vocab_size", "max_seq_len")
STRUCTURE_KEYS = (
    "position",
    "norm",
    "activation",
    "attention",
    "tie_word_embeddings",
    "bias",
)
TOKENIZER_TYPE = "uint16_flat"
SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")


class ContractError(ValueError):
    """Raised when a bundle violates the M1.5 contract."""


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding, used for hashing and on-disk docs."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_bytes(tensor: Any) -> bytes:
    """Raw bytes of a torch tensor, dtype-agnostic (works for bfloat16)."""
    import torch

    flat = tensor.detach().to("cpu").reshape(-1).contiguous()
    return flat.view(torch.uint8).numpy().tobytes()


def checkpoint_state(obj: Any) -> Any:
    """Extract ``model_state`` from an M1 Checkpoint (or a plain dict)."""
    if hasattr(obj, "model_state"):
        return obj.model_state
    if isinstance(obj, dict) and "model_state" in obj:
        return obj["model_state"]
    raise ContractError("checkpoint has no model_state")


def build_config(model_cfg: Any, max_seq_len: int, dtype: str) -> dict[str, Any]:
    """Build the self-describing config from a ModelConfig-like object."""
    vocab = int(model_cfg.vocab_size)
    return {
        "format_version": FORMAT_VERSION,
        "architecture": ARCHITECTURE,
        "model": {
            "layers": int(model_cfg.layers),
            "dim": int(model_cfg.dim),
            "heads": int(model_cfg.heads),
            "vocab_size": vocab,
            "max_seq_len": int(max_seq_len),
        },
        "structure": {
            "position": "learned",
            "norm": "rmsnorm",
            "activation": "swiglu",
            "attention": "sdpa_causal",
            "tie_word_embeddings": True,
            "bias": False,
            "tied_aliases": dict(TIED_ALIASES),
        },
        "tokenizer": build_tokenizer(vocab),
        "dtype": dtype,
    }


def build_tokenizer(vocab_size: int) -> dict[str, Any]:
    """Minimal token specification.

    M1 consumes a flat uint16 token stream and has no tokenizer yet. M2 will
    replace this document; until then the contract records exactly what a
    token id means so a bundle is never ambiguous.
    """
    return {
        "type": TOKENIZER_TYPE,
        "vocab_size": int(vocab_size),
        "token_id_range": [0, int(vocab_size) - 1],
        "encoding": "numpy.uint16, little-endian, flat token stream",
        "special_tokens": {},
        "note": "M1 uint16 token stream; M2 data pipeline will replace this.",
    }


def validate_config(doc: Any) -> dict[str, Any]:
    """Validate a config.json document; return it on success."""
    if not isinstance(doc, dict):
        raise ContractError("config.json root must be an object")
    if doc.get("format_version") != FORMAT_VERSION:
        raise ContractError(
            f"unsupported format_version: {doc.get('format_version')!r} "
            f"(expected {FORMAT_VERSION!r})"
        )
    if doc.get("architecture") != ARCHITECTURE:
        raise ContractError(f"unsupported architecture: {doc.get('architecture')!r}")

    model = doc.get("model")
    if not isinstance(model, dict):
        raise ContractError("config.model must be an object")
    for key in MODEL_KEYS:
        if key not in model or not isinstance(model[key], int):
            raise ContractError(f"config.model.{key} must be an int")
    if model["layers"] < 1 or model["dim"] < 32 or model["heads"] < 1:
        raise ContractError("invalid config.model dimensions")
    if model["dim"] % model["heads"] != 0:
        raise ContractError("config.model.dim must be divisible by heads")
    if model["vocab_size"] > 65536:
        raise ContractError("uint16 token stream cannot address > 65536 tokens")
    if model["max_seq_len"] < 32:
        raise ContractError("config.model.max_seq_len must be >= 32")

    structure = doc.get("structure")
    if not isinstance(structure, dict):
        raise ContractError("config.structure must be an object")
    for key in STRUCTURE_KEYS:
        if key not in structure:
            raise ContractError(f"config.structure.{key} missing")
    if structure["tie_word_embeddings"] is not True:
        raise ContractError("M1.5 requires tied word embeddings")
    if structure["position"] != "learned":
        raise ContractError("M1.5 only supports learned positional embeddings")
    if structure["tied_aliases"] != dict(TIED_ALIASES):
        raise ContractError("config.structure.tied_aliases does not match contract")

    tokenizer = doc.get("tokenizer")
    if not isinstance(tokenizer, dict) or tokenizer.get("type") != TOKENIZER_TYPE:
        raise ContractError(f"config.tokenizer.type must be {TOKENIZER_TYPE!r}")
    if tokenizer.get("vocab_size") != model["vocab_size"]:
        raise ContractError("tokenizer.vocab_size != model.vocab_size")

    if doc.get("dtype") not in SUPPORTED_DTYPES:
        raise ContractError(f"config.dtype must be one of {SUPPORTED_DTYPES}")
    return doc


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(canonical_json(obj))
        f.write("\n")


def validate_bundle_files(bundle_dir: str | Path) -> tuple[dict, dict, dict]:
    """Verify a bundle's presence, schema and file checksums.

    Returns ``(config, tokenizer, manifest)``. Raises ContractError on any
    violation. This is the read-side half of the export contract.
    """
    d = Path(bundle_dir)
    if not d.is_dir():
        raise ContractError(f"bundle directory not found: {d}")
    for name in BUNDLE_FILES:
        if not (d / name).is_file():
            raise ContractError(f"bundle missing {name}")

    manifest = read_json(d / MANIFEST_FILE)
    if not isinstance(manifest, dict):
        raise ContractError("MANIFEST.json root must be an object")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ContractError("MANIFEST.json format_version mismatch")

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(CHECKSUMMED_FILES):
        raise ContractError("MANIFEST.files must list exactly the checksummed files")
    for name, expected in files.items():
        actual = sha256_file(d / name)
        if actual != expected:
            raise ContractError(
                f"checksum mismatch for {name}: manifest={expected} actual={actual}"
            )

    config = validate_config(read_json(d / CONFIG_FILE))
    tokenizer = read_json(d / TOKENIZER_FILE)
    if tokenizer != config["tokenizer"]:
        raise ContractError("tokenizer.json does not match config.tokenizer")
    return config, tokenizer, manifest
