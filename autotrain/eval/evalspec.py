"""Frozen M1.5 evaluation specification.

Everything in this module is a constant. Changing any threshold, window
count, seed or corpus changes ``evalspec_hash``, which makes reports produced
before and after the change INCOMPARABLE. That is the mechanism that stops a
training loop (or a self-improving agent) from quietly moving the goalposts.

The evaluator never writes into ``evals/``. It only reads it and verifies it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EVALSPEC_VERSION = "wanxiang-eval-1"
CORPUS_FORMAT_VERSION = "wanxiang-evalcorpus-1"
REPORT_VERSION = "wanxiang-evalreport-1"

VOCAB_SIZE = 8192
MARKOV_P = 0.8

STRUCTURED_FILE = "structured_eval.bin"
RANDOM_CONTROL_FILE = "random_control.bin"
EXPECTED_CORPUS_FILES = (STRUCTURED_FILE, RANDOM_CONTROL_FILE)
CORPUS_MANIFEST_FILE = "manifest.json"

CORPUS_TOKENS = 200_000
SEED_STRUCTURED = 20261113
SEED_RANDOM = 20261114
SEED_TRAIN_CORPUS = 20261115

# Measurement shape
EVAL_WINDOWS = 8
EVAL_BATCH = 4
GEN_TOKENS = 64
GEN_PROMPT_LEN = 32

# Frozen decision thresholds
MIN_STRUCTURE_GAIN_RATIO = 0.10
BASELINE_GAIN_RATIO = 0.02
RANDOM_CONTROL_TOLERANCE_NATS = 0.30
MIN_IMPROVEMENT_BITS = 0.05

CORPUS_DIR = Path(__file__).resolve().parents[2] / "evals"


class EvalSpecError(RuntimeError):
    """Raised when the frozen evaluation set or spec is violated."""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def corpus_manifest_hash(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()


def evalspec_hash(corpus_hash: str) -> str:
    """Digest of every frozen decision relevant to comparing two reports."""
    spec = {
        "evalspec_version": EVALSPEC_VERSION,
        "corpus_format_version": CORPUS_FORMAT_VERSION,
        "vocab_size": VOCAB_SIZE,
        "markov_p": MARKOV_P,
        "eval_windows": EVAL_WINDOWS,
        "eval_batch": EVAL_BATCH,
        "gen_tokens": GEN_TOKENS,
        "gen_prompt_len": GEN_PROMPT_LEN,
        "min_structure_gain_ratio": MIN_STRUCTURE_GAIN_RATIO,
        "baseline_gain_ratio": BASELINE_GAIN_RATIO,
        "random_control_tolerance_nats": RANDOM_CONTROL_TOLERANCE_NATS,
        "min_improvement_bits": MIN_IMPROVEMENT_BITS,
        "corpus_manifest_hash": corpus_hash,
    }
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


def verify_corpus(eval_dir: str | Path | None = None) -> dict[str, Any]:
    """Verify the frozen corpora byte-for-byte; return the manifest.

    The file set is fixed. Adding, removing, replacing or editing any file
    raises EvalSpecError, so an automated loop cannot improve its own score
    by touching the measurement instrument.
    """
    from .corpus import sha256_file

    d = Path(eval_dir) if eval_dir is not None else CORPUS_DIR
    manifest_path = d / CORPUS_MANIFEST_FILE
    if not manifest_path.is_file():
        raise EvalSpecError(
            f"corpus manifest not found: {manifest_path}. evals/ is a "
            "repository-level frozen asset: use 'pip install -e .' or run "
            "from the repository root (it is not shipped inside the wheel)."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("format_version") != CORPUS_FORMAT_VERSION:
        raise EvalSpecError("corpus manifest format_version mismatch")
    if manifest.get("frozen") is not True:
        raise EvalSpecError("corpus manifest is not marked frozen")
    if manifest.get("vocab_size") != VOCAB_SIZE:
        raise EvalSpecError("corpus vocab_size does not match evalspec")

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(EXPECTED_CORPUS_FILES):
        raise EvalSpecError("corpus file set differs from the frozen spec")

    # The manifest alone is not enough: an extra .bin dropped into evals/ must
    # also be rejected, otherwise a loop could select a more favourable corpus
    # without ever touching the manifest.
    present = {p.name for p in d.glob("*.bin")}
    if present != set(EXPECTED_CORPUS_FILES):
        raise EvalSpecError(
            "frozen corpus directory file set differs from the frozen spec: "
            f"missing={sorted(set(EXPECTED_CORPUS_FILES) - present)} "
            f"extra={sorted(present - set(EXPECTED_CORPUS_FILES))}"
        )

    for name in EXPECTED_CORPUS_FILES:
        path = d / name
        if not path.is_file():
            raise EvalSpecError(f"corpus file missing: {name}")
        rec = files[name]
        if sha256_file(path) != rec.get("sha256"):
            raise EvalSpecError(f"corpus checksum mismatch: {name}")
        expected_tokens = int(rec.get("tokens", 0))
        if path.stat().st_size != expected_tokens * 2:
            raise EvalSpecError(f"corpus token count mismatch: {name}")
        if int(rec.get("vocab_size", VOCAB_SIZE)) != VOCAB_SIZE:
            raise EvalSpecError(f"corpus vocab_size mismatch in manifest: {name}")
    return manifest
