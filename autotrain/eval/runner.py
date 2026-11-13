"""M1.5 EvalRunner: bundle -> comparable, hash-sealed report.

The runner is read-only with respect to the frozen corpora and to the bundle.
It never imports runner/checkpoint/data from the training side, so the
measurement cannot be influenced by training state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..export.loader import load_bundle
from .corpus import read_tokens
from .evalspec import (
    BASELINE_GAIN_RATIO,
    CORPUS_DIR,
    EVAL_BATCH,
    EVAL_WINDOWS,
    MARKOV_P,
    MIN_IMPROVEMENT_BITS,
    MIN_STRUCTURE_GAIN_RATIO,
    RANDOM_CONTROL_FILE,
    RANDOM_CONTROL_TOLERANCE_NATS,
    REPORT_VERSION,
    STRUCTURED_FILE,
    VOCAB_SIZE,
    EvalSpecError,
    corpus_manifest_hash,
    evalspec_hash,
    verify_corpus,
)
from .metrics import (
    greedy_transition_accuracy,
    markov_optimal_nats,
    measure_nats,
    nats_to_bits,
    safe_perplexity,
    structure_gain_ratio,
    uniform_baseline_nats,
)

# Two reports may only be compared if the measurement instrument, the model
# structure AND the numerical backend are identical. `device` is included
# because CPU / MPS / CUDA do not produce bitwise-identical logits: training on
# CPU while evaluating on MPS would otherwise look like a real improvement.
COMPAT_KEYS = (
    "format_version",
    "evalspec_hash",
    "corpus_manifest_hash",
    "bundle_config_hash",
    "vocab_size",
    "seq_len",
    "device",
)


class EvalRunner:
    def __init__(
        self,
        bundle_dir: str | Path,
        eval_dir: str | Path | None = None,
        device: str | None = None,
    ):
        self.eval_dir = Path(eval_dir) if eval_dir is not None else CORPUS_DIR
        self.corpus_manifest = verify_corpus(self.eval_dir)
        self.corpus_hash = corpus_manifest_hash(self.corpus_manifest)
        self.model, self.info = load_bundle(bundle_dir, device=device)
        if self.info.vocab_size != VOCAB_SIZE:
            raise EvalSpecError(
                f"bundle vocab_size {self.info.vocab_size} != frozen spec {VOCAB_SIZE}"
            )
        need = EVAL_WINDOWS * EVAL_BATCH * self.info.max_seq_len + 1
        if need > int(self.corpus_manifest["files"][STRUCTURED_FILE]["tokens"]):
            raise EvalSpecError(
                f"frozen corpus too short for seq_len={self.info.max_seq_len}"
            )

    def run(self) -> dict[str, Any]:
        seq_len = self.info.max_seq_len
        vocab = self.info.vocab_size
        device = self.info.device
        structured = read_tokens(self.eval_dir / STRUCTURED_FILE)
        control = read_tokens(self.eval_dir / RANDOM_CONTROL_FILE)

        baseline = uniform_baseline_nats(vocab)
        optimal = markov_optimal_nats(vocab, MARKOV_P)
        s_nats = measure_nats(
            self.model, structured, seq_len, EVAL_BATCH, EVAL_WINDOWS, device
        )
        c_nats = measure_nats(
            self.model, control, seq_len, EVAL_BATCH, EVAL_WINDOWS, device
        )
        gain = structure_gain_ratio(s_nats, baseline, optimal)
        deviation = c_nats - baseline
        transition = greedy_transition_accuracy(self.model, structured, vocab)

        checks = {
            "control_consistent": abs(deviation) <= RANDOM_CONTROL_TOLERANCE_NATS,
            "structure_captured": gain >= MIN_STRUCTURE_GAIN_RATIO,
        }
        if not checks["control_consistent"]:
            status = "INVALID"
        elif checks["structure_captured"]:
            status = "STRUCTURE_CAPTURED"
        elif gain <= BASELINE_GAIN_RATIO:
            status = "BASELINE"
        else:
            status = "PARTIAL"

        return {
            "format_version": REPORT_VERSION,
            "evalspec_hash": evalspec_hash(self.corpus_hash),
            "corpus_manifest_hash": self.corpus_hash,
            "bundle_config_hash": self.info.config_hash,
            "bundle_step": self.info.step,
            "vocab_size": vocab,
            "seq_len": seq_len,
            "device": device,
            "uniform_baseline_nats": baseline,
            "markov_optimal_nats": optimal,
            "metrics": {
                "structured": {
                    "nats_per_token": s_nats,
                    "bits_per_token": nats_to_bits(s_nats),
                    "perplexity": safe_perplexity(s_nats),
                    "structure_gain_ratio": gain,
                    "greedy_transition_accuracy": transition,
                },
                "random_control": {
                    "nats_per_token": c_nats,
                    "bits_per_token": nats_to_bits(c_nats),
                    "deviation_from_uniform_nats": deviation,
                },
            },
            "checks": checks,
            "status": status,
        }


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """A/B verdict. Reports with different frozen specs are INCOMPARABLE."""
    differences = [
        key for key in COMPAT_KEYS if before.get(key) != after.get(key)
    ]
    result: dict[str, Any] = {
        "format_version": "wanxiang-evalcompare-1",
        "compatible": not differences,
        "differences": differences,
        "before_step": before.get("bundle_step"),
        "after_step": after.get("bundle_step"),
    }
    if differences:
        result["verdict"] = "INCOMPARABLE"
        return result

    b = before["metrics"]["structured"]
    a = after["metrics"]["structured"]
    delta_bits = b["bits_per_token"] - a["bits_per_token"]
    delta_gain = a["structure_gain_ratio"] - b["structure_gain_ratio"]
    result.update(
        {
            "delta_bits_per_token": delta_bits,
            "delta_structure_gain_ratio": delta_gain,
            "delta_greedy_transition_accuracy": (
                a["greedy_transition_accuracy"] - b["greedy_transition_accuracy"]
            ),
            "verdict": (
                "IMPROVED"
                if delta_bits >= MIN_IMPROVEMENT_BITS
                else "REGRESSED"
                if delta_bits <= -MIN_IMPROVEMENT_BITS
                else "NO_CHANGE"
            ),
        }
    )
    return result
