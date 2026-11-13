"""M1.5 evaluator: frozen spec, objective metrics, comparable A/B.

All tests are offline and deterministic. No LLM-as-a-judge, no network, and
nothing is ever written into the repository's ``evals/`` directory -- the
tamper tests copy the frozen corpora into ``tmp_path`` first.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from autotrain.eval.corpus import (
    make_markov_tokens,
    make_uniform_tokens,
    read_tokens,
    sha256_file,
)
from autotrain.eval.evalspec import (
    CORPUS_DIR,
    EXPECTED_CORPUS_FILES,
    MARKOV_P,
    RANDOM_CONTROL_FILE,
    SEED_RANDOM,
    SEED_STRUCTURED,
    STRUCTURED_FILE,
    VOCAB_SIZE,
    EvalSpecError,
    corpus_manifest_hash,
    evalspec_hash,
    verify_corpus,
)
from autotrain.eval.metrics import (
    markov_optimal_nats,
    nats_to_bits,
    structure_gain_ratio,
    uniform_baseline_nats,
)
from autotrain.eval.runner import COMPAT_KEYS, EvalRunner, compare_reports
from autotrain.export.exporter import export_initial
from conftest import make_raw_config, write_config

CORPUS_TOKENS = 200_000


def _copy_frozen(d: Path) -> Path:
    """Copy the frozen corpora into a scratch dir so tamper tests stay local."""
    d.mkdir(parents=True, exist_ok=True)
    for name in (*EXPECTED_CORPUS_FILES, "manifest.json"):
        shutil.copyfile(CORPUS_DIR / name, d / name)
    return d


def _rewrite_manifest(d: Path, change) -> None:
    p = d / "manifest.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    change(doc)
    p.write_text(json.dumps(doc), encoding="utf-8")


def _digest_dir(d: Path) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(d.iterdir())}


def _report(bits: float = 13.0, gain: float = 0.0, **overrides) -> dict:
    report = {
        "format_version": "wanxiang-evalreport-1",
        "evalspec_hash": "a" * 64,
        "corpus_manifest_hash": "b" * 64,
        "bundle_config_hash": "c" * 64,
        "vocab_size": VOCAB_SIZE,
        "seq_len": 256,
        "device": "cpu",
        "bundle_step": 0,
        "metrics": {
            "structured": {
                "bits_per_token": bits,
                "structure_gain_ratio": gain,
                "greedy_transition_accuracy": 0.0,
            }
        },
    }
    report.update(overrides)
    return report


class TestFrozenCorpus:
    def test_repository_corpus_verifies(self):
        manifest = verify_corpus()
        assert manifest["frozen"] is True
        assert set(manifest["files"]) == set(EXPECTED_CORPUS_FILES)
        for name in EXPECTED_CORPUS_FILES:
            assert sha256_file(CORPUS_DIR / name) == manifest["files"][name]["sha256"]
            assert (CORPUS_DIR / name).stat().st_size == CORPUS_TOKENS * 2

    def test_frozen_bytes_match_the_generator(self):
        """A frozen corpus must be reproducible from the frozen spec."""
        structured = make_markov_tokens(
            CORPUS_TOKENS, VOCAB_SIZE, MARKOV_P, SEED_STRUCTURED
        )
        control = make_uniform_tokens(CORPUS_TOKENS, VOCAB_SIZE, SEED_RANDOM)
        np.testing.assert_array_equal(
            read_tokens(CORPUS_DIR / STRUCTURED_FILE), structured
        )
        np.testing.assert_array_equal(
            read_tokens(CORPUS_DIR / RANDOM_CONTROL_FILE), control
        )

    def test_structured_corpus_has_structure_but_uniform_unigrams(self):
        """Beating ln(V) here means learning transitions, not frequencies."""
        tokens = make_markov_tokens(
            CORPUS_TOKENS, VOCAB_SIZE, MARKOV_P, SEED_STRUCTURED
        )
        prev = tokens[:-1].astype(np.int64)
        hits = np.mean(tokens[1:] == (prev + 1) % VOCAB_SIZE)
        assert hits == pytest.approx(MARKOV_P, abs=0.01)
        counts = np.bincount(tokens.astype(np.int64), minlength=VOCAB_SIZE)
        assert counts.min() > 0
        assert np.mean(counts) == pytest.approx(CORPUS_TOKENS / VOCAB_SIZE, rel=1e-9)

    def test_tampered_corpus_file_rejected(self, tmp_path):
        d = _copy_frozen(tmp_path / "evals")
        with open(d / STRUCTURED_FILE, "r+b") as f:
            first = f.read(1)
            f.seek(0)
            f.write(bytes([first[0] ^ 0xFF]))
        with pytest.raises(EvalSpecError, match="checksum mismatch"):
            verify_corpus(d)

    def test_extra_bin_rejected(self, tmp_path):
        d = _copy_frozen(tmp_path / "evals")
        (d / "secret_eval.bin").write_bytes(b"\x00" * 4)
        with pytest.raises(EvalSpecError, match="file set"):
            verify_corpus(d)

    def test_missing_bin_rejected(self, tmp_path):
        d = _copy_frozen(tmp_path / "evals")
        (d / RANDOM_CONTROL_FILE).unlink()
        with pytest.raises(EvalSpecError, match="file set|missing"):
            verify_corpus(d)

    def test_unfrozen_manifest_rejected(self, tmp_path):
        d = _copy_frozen(tmp_path / "evals")
        _rewrite_manifest(d, lambda doc: doc.update(frozen=False))
        with pytest.raises(EvalSpecError, match="not marked frozen"):
            verify_corpus(d)

    def test_manifest_hash_is_content_addressed(self):
        manifest = verify_corpus()
        h = corpus_manifest_hash(manifest)
        assert h == corpus_manifest_hash(verify_corpus())
        assert len(h) == 64
        changed = dict(manifest)
        changed["vocab_size"] = manifest["vocab_size"] + 1
        assert corpus_manifest_hash(changed) != h


class TestObjectiveMetrics:
    def test_uniform_baseline_is_ln_vocab(self):
        baseline = uniform_baseline_nats(VOCAB_SIZE)
        assert baseline == pytest.approx(math.log(8192))
        assert baseline == pytest.approx(9.010913, abs=1e-5)
        assert nats_to_bits(baseline) == pytest.approx(baseline / math.log(2.0))

    def test_markov_optimal_closed_form_and_boundaries(self):
        p_hit = MARKOV_P + (1 - MARKOV_P) / VOCAB_SIZE
        p_other = (1 - MARKOV_P) / VOCAB_SIZE
        expected = -(
            p_hit * math.log(p_hit)
            + (VOCAB_SIZE - 1) * p_other * math.log(p_other)
        )
        assert markov_optimal_nats(VOCAB_SIZE, MARKOV_P) == pytest.approx(expected)
        assert 0 < expected < uniform_baseline_nats(VOCAB_SIZE)
        # p=0 degenerates to i.i.d. uniform, p=1 to a deterministic chain.
        assert markov_optimal_nats(VOCAB_SIZE, 0.0) == pytest.approx(
            math.log(VOCAB_SIZE)
        )
        assert markov_optimal_nats(VOCAB_SIZE, 1.0) == pytest.approx(0.0)

    def test_structure_gain_ratio_is_normalized(self):
        baseline = uniform_baseline_nats(VOCAB_SIZE)
        optimal = markov_optimal_nats(VOCAB_SIZE, MARKOV_P)
        assert structure_gain_ratio(baseline, baseline, optimal) == pytest.approx(0.0)
        assert structure_gain_ratio(optimal, baseline, optimal) == pytest.approx(1.0)
        mid = baseline - 0.5 * (baseline - optimal)
        assert structure_gain_ratio(mid, baseline, optimal) == pytest.approx(0.5)


class TestEvalSpecSeal:
    def test_hash_is_deterministic_and_spec_sensitive(self):
        h = evalspec_hash("b" * 64)
        assert h == evalspec_hash("b" * 64)
        assert h != evalspec_hash("c" * 64)
        assert len(h) == 64

    def test_compare_verdicts(self):
        assert compare_reports(_report(13.0), _report(12.5))["verdict"] == "IMPROVED"
        assert compare_reports(_report(12.5), _report(13.0))["verdict"] == "REGRESSED"
        assert compare_reports(_report(13.0), _report(12.99))["verdict"] == "NO_CHANGE"
        result = compare_reports(_report(13.0), _report(12.5))
        assert result["compatible"] is True
        assert result["delta_bits_per_token"] == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "key,value",
        [
            ("format_version", "other"),
            ("evalspec_hash", "d" * 64),
            ("corpus_manifest_hash", "d" * 64),
            ("bundle_config_hash", "d" * 64),
            ("vocab_size", 4096),
            ("seq_len", 128),
            ("device", "mps"),
        ],
    )
    def test_incompatible_reports_are_not_scored(self, key, value):
        result = compare_reports(_report(13.0), _report(12.0, **{key: value}))
        assert result["verdict"] == "INCOMPARABLE"
        assert result["compatible"] is False
        assert result["differences"] == [key]
        assert "delta_bits_per_token" not in result


class TestEvalRunner:
    @staticmethod
    def _tiny_bundle(tmp_path: Path, vocab_size: int = VOCAB_SIZE) -> tuple[Path, Path]:
        raw = make_raw_config()
        raw["model"] = {"layers": 1, "dim": 64, "heads": 4, "vocab_size": vocab_size}
        raw["data"] = {"mix": "mix_test", "seq_len": 64, "val_fraction": 0.05}
        cfg_path = write_config(tmp_path, raw)
        bundle = tmp_path / "bundle"
        export_initial(cfg_path, bundle, seed=123)
        return Path(cfg_path), bundle

    def test_runner_is_read_only_and_emits_sealed_report(self, tmp_path):
        d = _copy_frozen(tmp_path / "evals")
        _cfg_path, bundle = self._tiny_bundle(tmp_path)
        before_corpus = _digest_dir(d)
        before_bundle = _digest_dir(bundle)

        report = EvalRunner(bundle, eval_dir=d, device="cpu").run()

        assert _digest_dir(d) == before_corpus
        assert _digest_dir(bundle) == before_bundle
        for key in COMPAT_KEYS:
            assert key in report
        assert report["device"] == "cpu"
        assert report["evalspec_hash"] == evalspec_hash(
            corpus_manifest_hash(verify_corpus(d))
        )
        assert report["uniform_baseline_nats"] == pytest.approx(math.log(VOCAB_SIZE))
        assert 0 < report["markov_optimal_nats"] < report["uniform_baseline_nats"]
        assert (
            0.0
            <= report["metrics"]["structured"]["greedy_transition_accuracy"]
            <= 1.0
        )
        # Negative control: an untrained model must not appear to "learn" noise.
        assert report["checks"]["control_consistent"] is True

    def test_runner_rejects_wrong_vocab(self, tmp_path):
        d = _copy_frozen(tmp_path / "evals")
        _cfg_path, bundle = self._tiny_bundle(tmp_path / "wrong", vocab_size=512)
        with pytest.raises(EvalSpecError, match="vocab_size"):
            EvalRunner(bundle, eval_dir=d, device="cpu")
