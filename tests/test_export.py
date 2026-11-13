"""M1.5 export contract: schema, bitwise round-trip, tamper detection."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file

from autotrain.config import load_config
from autotrain.export.contract import (
    CONFIG_FILE,
    MANIFEST_FILE,
    TIED_ALIASES,
    WEIGHTS_FILE,
    ContractError,
    sha256_file,
    validate_bundle_files,
)
from autotrain.export.exporter import export_checkpoint
from autotrain.export.loader import load_bundle, validate_bundle
from autotrain.trainer.checkpoint import Checkpoint
from autotrain.trainer.model import WanxiangModel
from conftest import make_raw_config, write_config


def _small_model_cfg():
    from autotrain.config import ModelConfig

    return ModelConfig(layers=1, dim=64, heads=4, vocab_size=8192)


def _make_inputs(tmp_path):
    raw = make_raw_config()
    raw["model"] = {"layers": 1, "dim": 64, "heads": 4, "vocab_size": 8192}
    raw["data"] = {"mix": "mix_test", "seq_len": 64, "val_fraction": 0.05}
    cfg_path = write_config(tmp_path, raw)
    cfg = load_config(cfg_path)
    model = WanxiangModel(cfg.model, max_seq_len=cfg.data.seq_len)
    ckpt = Checkpoint(
        step=7,
        config_hash=cfg.config_hash,
        model_state=model.state_dict(),
        optimizer_state={},
        scheduler_state={"step_num": 7},
        data_cursor=0,
        rng_states={},
        metrics={"loss": 8.9},
    )
    ckpt_path = tmp_path / "ckpt_step_7.pt"
    torch.save(ckpt, ckpt_path)
    return cfg_path, ckpt_path, model


class TestExportRoundTrip:
    def test_bundle_files_and_manifest(self, tmp_path):
        cfg_path, ckpt_path, _ = _make_inputs(tmp_path)
        out = tmp_path / "bundle"
        manifest = export_checkpoint(ckpt_path, cfg_path, out)

        for name in (CONFIG_FILE, WEIGHTS_FILE, "tokenizer.json", MANIFEST_FILE):
            assert (out / name).is_file()
        assert manifest["step"] == 7
        assert manifest["source_metrics"] == {"loss": 8.9}
        assert manifest["dtype"] == "float32"
        for name, digest in manifest["files"].items():
            assert sha256_file(out / name) == digest
        validate_bundle_files(out)

    def test_bitwise_against_source_checkpoint(self, tmp_path):
        cfg_path, ckpt_path, model = _make_inputs(tmp_path)
        out = tmp_path / "bundle"
        export_checkpoint(ckpt_path, cfg_path, out)

        assert validate_bundle(out, ckpt_path)["status"] == "OK"
        state = load_file(str(out / WEIGHTS_FILE))
        assert set(TIED_ALIASES).isdisjoint(set(state))
        for name, expected in model.state_dict().items():
            key = TIED_ALIASES.get(name, name)
            assert torch.equal(state[key], expected), name

    def test_loader_restores_tied_embedding(self, tmp_path):
        cfg_path, ckpt_path, model = _make_inputs(tmp_path)
        out = tmp_path / "bundle"
        export_checkpoint(ckpt_path, cfg_path, out)
        loaded, info = load_bundle(out, device="cpu")

        assert info.step == 7
        assert info.vocab_size == 8192
        assert info.max_seq_len == 64
        assert torch.equal(loaded.lm_head.weight, loaded.tok_emb.weight)
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2)

    def test_manifest_tensor_inventory(self, tmp_path):
        cfg_path, ckpt_path, _ = _make_inputs(tmp_path)
        out = tmp_path / "bundle"
        manifest = export_checkpoint(ckpt_path, cfg_path, out)
        inventory = {r["name"]: r for r in manifest["tensors"]}
        state = load_file(str(out / WEIGHTS_FILE))
        assert set(inventory) == set(state)
        for name, tensor in state.items():
            assert inventory[name]["shape"] == list(tensor.shape)
            assert inventory[name]["dtype"] == str(tensor.dtype).replace("torch.", "")


class TestContractEnforcement:
    def test_tampered_config_rejected(self, tmp_path):
        cfg_path, ckpt_path, _ = _make_inputs(tmp_path)
        out = tmp_path / "bundle"
        export_checkpoint(ckpt_path, cfg_path, out)
        doc = json.loads((out / CONFIG_FILE).read_text(encoding="utf-8"))
        doc["model"]["layers"] = 99
        (out / CONFIG_FILE).write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ContractError, match="checksum mismatch"):
            validate_bundle_files(out)

    def test_tampered_tensor_rejected(self, tmp_path):
        cfg_path, ckpt_path, _ = _make_inputs(tmp_path)
        out = tmp_path / "bundle"
        export_checkpoint(ckpt_path, cfg_path, out)
        state = load_file(str(out / WEIGHTS_FILE))
        key = "tok_emb.weight"
        state[key] = state[key] + 1
        from safetensors.torch import save_file

        save_file(state, str(out / WEIGHTS_FILE))
        with pytest.raises(ContractError, match="checksum mismatch"):
            validate_bundle_files(out)

    def test_missing_file_rejected(self, tmp_path):
        cfg_path, ckpt_path, _ = _make_inputs(tmp_path)
        out = tmp_path / "bundle"
        export_checkpoint(ckpt_path, cfg_path, out)
        (out / WEIGHTS_FILE).unlink()
        with pytest.raises(ContractError, match="missing"):
            validate_bundle_files(out)

    def test_overwrite_requires_force(self, tmp_path):
        cfg_path, ckpt_path, _ = _make_inputs(tmp_path)
        out = tmp_path / "bundle"
        export_checkpoint(ckpt_path, cfg_path, out)
        with pytest.raises(ContractError, match="already exists"):
            export_checkpoint(ckpt_path, cfg_path, out)
        first = sha256_file(out / CONFIG_FILE)
        export_checkpoint(ckpt_path, cfg_path, out, force=True)
        assert sha256_file(out / CONFIG_FILE) == first
        assert not (tmp_path / "bundle.m15tmp").exists()

    def test_config_schema_rejects_non_tied(self, tmp_path):
        from autotrain.export.contract import validate_config

        cfg_path, _, _ = _make_inputs(tmp_path)
        cfg = load_config(cfg_path)
        from autotrain.export.contract import build_config

        doc = build_config(cfg.model, cfg.data.seq_len, "float32")
        doc["structure"]["tie_word_embeddings"] = False
        with pytest.raises(ContractError, match="tied"):
            validate_config(doc)


class TestInitialBaseline:
    """M1 writes no step-0 checkpoint, so the A/B 'before' must be synthesized."""

    def test_export_initial_is_deterministic(self, tmp_path):
        from autotrain.export.exporter import export_initial

        cfg_path, _, _ = _make_inputs(tmp_path)
        a, b = tmp_path / "a", tmp_path / "b"
        ma = export_initial(cfg_path, a, seed=123)
        mb = export_initial(cfg_path, b, seed=123)

        assert ma["step"] == 0 and mb["step"] == 0
        assert ma["source_checkpoint"] == "<untrained>"
        assert ma["files"] == mb["files"]
        validate_bundle_files(a)
        loaded, info = load_bundle(a, device="cpu")
        assert info.step == 0
        assert torch.equal(loaded.lm_head.weight, loaded.tok_emb.weight)

    def test_export_initial_differs_from_trained_bundle(self, tmp_path):
        from autotrain.export.exporter import export_initial

        cfg_path, ckpt_path, _ = _make_inputs(tmp_path)
        initial = export_initial(cfg_path, tmp_path / "initial", seed=123)
        trained = export_checkpoint(ckpt_path, cfg_path, tmp_path / "trained")
        assert initial["files"][WEIGHTS_FILE] != trained["files"][WEIGHTS_FILE]

    def test_export_initial_preserves_caller_rng(self, tmp_path):
        from autotrain.export.exporter import export_initial

        cfg_path, _, _ = _make_inputs(tmp_path)
        torch.manual_seed(999)
        before = torch.get_rng_state()
        export_initial(cfg_path, tmp_path / "initial", seed=123)
        assert torch.equal(before, torch.get_rng_state())


class TestAtomicPublish:
    def test_write_failure_leaves_no_partial_bundle(self, tmp_path, monkeypatch):
        import autotrain.export.exporter as exporter_mod

        cfg_path, ckpt_path, _ = _make_inputs(tmp_path)

        def boom(*_args, **_kwargs):
            raise OSError("simulated disk failure")

        monkeypatch.setattr(exporter_mod, "save_file", boom)
        out = tmp_path / "bundle"
        with pytest.raises(OSError, match="simulated disk failure"):
            export_checkpoint(ckpt_path, cfg_path, out)
        assert not out.exists()
        assert not (tmp_path / "bundle.m15tmp").exists()
