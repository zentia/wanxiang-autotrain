"""ExperimentStore CRUD and metric series."""

from __future__ import annotations

from autotrain.store.experiment_store import ExperimentStore


class TestExperiments:
    def test_create_and_get(self, tmp_path):
        store = ExperimentStore(tmp_path / "exp.db")
        eid = store.create_experiment("wanxiang-4m", "abc123", {"experiment": "wanxiang-4m"})
        exp = store.get_experiment(eid)
        assert exp is not None
        assert exp["name"] == "wanxiang-4m"
        assert exp["config_hash"] == "abc123"
        assert exp["status"] == "RUNNING"
        assert exp["config"]["experiment"] == "wanxiang-4m"
        store.close()

    def test_get_missing_returns_none(self, tmp_path):
        store = ExperimentStore(tmp_path / "exp.db")
        assert store.get_experiment(999) is None
        store.close()

    def test_set_status(self, tmp_path):
        store = ExperimentStore(tmp_path / "exp.db")
        eid = store.create_experiment("e", "h", {})
        store.set_status(eid, "DONE")
        assert store.get_experiment(eid)["status"] == "DONE"
        store.close()

    def test_find_by_name_and_status(self, tmp_path):
        store = ExperimentStore(tmp_path / "exp.db")
        e1 = store.create_experiment("a", "h1", {})
        e2 = store.create_experiment("a", "h2", {})
        store.create_experiment("b", "h3", {})
        store.set_status(e2, "FAILED")
        assert [x["id"] for x in store.find_experiments(name="a")] == [e1, e2]
        assert [x["id"] for x in store.find_experiments(name="a", status="FAILED")] == [e2]
        store.close()


class TestMetrics:
    def test_log_and_read_series(self, tmp_path):
        store = ExperimentStore(tmp_path / "exp.db")
        eid = store.create_experiment("e", "h", {})
        store.log_metric(eid, 1, "loss", 5.0)
        store.log_metric(eid, 2, "loss", 4.5)
        store.log_metric(eid, 2, "lr", 1e-3)
        series = store.get_metric_series(eid, "loss")
        assert series == [(1, 5.0), (2, 4.5)]
        assert store.get_metric_series(eid, "lr") == [(2, 1e-3)]
        store.close()

    def test_replace_same_step_key(self, tmp_path):
        store = ExperimentStore(tmp_path / "exp.db")
        eid = store.create_experiment("e", "h", {})
        store.log_metric(eid, 1, "loss", 5.0)
        store.log_metric(eid, 1, "loss", 4.0)
        assert store.get_metric_series(eid, "loss") == [(1, 4.0)]
        store.close()

    def test_log_metrics_batch(self, tmp_path):
        store = ExperimentStore(tmp_path / "exp.db")
        eid = store.create_experiment("e", "h", {})
        store.log_metrics(eid, 10, {"loss": 3.0, "val_loss": 3.5, "lr": 1e-4})
        assert len(store.get_metric_series(eid, "loss")) == 1
        assert store.get_metric_series(eid, "val_loss") == [(10, 3.5)]
        store.close()

    def test_series_isolated_per_experiment(self, tmp_path):
        store = ExperimentStore(tmp_path / "exp.db")
        e1 = store.create_experiment("e1", "h", {})
        e2 = store.create_experiment("e2", "h", {})
        store.log_metric(e1, 1, "loss", 1.0)
        assert store.get_metric_series(e2, "loss") == []
        store.close()
