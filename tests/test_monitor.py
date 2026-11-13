"""Loss spike detection and early-stop tracking."""

from __future__ import annotations

from autotrain.config import RecoverConfig
from autotrain.trainer.monitor import EarlyStopTracker, LossSpikeDetector


class TestLossSpikeDetector:
    def _cfg(self, **kw) -> RecoverConfig:
        base = {"max_retries": 3, "loss_spike_sigma": 3.0, "spike_window": 50, "early_stop_patience": 5}
        base.update(kw)
        return RecoverConfig(**base)

    def test_no_spike_on_stable_loss(self):
        det = LossSpikeDetector(self._cfg())
        spikes = [det.update(3.0 + 0.01 * (i % 5)) for i in range(100)]
        assert not any(spikes)

    def test_spike_detected(self):
        det = LossSpikeDetector(self._cfg())
        for i in range(60):
            det.update(3.0 + 0.01 * (i % 5))
        assert det.update(30.0) is True

    def test_no_spike_during_warmup_history(self):
        det = LossSpikeDetector(self._cfg())
        # fewer than min_history points: never flags
        assert det.update(100.0) is False
        assert det.update(100.0) is False

    def test_reset_clears_history(self):
        det = LossSpikeDetector(self._cfg())
        for i in range(60):
            det.update(3.0)
        det.reset()
        assert det.update(100.0) is False  # history gone, no spike call

    def test_sigma_sensitivity(self):
        # higher sigma -> less sensitive; use data with realistic variance
        det_loose = LossSpikeDetector(self._cfg(loss_spike_sigma=10.0))
        for i in range(60):
            det_loose.update(2.0 + 3.0 * ((i % 7) / 6))  # uniform-ish 2.0..5.0
        assert det_loose.update(6.0) is False
        # but an extreme spike is still caught even at sigma=10
        assert det_loose.update(40.0) is True


class TestEarlyStopTracker:
    def test_improving_never_stops(self):
        t = EarlyStopTracker(patience=3)
        losses = [5.0, 4.5, 4.2, 4.0, 3.9, 3.8]
        assert not any(t.update(v) for v in losses)

    def test_stalled_loss_stops_after_patience(self):
        t = EarlyStopTracker(patience=3)
        t.update(4.0)
        assert t.update(4.0) is False
        assert t.update(4.0) is False
        assert t.update(4.0) is True
        assert t.converged_early

    def test_improvement_resets_counter(self):
        t = EarlyStopTracker(patience=3)
        t.update(4.0)
        t.update(4.0)
        t.update(3.9)  # improvement resets
        t.update(3.9)
        assert t.update(3.9) is False
        assert t.update(3.9) is True

    def test_best_tracks_minimum(self):
        t = EarlyStopTracker(patience=2)
        t.update(4.0)
        t.update(3.0)
        assert t.best == 3.0
