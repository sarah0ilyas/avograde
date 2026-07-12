import numpy as np
from avograde.monitoring.drift import DriftMonitor


def _ref():
    rng = np.random.default_rng(0)
    return {"f": rng.normal(0.7, 0.06, 4000)}


def test_matched_does_not_alert():
    rng = np.random.default_rng(1)
    m = DriftMonitor(_ref())
    assert not DriftMonitor.any_alert(m.check({"f": rng.normal(0.7, 0.06, 2000)}))


def test_shift_alerts():
    rng = np.random.default_rng(2)
    m = DriftMonitor(_ref())
    assert DriftMonitor.any_alert(m.check({"f": rng.normal(0.4, 0.09, 2000)}))
