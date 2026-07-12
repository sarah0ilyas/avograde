import time
import pytest
from avograde.serving.service import PredictionService, _CacheEntry


def _svc(model_fn):
    return PredictionService(model_fn=model_fn, baseline_fn=lambda k: 1.23,
                             latency_budget_ms=1e6)


def _boom(_):
    raise ConnectionError("model down")


def test_cache_hit():
    svc = _svc(lambda k: 9.9)
    svc.prime_cache("a", 5.0)
    assert svc.predict("a").source == "cache"


def test_model_then_cached():
    svc = _svc(lambda k: 7.0)
    assert svc.predict("b").source == "model"
    assert svc.predict("b").source == "cache"


def test_fallback_to_baseline_when_model_down():
    svc = _svc(_boom)
    p = svc.predict("c")
    assert p.source == "baseline" and p.value == pytest.approx(1.23)


def test_stale_cache_preferred_over_baseline():
    svc = _svc(_boom)
    # Old model_version -> _cache_get() misses, but the entry still exists,
    # so a downed model should serve it stale rather than fall to baseline.
    svc._cache["d"] = _CacheEntry(4.0, time.time(), "OLD-version")
    p = svc.predict("d")
    assert p.source == "cache" and p.stale is True and p.value == 4.0
