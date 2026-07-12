"""Generic cache-first prediction service with a fallback ladder.

Domain-agnostic: `model_fn` and `baseline_fn` are `key -> float` callables the
caller supplies. The avocado wiring lives in serving/app.py and demo.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from avograde import config


@dataclass
class Prediction:
    key: str
    value: float
    source: str          # "model" | "cache" | "baseline" | "error"
    latency_ms: float
    stale: bool = False
    reason: Optional[str] = None


@dataclass
class _CacheEntry:
    value: float
    stored_at: float
    model_version: str


class PredictionService:
    def __init__(
        self,
        model_fn: Callable[[str], float],
        baseline_fn: Callable[[str], float],
        model_version: str = config.MODEL_VERSION,
        latency_budget_ms: float = config.LATENCY_BUDGET_MS,
        cache_ttl_s: float = config.CACHE_TTL_S,
    ):
        self._model_fn = model_fn
        self._baseline_fn = baseline_fn
        self._model_version = model_version
        self._budget_ms = latency_budget_ms
        self._ttl_s = cache_ttl_s
        self._cache: dict[str, _CacheEntry] = {}

    def prime_cache(self, key: str, value: float) -> None:
        """Called by the nightly batch scorer to precompute predictions."""
        self._cache[key] = _CacheEntry(value, time.time(), self._model_version)

    def _cache_get(self, key: str) -> Optional[_CacheEntry]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        fresh = (time.time() - entry.stored_at) <= self._ttl_s
        return entry if (fresh and entry.model_version == self._model_version) else None

    def predict(self, key: str, force_refresh: bool = False) -> Prediction:
        t0 = time.perf_counter()

        if not force_refresh:
            entry = self._cache_get(key)
            if entry is not None:
                return Prediction(key, entry.value, "cache", (time.perf_counter() - t0) * 1e3)

        try:
            value = self._model_fn(key)
            elapsed = (time.perf_counter() - t0) * 1e3
            if elapsed > self._budget_ms:
                return Prediction(key, value, "model", elapsed,
                                  reason=f"latency budget exceeded ({elapsed:.0f}ms)")
            self.prime_cache(key, value)
            return Prediction(key, value, "model", elapsed)
        except Exception as model_err:
            stale = self._cache.get(key)
            if stale is not None:
                return Prediction(key, stale.value, "cache", (time.perf_counter() - t0) * 1e3,
                                  stale=True, reason=f"model unavailable: {model_err}")
            try:
                value = self._baseline_fn(key)
                return Prediction(key, value, "baseline", (time.perf_counter() - t0) * 1e3,
                                  reason=f"model+cache unavailable: {model_err}")
            except Exception as base_err:
                return Prediction(key, float("nan"), "error", (time.perf_counter() - t0) * 1e3,
                                  reason=f"all paths failed: {base_err}")
