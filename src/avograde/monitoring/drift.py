"""Drift monitoring — PSI/KS on image features vs a training reference."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from avograde.config import PSI_WATCH, PSI_ALERT


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-6, None)
    cur_frac = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-6, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    ref = np.sort(np.asarray(reference, dtype=float))
    cur = np.sort(np.asarray(current, dtype=float))
    grid = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, grid, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, grid, side="right") / len(cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def psi_status(value: float) -> str:
    if value < PSI_WATCH:
        return "stable"
    if value < PSI_ALERT:
        return "watch"
    return "alert"


@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    ks: float
    status: str


class DriftMonitor:
    def __init__(self, reference: dict[str, np.ndarray]):
        self._reference = {k: np.asarray(v, dtype=float) for k, v in reference.items()}

    def check(self, current: dict[str, np.ndarray]) -> list[FeatureDriftResult]:
        results = []
        for feat, ref in self._reference.items():
            if feat not in current:
                continue
            p, k = psi(ref, current[feat]), ks_statistic(ref, current[feat])
            results.append(FeatureDriftResult(feat, p, k, psi_status(p)))
        return results

    @staticmethod
    def any_alert(results: list[FeatureDriftResult]) -> bool:
        return any(r.status == "alert" for r in results)
