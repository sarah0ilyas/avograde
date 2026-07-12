"""Color-based ripeness baseline — the fallback when the model is unavailable.

For HASS: darker, less-green skin -> riper. Deliberately WRONG for green-skin
varieties, which is exactly why variety drift is monitored.
"""
from __future__ import annotations

import numpy as np

from avograde.config import STAGES
from avograde.features import image_summary


def color_ripeness_index(img: np.ndarray) -> float:
    s = image_summary(img)
    idx = 0.5 + 1.2 * (0.55 - s["brightness"]) - 1.0 * (s["green_ratio"] - 0.34)
    return float(np.clip(idx, 0.0, 1.0))


def stage(index: float) -> str:
    if index < 0.30:
        return STAGES[0]   # underripe
    if index < 0.50:
        return STAGES[1]   # breaking
    if index < 0.75:
        return STAGES[2]   # ripe
    return STAGES[3]       # overripe
