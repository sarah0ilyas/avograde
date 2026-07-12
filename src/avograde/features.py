"""Shared image feature extraction — used by the baseline AND drift monitor."""
from __future__ import annotations

import hashlib
import numpy as np


def image_key(image_bytes: bytes) -> str:
    """Content hash: the same photo grades once."""
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def image_summary(img: np.ndarray) -> dict[str, float]:
    """img: HxWx3 uint8 -> cheap scalar features."""
    x = img.astype(float) / 255.0
    r, g, b = x[..., 0].mean(), x[..., 1].mean(), x[..., 2].mean()
    return {
        "brightness": float(x.mean()),
        "green_ratio": float(g / (r + g + b + 1e-8)),
        "dark_frac": float((x.mean(axis=2) < 0.20).mean()),
    }
