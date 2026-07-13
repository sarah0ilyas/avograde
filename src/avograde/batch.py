"""Batch scoring job — pre-scores images into the cache so the API serves
cache-hits instead of cold model calls. In production this runs on a schedule.
"""
from __future__ import annotations

import argparse
import glob
import time

import numpy as np
from PIL import Image

from avograde.serving.app import build_service, _images
from avograde.features import image_key


def run(image_glob: str, limit: int | None = None) -> None:
    service = build_service()
    paths = sorted(glob.glob(image_glob))
    if limit:
        paths = paths[:limit]

    t0 = time.perf_counter()
    scored = 0
    for p in paths:
        with open(p, "rb") as f:
            raw = f.read()
        key = image_key(raw)
        _images[key] = np.array(Image.open(p).convert("RGB"))
        service.predict(key)          # runs model, fills cache
        scored += 1

    dt = time.perf_counter() - t0
    print(f"batch scored {scored} images in {dt:.1f}s "
          f"({scored/dt:.1f}/s) · cache now holds {len(service._cache)}")


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="glob, e.g. 'data/.../*.jpg'")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.images, a.limit)


if __name__ == "__main__":
    cli()
