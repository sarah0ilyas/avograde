"""End-to-end demo: serving + fallback + drift on synthetic avocado images."""
from __future__ import annotations

import numpy as np

from avograde.features import image_key, image_summary
from avograde.serving.baseline import color_ripeness_index, stage
from avograde.serving.service import PredictionService
from avograde.monitoring.drift import DriftMonitor
from avograde.config import DRIFT_FEATURES


def fake_avocado(variety: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    means = [62, 56, 72] if variety == "hass" else [112, 150, 92]  # fuerte = green
    chans = [np.clip(rng.normal(m, 18, (64, 64)), 0, 255) for m in means]
    return np.stack(chans, axis=2).astype(np.uint8)


def main() -> None:
    hass = fake_avocado("hass", seed=1)
    key = image_key(hass.tobytes())
    svc = PredictionService(
        model_fn=lambda k: 0.62,
        baseline_fn=lambda k: color_ripeness_index(hass),
    )

    print("== serving ==")
    svc.prime_cache(key, 0.64)
    p = svc.predict(key); print("  cache   ->", p.value, stage(p.value), f"[{p.source}]")
    p = svc.predict("new"); print("  model   ->", p.value, stage(p.value), f"[{p.source}]")
    down = PredictionService(lambda k: (_ for _ in ()).throw(ConnectionError("down")),
                             lambda k: color_ripeness_index(hass))
    p = down.predict("x"); print("  fallback->", round(p.value, 3), stage(p.value), f"[{p.source}]")

    print("\n== drift ==")
    ref = [fake_avocado("hass", s) for s in range(400)]
    monitor = DriftMonitor({k: np.array([image_summary(im)[k] for im in ref])
                            for k in DRIFT_FEATURES})
    for title, variety, rng in [("same variety", "hass", range(1000, 1300)),
                                ("VARIETY SHIFT (Fuerte)", "fuerte", range(2000, 2300))]:
        imgs = [fake_avocado(variety, s) for s in rng]
        res = monitor.check({k: np.array([image_summary(im)[k] for im in imgs])
                             for k in DRIFT_FEATURES})
        print(f"  {title:24s} alert={DriftMonitor.any_alert(res)}")


if __name__ == "__main__":
    main()
