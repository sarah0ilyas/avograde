"""Demonstrate the drift monitor on real avocado photos:
quiet on a normal batch, alerting on a shifted (dimmed) batch."""
import glob, random
import numpy as np
from PIL import Image

from avograde.features import image_summary
from avograde.monitoring.drift import DriftMonitor
from avograde.config import DRIFT_FEATURES

IMG_DIR = "data/Hass Avocado Ripening Photographic Dataset/Avocado Ripening Dataset"
paths = glob.glob(IMG_DIR + "/*.jpg")
random.seed(0)
random.shuffle(paths)

def load(p, dim=1.0):
    img = np.array(Image.open(p).convert("RGB").resize((128, 128)))
    return np.clip(img * dim, 0, 255).astype(np.uint8)   # dim<1 darkens the image

def feats(batch):
    return {k: np.array([image_summary(im)[k] for im in batch]) for k in DRIFT_FEATURES}

# reference = 400 normal training-style photos
ref = [load(p) for p in paths[:400]]
monitor = DriftMonitor(feats(ref))

def report(title, batch):
    res = monitor.check(feats(batch))
    print(f"\n{title}")
    for r in res:
        print(f"  {r.feature:11s} PSI={r.psi:6.3f} -> {r.status}")
    print("  ALERT:", DriftMonitor.any_alert(res))

# normal batch -> should stay quiet
report("normal Hass batch (expect stable)", [load(p) for p in paths[400:700]])

# shifted batch: dimmed to 55% brightness, simulating a new/worse camera
report("shifted batch — dimmer lighting (expect ALERT)",
       [load(p, dim=0.55) for p in paths[400:700]])
