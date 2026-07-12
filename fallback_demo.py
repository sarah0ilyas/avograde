"""Demonstrate the fallback ladder: when the model is unavailable, the service
still answers via the colour-based baseline instead of failing."""
import io, numpy as np
from PIL import Image

from avograde.serving.service import PredictionService
from avograde.serving.baseline import color_ripeness_index, stage
from avograde.features import image_key
from avograde.serving import app as appmod

path = "data/Hass Avocado Ripening Photographic Dataset/Avocado Ripening Dataset/T20_d01_001_a_1.jpg"
raw = open(path, "rb").read()
img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
key = image_key(raw)
appmod._images[key] = img

def broken_model(k):
    raise ConnectionError("model server unreachable")   # simulate model down

service = PredictionService(
    model_fn=broken_model,
    baseline_fn=lambda k: color_ripeness_index(appmod._images[k]),
)

p = service.predict(key)
print("source :", p.source)      # expect: baseline
print("stage  :", stage(p.value))
print("reason :", p.reason)
