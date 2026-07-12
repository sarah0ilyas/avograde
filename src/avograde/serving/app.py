"""FastAPI wrapper. Image lookup lives in domain closures so the generic
PredictionService stays key -> float. Lazy imports keep tests dependency-free.
"""
from __future__ import annotations

import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from avograde.features import image_key, image_summary
from avograde.serving.baseline import color_ripeness_index, stage
from avograde.serving.service import PredictionService

_images: dict[str, np.ndarray] = {}   # request-scoped image store, keyed by hash


import torch
from avograde.models.grader_cnn import build_resnet18_grader, build_transforms

_CKPT = "avograder_resnet.pt"
_ckpt = torch.load(_CKPT, map_location="cpu", weights_only=False)
_CLASSES = _ckpt["classes"]
_model = build_resnet18_grader(num_classes=len(_CLASSES), pretrained=False)
_model.load_state_dict(_ckpt["state_dict"])
_model.eval()
_tfm = build_transforms(train=False, img_size=128)


def _model_predict(key: str) -> float:
    """Run the trained ResNet on the stored image; return a ripeness index in [0,1]."""
    from PIL import Image
    img = Image.fromarray(_images[key])
    x = _tfm(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_model(x), dim=1)[0]
    stage_idx = int(probs.argmax())
    # map stage (0..n-1) to a 0..1 ripeness index
    return stage_idx / (len(_CLASSES) - 1)


def build_service() -> PredictionService:
    return PredictionService(
        model_fn=_model_predict,
        baseline_fn=lambda k: color_ripeness_index(_images[k]),
    )


def build_app(service: PredictionService | None = None):
    service = service or build_service()
    app = FastAPI(title="AvoGrade")

    @app.post("/grade")
    async def grade(image: UploadFile = File(...)):
        raw = await image.read()
        img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        key = image_key(raw)
        _images[key] = img
        p = service.predict(key)
        return {"stage": stage(p.value), "ripeness_index": round(p.value, 3),
                "source": p.source, "latency_ms": round(p.latency_ms, 1),
                "model_version": service._model_version, "reason": p.reason}

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "cached_images": len(service._cache)}

    return app
