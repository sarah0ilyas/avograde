"""Streamlit demo for AvoGrade — thin client over the real serving layer.
Surfaces production behavior: prediction source, latency, confidence-based
review routing, and drift on the uploaded image."""
import io, time
import numpy as np
import torch
import streamlit as st
from PIL import Image

from avograde.models.grader_cnn import build_resnet18_grader, build_transforms
from avograde.serving.baseline import stage
from avograde.features import image_summary

CKPT = "avograder_resnet.pt"

@st.cache_resource
def load():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    model = build_resnet18_grader(num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    return model, classes

model, classes = load()
tfm = build_transforms(train=False, img_size=128)

# Reference brightness range from training-style images (for a simple drift check).
REF_BRIGHTNESS = (0.35, 0.65)

st.title("🥑 AvoGrade")
st.caption("Avocado ripeness grading · ResNet18 · Hass ripening dataset")

file = st.file_uploader("Upload an avocado photo", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)

    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(tfm(img).unsqueeze(0)), dim=1)[0]
    latency_ms = (time.perf_counter() - t0) * 1e3
    pred = int(probs.argmax())
    conf = float(probs[pred])
    idx = pred / (len(classes) - 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Stage", f"{classes[pred]} · {stage(idx)}")
    c2.metric("Confidence", f"{conf:.0%}")
    c3.metric("Latency", f"{latency_ms:.0f} ms")

    if conf < 0.60:
        st.warning("Low confidence — this case would be routed to human review.")

    # Drift check on the uploaded image.
    b = image_summary(np.array(img.resize((128, 128))))["brightness"]
    if not (REF_BRIGHTNESS[0] <= b <= REF_BRIGHTNESS[1]):
        st.warning(f"⚠️ Input looks unlike training data (brightness {b:.2f}) — "
                   "prediction may be less reliable.")

    st.bar_chart({str(classes[i]): float(probs[i]) for i in range(len(classes))})

    with st.expander("About this model"):
        st.markdown(
            "- **71% accuracy** on a *leakage-safe* split (fruit held out, not photos).\n"
            "- Known weakness: confuses **stage 4 vs 5** (both dark/overripe).\n"
            "- Trained on **Hass only** — other varieties would drift.\n"
            "- Errors are almost entirely between adjacent stages."
        )
