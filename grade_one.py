"""Grade a single avocado photo with the trained model. Proof the model works
on one real image before we wrap it in an API."""
import sys
import torch
from PIL import Image

from avograde.models.grader_cnn import GraderCNN, build_resnet18_grader, build_transforms

CKPT = "avograder_resnet.pt"
ARCH = "resnet18"

# Load the model ONCE.
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
classes = ckpt["classes"]
model = build_resnet18_grader(num_classes=len(classes), pretrained=False) if ARCH == "resnet18" \
        else GraderCNN(num_classes=len(classes))
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Load and prep the image passed on the command line.
path = sys.argv[1]
img = Image.open(path).convert("RGB")
x = build_transforms(train=False, img_size=128)(img).unsqueeze(0)  # add batch dim

# Predict.
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]
pred = int(probs.argmax())

print(f"\nimage: {path}")
print(f"predicted ripeness stage: {classes[pred]}  (confidence {probs[pred]:.2f})")
print("all stages:", {classes[i]: round(float(probs[i]), 2) for i in range(len(classes))})
