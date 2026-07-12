"""
AvoGrade — the PyTorch model at the core of the avocado ripeness grading service.

Classes (ripeness stages): underripe -> breaking -> ripe -> overripe, plus a
reject flag for bruising/rot. The served ripeness index derives from the class
probabilities; the top-class probability doubles as the confidence signal for
the fallback / human-review decision.

Two model options:
  - GraderCNN            : compact from-scratch conv net; runs offline (default).
  - build_resnet18_grader: transfer learning from a pretrained ResNet18 (the
                           senior choice on limited data). Needs a one-time
                           weight download in your environment.

Run `python grader_cnn.py` to build the model and run a forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

STAGES = ["underripe", "breaking", "ripe", "overripe"]  # + reject flag (5th class)
CLASS_NAMES = [*STAGES, "reject"]


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class GraderCNN(nn.Module):
    """Compact CNN for avocado ripeness classification (5 classes by default)."""

    def __init__(self, num_classes: int = 5, in_ch: int = 3, widths=(32, 64, 128)):
        super().__init__()
        chs = [in_ch, *widths]
        self.features = nn.Sequential(
            *[_conv_block(chs[i], chs[i + 1]) for i in range(len(widths))]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(widths[-1], num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)

    @torch.no_grad()
    def ripeness_index(self, x: torch.Tensor) -> torch.Tensor:
        """Expected stage position in [0,1] over ripeness classes (ignores reject).

        Weighted average of the ripeness-stage probabilities, so a fruit that is
        mostly 'ripe' with a little 'overripe' scores just past the ready band.
        """
        proba = self.predict_proba(x)[:, : len(STAGES)]
        proba = proba / proba.sum(dim=1, keepdim=True).clamp_min(1e-8)
        positions = torch.linspace(0, 1, len(STAGES))
        return (proba * positions).sum(dim=1)


def build_resnet18_grader(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    from torchvision import models
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_transforms(train: bool, img_size: int = 128):
    from torchvision import transforms
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(), norm,
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), norm,
    ])


def train(model, train_loader, val_loader, epochs=10, lr=3e-4, device="cpu"):
    """Standard training loop skeleton — champion metric is val accuracy."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += yb.numel()
        acc = correct / max(total, 1)
        best_acc = max(best_acc, acc)
        print(f"epoch {epoch:2d}  val_acc={acc:.3f}")
    return best_acc


if __name__ == "__main__":
    torch.manual_seed(0)
    model = GraderCNN(num_classes=5)
    n_params = sum(p.numel() for p in model.parameters())

    x = torch.randn(4, 3, 128, 128)          # a batch of 4 fake avocado images
    logits = model(x)
    proba = model.predict_proba(x)
    idx = model.ripeness_index(x)

    print("classes:", CLASS_NAMES)
    print(f"GraderCNN parameters: {n_params:,}")
    print("logits shape:", tuple(logits.shape))          # (4, 5)
    print("proba row sums:", proba.sum(1).round(decimals=3).tolist())
    print("ripeness index:", idx.round(decimals=3).tolist())
    assert logits.shape == (4, 5)
    assert torch.allclose(proba.sum(1), torch.ones(4), atol=1e-5)
    print("\nForward pass OK — model is ready to train on real avocado images.")
