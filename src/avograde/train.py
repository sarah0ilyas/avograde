"""Training entry point: Mendeley Hass images -> trained GraderCNN.

Usage:
    python -m avograde.train --excel labels.xlsx --images ./photos --epochs 10
"""
from __future__ import annotations

import argparse


def main(excel: str, images: str, epochs: int = 10, batch_size: int = 32,
         img_size: int = 128, out: str = "avograder.pt", arch: str = "cnn") -> None:
    import torch
    from torch.utils.data import DataLoader

    from avograde.data.labels import load_label_table, split_by_fruit
    from avograde.data.dataset import AvocadoDataset
    from avograde.models.grader_cnn import GraderCNN, build_resnet18_grader, build_transforms, train as fit

    table = load_label_table(excel, images)
    classes = table.attrs["classes"]
    print(f"{len(table)} photos, {table['sample_id'].nunique()} fruit, classes={classes}")

    split = split_by_fruit(table)
    train_ds = AvocadoDataset(split.train, build_transforms(True, img_size))
    val_ds = AvocadoDataset(split.val, build_transforms(False, img_size))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if arch == "resnet18":
        model = build_resnet18_grader(num_classes=len(classes), pretrained=True)
    else:
        model = GraderCNN(num_classes=len(classes))
    print(f"architecture: {arch}")
    best = fit(model, train_dl, val_dl, epochs=epochs, device=device)
    torch.save({"state_dict": model.state_dict(), "classes": classes}, out)
    print(f"best val_acc={best:.3f}  saved -> {out}")


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--excel", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--out", default="avograder.pt")
    p.add_argument("--arch", default="cnn", choices=["cnn", "resnet18"])
    a = p.parse_args()
    main(a.excel, a.images, a.epochs, a.batch_size, a.img_size, a.out, a.arch)


if __name__ == "__main__":
    cli()
