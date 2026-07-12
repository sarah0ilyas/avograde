"""Evaluate the trained grader on the fruit-held-out validation split:
overall accuracy, a majority-class baseline, per-class precision/recall,
and a confusion matrix."""
from __future__ import annotations

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from avograde.data.labels import load_label_table, split_by_fruit
from avograde.data.dataset import AvocadoDataset
from avograde.models.grader_cnn import GraderCNN, build_resnet18_grader, build_transforms


def main(excel, images, model_path="avograder.pt", img_size=128, batch_size=32, arch="cnn"):
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")
    classes = ckpt["classes"]
    C = len(classes)

    table = load_label_table(excel, images)
    split = split_by_fruit(table)                 # same seed -> same val set as training
    val = split.val
    print(f"val photos: {len(val)}  fruit: {val['sample_id'].nunique()}  classes: {classes}")

    ds = AvocadoDataset(val, build_transforms(False, img_size))
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0)

    if arch == "resnet18":
        model = build_resnet18_grader(num_classes=C, pretrained=False)
    else:
        model = GraderCNN(num_classes=C)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    conf = np.zeros((C, C), dtype=int)            # conf[true, pred]
    with torch.no_grad():
        for xb, yb in dl:
            preds = model(xb).argmax(1)
            for t, p in zip(yb.tolist(), preds.tolist()):
                conf[t, p] += 1

    acc = np.trace(conf) / conf.sum()
    maj_idx = int(split.train["label_idx"].value_counts().idxmax())
    maj_acc = (val["label_idx"] == maj_idx).mean()

    print(f"\noverall accuracy : {acc:.3f}")
    print(f"majority baseline: {maj_acc:.3f}  (always predict stage {classes[maj_idx]})")

    print("\nper-class  (stage : precision / recall / support)")
    for c in range(C):
        support = conf[c, :].sum()
        col = conf[:, c].sum()
        prec = conf[c, c] / col if col else 0.0
        rec = conf[c, c] / support if support else 0.0
        print(f"  stage {classes[c]} : {prec:.2f} / {rec:.2f} / {support}")

    print("\nconfusion matrix  (rows = true, cols = predicted)")
    print("         " + "  ".join(f"{classes[c]:>4}" for c in range(C)))
    for c in range(C):
        row = "  ".join(f"{conf[c, j]:>4}" for j in range(C))
        print(f"true {classes[c]} |  {row}")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--excel", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--model", default="avograder.pt")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--arch", default="cnn", choices=["cnn", "resnet18"])
    a = p.parse_args()
    main(a.excel, a.images, a.model, a.img_size, a.batch_size, a.arch)


if __name__ == "__main__":
    cli()
