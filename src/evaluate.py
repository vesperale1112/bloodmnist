"""Evaluate a saved BloodMNIST checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .constants import CLASS_NAMES, NUM_CLASSES
from .dataset import BloodMNISTDataset, load_split
from .metrics import classification_metrics, per_class_dataframe
from .models import build_model
from .utils import describe_device, ensure_dir, resolve_device, save_json
from .visualize import plot_confusion_matrix, save_misclassified_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a BloodMNIST checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint, usually outputs/runs/.../checkpoints/best.pt.")
    parser.add_argument("--data", default="bloodmnist.npz", help="Path to bloodmnist.npz.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None, help="Defaults to checkpoint parent run directory/eval_<split>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    meta = checkpoint.get("meta", {})
    mean = meta.get("mean")
    std = meta.get("std")
    model_name = checkpoint.get("model_name", checkpoint.get("args", {}).get("model", "improved_cnn"))

    dataset = BloodMNISTDataset(args.data, args.split, mean=mean, std=std, augment=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = build_model(model_name, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    losses = []
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.append(float(loss.detach().cpu()) * labels.size(0))
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    metrics = classification_metrics(y_true_arr, y_pred_arr, CLASS_NAMES)
    metrics["loss"] = float(np.sum(losses) / len(y_true_arr))
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["device"] = describe_device(device)

    if args.output_dir is None:
        output_dir = checkpoint_path.parent.parent / f"eval_{args.split}"
    else:
        output_dir = Path(args.output_dir)
    output_dir = ensure_dir(output_dir)
    ensure_dir(output_dir / "figures")
    ensure_dir(output_dir / "tables")

    save_json(metrics, output_dir / f"{args.split}_metrics.json")
    per_class_dataframe(metrics).to_csv(output_dir / "tables" / f"{args.split}_per_class_metrics.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"]).to_csv(output_dir / "tables" / f"{args.split}_confusion_matrix.csv", index=False)
    plot_confusion_matrix(metrics["confusion_matrix"], output_dir / "figures" / f"{args.split}_confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], output_dir / "figures" / f"{args.split}_confusion_matrix_raw.png", normalize=False)

    if args.split == "test":
        images, labels = load_split(args.data, args.split)
        save_misclassified_examples(
            images,
            labels,
            y_pred_arr,
            output_dir / "figures" / f"{args.split}_misclassified_examples.png",
        )

    print(
        f"{args.split} metrics | loss {metrics['loss']:.4f} acc {metrics['accuracy']:.4f} "
        f"macroF1 {metrics['macro_f1']:.4f} weightedF1 {metrics['weighted_f1']:.4f}"
    )
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()

