"""Train BloodMNIST CNN classifiers."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .constants import CLASS_NAMES, NUM_CLASSES
from .dataset import compute_class_weights, load_split, make_data_loaders
from .metrics import classification_metrics, per_class_dataframe
from .models import build_model
from .utils import configure_torch_runtime, count_parameters, describe_device, ensure_dir, resolve_device, save_json, set_seed
from .visualize import plot_class_distribution, plot_confusion_matrix, plot_history, save_misclassified_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BloodMNIST classifier.")
    parser.add_argument("--data", default="bloodmnist.npz", help="Path to bloodmnist.npz.")
    parser.add_argument("--output-dir", default="outputs/runs", help="Root directory for training outputs.")
    parser.add_argument("--run-name", default=None, help="Run directory name. Defaults to timestamp plus model/loss.")
    parser.add_argument("--model", default="improved_cnn", choices=["simple_cnn", "improved_cnn", "resnet18"])
    parser.add_argument("--loss", default="ce", choices=["ce", "weighted_ce"])
    parser.add_argument("--weighted-sampler", action="store_true", help="Use balanced sampling for training batches.")
    parser.add_argument("--augment", action="store_true", help="Enable training-time image augmentation.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights for supported models.")
    parser.add_argument("--image-size", type=int, default=None, help="Optional resize for input images (useful for pretrained models, e.g. 224).")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--skip-test", action="store_true", help="Skip final test-set evaluation.")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    scaler = torch.amp.GradScaler("cuda", enabled=training and device.type == "cuda")

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = labels.size(0)
        losses.append(float(loss.detach().cpu()) * batch_size)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    if not y_true:
        raise RuntimeError("No batches were processed. Check batch limits and dataloader size.")

    metrics = classification_metrics(np.asarray(y_true), np.asarray(y_pred), CLASS_NAMES)
    metrics["loss"] = float(np.sum(losses) / len(y_true))
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    best_metric: float,
    meta: dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": args.model,
            "num_classes": NUM_CLASSES,
            "epoch": epoch,
            "best_metric": best_metric,
            "args": vars(args),
            "meta": meta,
            "class_names": CLASS_NAMES,
        },
        path,
    )


def final_evaluation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    output_dir: Path,
    split: str,
) -> dict[str, Any]:
    metrics = run_epoch(model, loader, criterion, device, optimizer=None)
    save_json(metrics, output_dir / "metrics" / f"{split}_metrics.json")
    per_class_dataframe(metrics).to_csv(output_dir / "tables" / f"{split}_per_class_metrics.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"]).to_csv(output_dir / "tables" / f"{split}_confusion_matrix.csv", index=False)
    plot_confusion_matrix(metrics["confusion_matrix"], output_dir / "figures" / f"{split}_confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], output_dir / "figures" / f"{split}_confusion_matrix_raw.png", normalize=False)
    return metrics


def main() -> None:
    configure_torch_runtime()
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    run_name = args.run_name
    if run_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = "sampler" if args.weighted_sampler else args.loss
        run_name = f"{timestamp}_{args.model}_{suffix}"

    output_dir = ensure_dir(Path(args.output_dir) / run_name)
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "figures")
    ensure_dir(output_dir / "metrics")
    ensure_dir(output_dir / "tables")

    # If using pretrained models, default image size to 224 unless overridden
    image_size = args.image_size
    if args.pretrained and image_size is None:
        image_size = 224

    train_loader, val_loader, test_loader, meta = make_data_loaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=args.augment,
        weighted_sampler=args.weighted_sampler,
        image_size=image_size,
    )

    model = build_model(args.model, num_classes=NUM_CLASSES)
    # if model supports pretrained weights and user asked for them, try to load
    if args.pretrained and args.model.lower() == "resnet18":
        try:
            from torchvision.models import resnet18, ResNet18_Weights

            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        except Exception:
            # fall back to uninitialized if torchvision weights unavailable
            pass
    model = model.to(device)
    if args.loss == "weighted_ce":
        _, train_labels = load_split(args.data, "train")
        weights = compute_class_weights(train_labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    run_config = {
        "args": vars(args),
        "run_name": run_name,
        "device": describe_device(device),
        "num_parameters": count_parameters(model),
        "class_names": CLASS_NAMES,
        "data_meta": meta,
    }
    save_json(run_config, output_dir / "run_config.json")
    plot_class_distribution(meta["class_distribution"], output_dir / "figures" / "class_distribution.png")

    print(f"Run directory: {output_dir}")
    print(f"Device: {describe_device(device)}")
    print(f"Model: {args.model} ({count_parameters(model):,} trainable parameters)")

    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = output_dir / "checkpoints" / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            max_batches=args.max_val_batches,
        )
        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "tables" / "history.csv", index=False)
        save_json({"history": history}, output_dir / "metrics" / "history.json")
        plot_history(history, output_dir / "figures" / "training_curves.png")

        print(
            "Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_accuracy:.4f} macroF1 {train_macro_f1:.4f} "
            "| val loss {val_loss:.4f} acc {val_accuracy:.4f} macroF1 {val_macro_f1:.4f}".format(**row)
        )

        current_metric = val_metrics["macro_f1"]
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, args, epoch, best_metric, meta)
            save_json(val_metrics, output_dir / "metrics" / "best_val_metrics.json")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best val macro F1 {best_metric:.4f} at epoch {best_epoch}.")
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_summary = {"best_epoch": best_epoch, "best_val_macro_f1": best_metric, "best_checkpoint": str(best_path)}
    save_json(best_summary, output_dir / "metrics" / "best_summary.json")

    val_metrics = final_evaluation(model, val_loader, criterion, device, output_dir, "val")
    if not args.skip_test:
        test_metrics = final_evaluation(model, test_loader, criterion, device, output_dir, "test")
        test_images, test_labels = load_split(args.data, "test")
        all_pred: list[int] = []
        model.eval()
        with torch.no_grad():
            for images, _ in test_loader:
                logits = model(images.to(device, non_blocking=True))
                all_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        save_misclassified_examples(
            test_images,
            test_labels,
            np.asarray(all_pred),
            output_dir / "figures" / "test_misclassified_examples.png",
        )
        print(
            "Best epoch {best_epoch} | test acc {accuracy:.4f} macroF1 {macro_f1:.4f} weightedF1 {weighted_f1:.4f}".format(
                best_epoch=best_epoch,
                **test_metrics,
            )
        )
    else:
        print(f"Best epoch {best_epoch} | val acc {val_metrics['accuracy']:.4f} macroF1 {val_metrics['macro_f1']:.4f}")

    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
