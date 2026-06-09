"""Generate report-ready error-example grids from saved BloodMNIST checkpoints."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bloodmnist_mpl_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .constants import CLASS_NAMES, NUM_CLASSES
from .dataset import BloodMNISTDataset, load_split
from .models import build_model
from .utils import describe_device, ensure_dir, resolve_device


DEFAULT_RUNS = (
    ("resnet18_compare", "outputs/runs/resnet18_compare/checkpoints/best.pt"),
    ("improved_cnn_weighted_ce", "outputs/runs/improved_cnn_weighted_ce/checkpoints/best.pt"),
    ("improved_cnn_ce", "outputs/runs/improved_cnn_ce/checkpoints/best.pt"),
    ("simple_cnn_aug_ce", "outputs/runs/simple_cnn_aug_ce/checkpoints/best.pt"),
    ("simple_cnn_ce", "outputs/runs/simple_cnn_ce/checkpoints/best.pt"),
)


@dataclass
class PredictionResult:
    run_name: str
    checkpoint_path: Path
    y_true: np.ndarray
    y_pred: np.ndarray
    confidence: np.ndarray
    true_probability: np.ndarray

    @property
    def wrong_indices(self) -> np.ndarray:
        return np.where(self.y_true != self.y_pred)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BloodMNIST error-example figures.")
    parser.add_argument("--data", default="bloodmnist.npz", help="Path to bloodmnist.npz.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="outputs/summary/error_examples")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec as name=checkpoint_path. Defaults to the completed formal runs.",
    )
    parser.add_argument("--max-examples", type=int, default=24)
    parser.add_argument("--overview-per-model", type=int, default=6)
    parser.add_argument("--pair-examples", type=int, default=4)
    parser.add_argument("--top-pairs", type=int, default=6)
    return parser.parse_args()


def parse_run_specs(run_specs: list[str]) -> list[tuple[str, Path]]:
    if not run_specs:
        return [(name, Path(path)) for name, path in DEFAULT_RUNS if Path(path).exists()]

    parsed: list[tuple[str, Path]] = []
    for spec in run_specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
            parsed.append((name, Path(path)))
        else:
            path = Path(spec)
            parsed.append((path.parent.parent.name, path))
    return parsed


def torch_load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_checkpoint_model(checkpoint: dict[str, Any], device: torch.device) -> nn.Module:
    model_name = checkpoint.get("model_name", checkpoint.get("args", {}).get("model", "improved_cnn"))
    model = build_model(model_name, num_classes=NUM_CLASSES)
    state_dict = checkpoint["model_state_dict"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        if str(model_name).lower().replace("-", "_") != "resnet18":
            raise
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(state_dict)
    return model.to(device).eval()


def checkpoint_image_size(checkpoint: dict[str, Any]) -> int | None:
    args = checkpoint.get("args", {})
    image_size = args.get("image_size")
    if args.get("pretrained") and image_size is None:
        return 224
    return image_size


def predict_run(
    run_name: str,
    checkpoint_path: Path,
    data_path: str | Path,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> PredictionResult:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    meta = checkpoint.get("meta", {})
    dataset = BloodMNISTDataset(
        data_path,
        split,
        mean=meta.get("mean"),
        std=meta.get("std"),
        augment=False,
        image_size=checkpoint_image_size(checkpoint),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_checkpoint_model(checkpoint, device)

    y_true: list[int] = []
    y_pred: list[int] = []
    confidence: list[float] = []
    true_probability: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            confidence.extend(probs.max(dim=1).values.cpu().numpy().tolist())
            true_probability.extend(probs.gather(1, labels.to(device).view(-1, 1)).squeeze(1).cpu().numpy().tolist())

    return PredictionResult(
        run_name=run_name,
        checkpoint_path=checkpoint_path,
        y_true=np.asarray(y_true, dtype=np.int64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
        confidence=np.asarray(confidence, dtype=np.float32),
        true_probability=np.asarray(true_probability, dtype=np.float32),
    )


def sorted_wrong_indices(result: PredictionResult, max_examples: int) -> np.ndarray:
    wrong = result.wrong_indices
    order = np.argsort(result.confidence[wrong])[::-1]
    return wrong[order[:max_examples]]


def add_error_cell(ax: plt.Axes, image: np.ndarray, result: PredictionResult, idx: int, include_confidence: bool = True) -> None:
    ax.imshow(image)
    title = f"T: {CLASS_NAMES[int(result.y_true[idx])]}\nP: {CLASS_NAMES[int(result.y_pred[idx])]}"
    if include_confidence:
        title += f"\nconf: {result.confidence[idx]:.2f}"
    ax.set_title(title, fontsize=7, pad=2)
    ax.axis("off")


def save_high_confidence_errors(
    images: np.ndarray,
    result: PredictionResult,
    output_path: Path,
    max_examples: int,
    cols: int = 6,
) -> None:
    indices = sorted_wrong_indices(result, max_examples)
    rows = max(1, int(np.ceil(max(len(indices), 1) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.45))
    axes_array = np.asarray(axes).reshape(-1)
    for ax in axes_array:
        ax.axis("off")

    if len(indices) == 0:
        axes_array[0].text(0.5, 0.5, "No errors found", ha="center", va="center")
    else:
        for ax, idx in zip(axes_array, indices):
            add_error_cell(ax, images[idx], result, int(idx))

    fig.suptitle(f"{result.run_name}: high-confidence misclassified examples", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_model_error_overview(
    images: np.ndarray,
    results: list[PredictionResult],
    output_path: Path,
    examples_per_model: int,
) -> None:
    cols = examples_per_model
    rows = len(results)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.05, rows * 2.45), squeeze=False)

    for row, result in enumerate(results):
        indices = sorted_wrong_indices(result, examples_per_model)
        for col in range(cols):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(indices):
                add_error_cell(ax, images[int(indices[col])], result, int(indices[col]))
            elif col == 0:
                ax.text(0.5, 0.5, "No errors found", ha="center", va="center")

    fig.suptitle("High-confidence test errors by model", fontsize=12)
    fig.tight_layout(rect=(0.14, 0, 1, 0.965))
    for row, result in enumerate(results):
        bbox = axes[row, 0].get_position()
        fig.text(0.015, (bbox.y0 + bbox.y1) / 2, result.run_name, ha="left", va="center", fontsize=8)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_top_confusion_pair_examples(
    images: np.ndarray,
    result: PredictionResult,
    output_path: Path,
    top_pairs: int,
    examples_per_pair: int,
) -> None:
    wrong = result.wrong_indices
    if len(wrong) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.text(0.5, 0.5, "No errors found", ha="center", va="center")
        ax.axis("off")
        ensure_dir(output_path.parent)
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        return

    pair_counts: dict[tuple[int, int], int] = {}
    for idx in wrong:
        pair = (int(result.y_true[idx]), int(result.y_pred[idx]))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    pairs = sorted(pair_counts, key=lambda pair: pair_counts[pair], reverse=True)[:top_pairs]

    rows = len(pairs)
    cols = examples_per_pair
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.4), squeeze=False)

    for row, pair in enumerate(pairs):
        true_id, pred_id = pair
        pair_indices = wrong[(result.y_true[wrong] == true_id) & (result.y_pred[wrong] == pred_id)]
        order = np.argsort(result.confidence[pair_indices])[::-1]
        pair_indices = pair_indices[order[:examples_per_pair]]
        for col in range(cols):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(pair_indices):
                add_error_cell(ax, images[int(pair_indices[col])], result, int(pair_indices[col]))

    fig.suptitle(f"{result.run_name}: most frequent confusion pairs", fontsize=12)
    fig.tight_layout(rect=(0.18, 0, 1, 0.965))
    for row, pair in enumerate(pairs):
        true_id, pred_id = pair
        bbox = axes[row, 0].get_position()
        fig.text(
            0.015,
            (bbox.y0 + bbox.y1) / 2,
            f"{CLASS_NAMES[true_id]}\n-> {CLASS_NAMES[pred_id]}\nn={pair_counts[pair]}",
            ha="left",
            va="center",
            fontsize=7,
        )
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    images, _ = load_split(args.data, args.split)
    run_specs = parse_run_specs(args.run)
    if not run_specs:
        raise RuntimeError("No checkpoints found. Pass --run name=checkpoint_path or train the default runs first.")

    results: list[PredictionResult] = []
    for run_name, checkpoint_path in run_specs:
        result = predict_run(
            run_name=run_name,
            checkpoint_path=checkpoint_path,
            data_path=args.data,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        results.append(result)

    written: list[Path] = []
    for result in results:
        out_path = output_dir / f"{result.run_name}_{args.split}_high_confidence_errors.png"
        save_high_confidence_errors(images, result, out_path, max_examples=args.max_examples)
        written.append(out_path)

    overview_path = output_dir / f"{args.split}_model_error_overview.png"
    save_model_error_overview(images, results, overview_path, examples_per_model=args.overview_per_model)
    written.append(overview_path)

    for result in results:
        pair_path = output_dir / f"{result.run_name}_{args.split}_top_confusion_pairs.png"
        save_top_confusion_pair_examples(
            images,
            result,
            pair_path,
            top_pairs=args.top_pairs,
            examples_per_pair=args.pair_examples,
        )
        written.append(pair_path)

    print(f"Device: {describe_device(device)}")
    for result in results:
        print(
            f"{result.run_name}: {len(result.wrong_indices)} / {result.y_true.size} "
            f"{args.split} errors ({len(result.wrong_indices) / result.y_true.size:.2%})"
        )
    print("Saved figures:")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
