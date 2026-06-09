"""Plotting helpers for BloodMNIST experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bloodmnist_mpl_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .constants import CLASS_NAMES, NUM_CLASSES, SPLITS
from .utils import ensure_dir


def plot_class_distribution(distribution: dict[str, list[int]], output_path: str | Path) -> None:
    rows = []
    for split in SPLITS:
        for class_id, count in enumerate(distribution[split]):
            rows.append({"split": split, "class_id": class_id, "class_name": CLASS_NAMES[class_id], "count": count})
    df = pd.DataFrame(rows)

    plt.figure(figsize=(12, 5))
    sns.barplot(data=df, x="class_name", y="count", hue="split")
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.title("BloodMNIST class distribution")
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_history(history: list[dict[str, Any]], output_path: str | Path) -> None:
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_accuracy"], label="train")
    axes[1].plot(df["epoch"], df["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(df["epoch"], df["train_macro_f1"], label="train")
    axes[2].plot(df["epoch"], df["val_macro_f1"], label="val")
    axes[2].set_title("Macro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_confusion_matrix(
    confusion: list[list[int]] | np.ndarray,
    output_path: str | Path,
    normalize: bool = True,
) -> None:
    cm = np.asarray(confusion, dtype=np.float32)
    values = cm
    fmt = ".2f" if normalize else ".0f"
    title = "Normalized confusion matrix" if normalize else "Confusion matrix"
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        values = np.divide(cm, np.maximum(row_sum, 1.0))

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        values,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_reliability_diagram(calibration_bins: list[dict[str, Any]], output_path: str | Path) -> None:
    df = pd.DataFrame(calibration_bins)
    if df.empty:
        return

    bin_width = float(df["confidence_max"].iloc[0] - df["confidence_min"].iloc[0])
    centers = (df["confidence_min"] + df["confidence_max"]) / 2
    accuracy = df["accuracy"].fillna(0.0).astype(float)
    confidence = df["confidence"].fillna(centers).astype(float)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        centers,
        accuracy,
        width=bin_width * 0.9,
        align="center",
        alpha=0.75,
        edgecolor="black",
        label="Accuracy",
    )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2, label="Perfect calibration")
    valid = df["count"] > 0
    ax.scatter(confidence[valid], accuracy[valid], color="#d62728", s=28, zorder=3, label="Non-empty bins")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def save_sample_grid(
    images: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    examples_per_class: int = 4,
) -> None:
    fig, axes = plt.subplots(NUM_CLASSES, examples_per_class, figsize=(examples_per_class * 2.0, NUM_CLASSES * 1.7))
    for class_id in range(NUM_CLASSES):
        idxs = np.where(labels == class_id)[0][:examples_per_class]
        for col in range(examples_per_class):
            ax = axes[class_id, col]
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[class_id], fontsize=8)
            if col < len(idxs):
                ax.imshow(images[idxs[col]])
    plt.suptitle("BloodMNIST examples by class", y=0.995)
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def save_misclassified_examples(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    max_examples: int = 24,
) -> None:
    wrong = np.where(y_true != y_pred)[0][:max_examples]
    cols = 6
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes_array = np.asarray(axes).reshape(-1)
    for ax in axes_array:
        ax.axis("off")

    if len(wrong) == 0:
        axes_array[0].text(0.5, 0.5, "No errors found", ha="center", va="center")
    else:
        for ax, idx in zip(axes_array, wrong):
            ax.imshow(images[idx])
            ax.set_title(f"T: {CLASS_NAMES[int(y_true[idx])]}\nP: {CLASS_NAMES[int(y_pred[idx])]}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path, dpi=220)
    plt.close(fig)
