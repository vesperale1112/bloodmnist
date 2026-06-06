"""BloodMNIST data loading, normalization, and class-imbalance helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .constants import NUM_CLASSES, SPLITS


def load_split(npz_path: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")

    data = np.load(npz_path)
    images = data[f"{split}_images"]
    labels = data[f"{split}_labels"].reshape(-1).astype(np.int64)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected NHWC RGB images, got shape {images.shape}.")
    return images, labels


def split_distribution(npz_path: str | Path) -> dict[str, list[int]]:
    summary: dict[str, list[int]] = {}
    for split in SPLITS:
        _, labels = load_split(npz_path, split)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        summary[split] = counts.astype(int).tolist()
    return summary


def compute_train_mean_std(npz_path: str | Path) -> tuple[list[float], list[float]]:
    images, _ = load_split(npz_path, "train")
    x = images.astype(np.float32) / 255.0
    mean = x.mean(axis=(0, 1, 2))
    std = x.std(axis=(0, 1, 2))
    std = np.maximum(std, 1e-6)
    return mean.tolist(), std.tolist()


class BloodMNISTDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        split: str,
        mean: list[float] | tuple[float, float, float] | None = None,
        std: list[float] | tuple[float, float, float] | None = None,
        augment: bool = False,
    ) -> None:
        self.images, self.labels = load_split(npz_path, split)
        self.split = split

        steps: list[Any] = [transforms.ToPILImage()]
        if augment:
            steps.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.2),
                    transforms.RandomRotation(degrees=12),
                    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
                ]
            )
        steps.append(transforms.ToTensor())
        if mean is not None and std is not None:
            steps.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.transform(self.images[idx])
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return image, label


def compute_class_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    counts = np.bincount(labels.reshape(-1), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_data_loaders(
    npz_path: str | Path,
    batch_size: int,
    num_workers: int,
    augment_train: bool,
    weighted_sampler: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    mean, std = compute_train_mean_std(npz_path)
    train_dataset = BloodMNISTDataset(npz_path, "train", mean=mean, std=std, augment=augment_train)
    val_dataset = BloodMNISTDataset(npz_path, "val", mean=mean, std=std, augment=False)
    test_dataset = BloodMNISTDataset(npz_path, "test", mean=mean, std=std, augment=False)

    sampler = None
    shuffle = True
    if weighted_sampler:
        class_weights = compute_class_weights(train_dataset.labels)
        sample_weights = class_weights[torch.as_tensor(train_dataset.labels, dtype=torch.long)]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=shuffle, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    meta = {
        "mean": mean,
        "std": std,
        "class_distribution": split_distribution(npz_path),
        "class_weights": compute_class_weights(train_dataset.labels).tolist(),
    }
    return train_loader, val_loader, test_loader, meta

