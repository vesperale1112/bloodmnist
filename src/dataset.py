"""BloodMNIST data loading, normalization, and class-imbalance helpers (Pure PyTorch, no torchvision)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
        image_size: int | None = None,
    ) -> None:
        self.images, self.labels = load_split(npz_path, split)
        self.split = split
        self.augment = augment
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1) if mean is not None else None
        self.std = torch.tensor(std).view(3, 1, 1) if std is not None else None

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_np = self.images[idx]
        # HWC to CHW
        image_tensor = torch.from_numpy(image_np.transpose((2, 0, 1))).float() / 255.0
        
        if self.image_size is not None:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image_tensor = torch.flip(image_tensor, [-1])
            if torch.rand(1).item() > 0.8:
                image_tensor = torch.flip(image_tensor, [-2])

        if self.mean is not None and self.std is not None:
            image_tensor = (image_tensor - self.mean) / self.std

        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return image_tensor, label


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
    image_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    mean, std = compute_train_mean_std(npz_path)
    train_dataset = BloodMNISTDataset(npz_path, "train", mean=mean, std=std, augment=augment_train, image_size=image_size)
    val_dataset = BloodMNISTDataset(npz_path, "val", mean=mean, std=std, augment=False, image_size=image_size)
    test_dataset = BloodMNISTDataset(npz_path, "test", mean=mean, std=std, augment=False, image_size=image_size)

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
