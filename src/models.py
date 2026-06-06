"""CNN model definitions for BloodMNIST classification."""

from __future__ import annotations

import torch
from torch import nn

from .constants import NUM_CLASSES


class SimpleCNN(nn.Module):
    """Small baseline CNN for 28x28 RGB images."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImprovedCNN(nn.Module):
    """Deeper CNN with normalization, regularization, and global pooling."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, pool=True, dropout=0.05),
            ConvBlock(32, 64, pool=True, dropout=0.10),
            ConvBlock(64, 128, pool=False, dropout=0.15),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if normalized == "improved_cnn":
        return ImprovedCNN(num_classes=num_classes)
    raise ValueError("Unknown model {!r}; choose from simple_cnn, improved_cnn.".format(name))

