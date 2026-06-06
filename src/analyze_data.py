"""Generate dataset summary tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .constants import CLASS_NAMES
from .dataset import compute_train_mean_std, load_split, split_distribution
from .utils import ensure_dir, save_json
from .visualize import plot_class_distribution, save_sample_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze BloodMNIST dataset.")
    parser.add_argument("--data", default="bloodmnist.npz", help="Path to bloodmnist.npz.")
    parser.add_argument("--output-dir", default="outputs/dataset", help="Directory for dataset artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")

    distribution = split_distribution(args.data)
    mean, std = compute_train_mean_std(args.data)
    summary = {"class_names": CLASS_NAMES, "class_distribution": distribution, "train_mean": mean, "train_std": std}
    save_json(summary, output_dir / "dataset_summary.json")

    rows = []
    for split, counts in distribution.items():
        for class_id, count in enumerate(counts):
            rows.append({"split": split, "class_id": class_id, "class_name": CLASS_NAMES[class_id], "count": count})
    pd.DataFrame(rows).to_csv(tables_dir / "class_distribution.csv", index=False)

    train_images, train_labels = load_split(args.data, "train")
    plot_class_distribution(distribution, figures_dir / "class_distribution.png")
    save_sample_grid(train_images, train_labels, figures_dir / "sample_grid.png")

    print(f"Saved dataset artifacts to {output_dir}")


if __name__ == "__main__":
    main()

