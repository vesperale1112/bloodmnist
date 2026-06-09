"""Summarize multiple experiment runs into report-ready tables and figures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bloodmnist_mpl_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize BloodMNIST experiment runs.")
    parser.add_argument("--runs-dir", default="outputs/runs", help="Directory containing run subdirectories.")
    parser.add_argument("--output-dir", default="outputs/summary", help="Directory for comparison outputs.")
    parser.add_argument("--include-smoke", action="store_true", help="Include smoke/check runs in the summary.")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_test_metrics(run_dir: Path) -> dict:
    metrics = read_json(run_dir / "metrics" / "test_metrics.json")
    eval_metrics_path = run_dir / "eval_test" / "test_metrics.json"
    if not eval_metrics_path.exists():
        return metrics

    eval_metrics = read_json(eval_metrics_path)
    for key in (
        "macro_auc_ovr",
        "weighted_auc_ovr",
        "ece",
        "mce",
        "nll",
        "brier_score",
    ):
        if eval_metrics.get(key) is not None:
            metrics[key] = eval_metrics[key]
    return metrics


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(output_dir / "figures")
    ensure_dir(output_dir / "tables")

    rows = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if not args.include_smoke and run_dir.name.startswith(("smoke_", "check_")):
            continue
        metrics_path = run_dir / "metrics" / "test_metrics.json"
        config_path = run_dir / "run_config.json"
        best_path = run_dir / "metrics" / "best_summary.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        metrics = load_test_metrics(run_dir)
        config = read_json(config_path)
        best = read_json(best_path) if best_path.exists() else {}
        run_args = config.get("args", {})
        rows.append(
            {
                "run_name": run_dir.name,
                "model": run_args.get("model"),
                "loss": run_args.get("loss"),
                "augment": run_args.get("augment"),
                "weighted_sampler": run_args.get("weighted_sampler"),
                "best_epoch": best.get("best_epoch"),
                "test_accuracy": metrics.get("accuracy"),
                "test_macro_f1": metrics.get("macro_f1"),
                "test_weighted_f1": metrics.get("weighted_f1"),
                "test_macro_auc_ovr": metrics.get("macro_auc_ovr"),
                "test_weighted_auc_ovr": metrics.get("weighted_auc_ovr"),
                "test_ece": metrics.get("ece"),
                "test_nll": metrics.get("nll"),
                "test_brier_score": metrics.get("brier_score"),
            }
        )

    if not rows:
        raise RuntimeError(f"No completed runs with test_metrics.json found in {runs_dir}.")

    df = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False)
    df.to_csv(output_dir / "tables" / "experiment_comparison.csv", index=False)

    plot_df = df.melt(
        id_vars=["run_name"],
        value_vars=[
            metric
            for metric in ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_macro_auc_ovr"]
            if metric in df.columns and df[metric].notna().any()
        ],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="run_name", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.xlabel("Run")
    plt.ylabel("Score")
    plt.title("BloodMNIST experiment comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "experiment_comparison.png", dpi=220)
    plt.close()

    print(f"Saved experiment summary to {output_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
