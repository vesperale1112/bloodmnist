"""Metric helpers for multiclass classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

from .constants import CLASS_NAMES, NUM_CLASSES


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    y_score: np.ndarray | None = None,
    calibration_bins: int = 15,
) -> dict[str, Any]:
    labels = list(range(NUM_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    per_class_auc: list[float | None] = [None] * NUM_CLASSES
    score_array: np.ndarray | None = None
    if y_score is not None:
        score_array = np.asarray(y_score, dtype=np.float64)
        if score_array.shape != (len(y_true), NUM_CLASSES):
            raise ValueError(f"Expected y_score shape {(len(y_true), NUM_CLASSES)}, got {score_array.shape}.")
        score_array = np.clip(score_array, 1e-12, 1.0)
        score_array = score_array / score_array.sum(axis=1, keepdims=True)
        for idx in labels:
            binary_true = (y_true == idx).astype(int)
            if binary_true.min() == binary_true.max():
                continue
            per_class_auc[idx] = float(roc_auc_score(binary_true, score_array[:, idx]))

    per_class = []
    for idx, name in enumerate(class_names):
        per_class.append(
            {
                "class_id": idx,
                "class_name": name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "auc_ovr": per_class_auc[idx],
                "support": int(support[idx]),
            }
        )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
    }

    if score_array is not None:
        valid_auc = [(idx, auc) for idx, auc in enumerate(per_class_auc) if auc is not None]
        if valid_auc:
            auc_values = np.asarray([auc for _, auc in valid_auc], dtype=np.float64)
            auc_weights = np.asarray([support[idx] for idx, _ in valid_auc], dtype=np.float64)
            metrics["macro_auc_ovr"] = float(auc_values.mean())
            metrics["weighted_auc_ovr"] = float(np.average(auc_values, weights=auc_weights))
        else:
            metrics["macro_auc_ovr"] = None
            metrics["weighted_auc_ovr"] = None

        true_prob = score_array[np.arange(len(y_true)), y_true]
        one_hot = np.eye(NUM_CLASSES, dtype=np.float64)[y_true]
        calibration = calibration_metrics(y_true, y_pred, score_array, n_bins=calibration_bins)
        metrics.update(
            {
                "nll": float(-np.mean(np.log(true_prob))),
                "brier_score": float(np.mean(np.sum((score_array - one_hot) ** 2, axis=1))),
                "ece": calibration["ece"],
                "mce": calibration["mce"],
                "calibration_n_bins": calibration_bins,
                "calibration_bins": calibration["bins"],
            }
        )

    return metrics


def calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 15,
) -> dict[str, Any]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    confidences = np.max(y_score, axis=1)
    correct = (y_pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    total = len(y_true)
    for bin_idx in range(n_bins):
        left = float(edges[bin_idx])
        right = float(edges[bin_idx + 1])
        if bin_idx == 0:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences > left) & (confidences <= right)

        count = int(mask.sum())
        fraction = float(count / total) if total else 0.0
        if count:
            accuracy = float(correct[mask].mean())
            confidence = float(confidences[mask].mean())
            gap = float(abs(accuracy - confidence))
            ece += fraction * gap
            mce = max(mce, gap)
        else:
            accuracy = None
            confidence = None
            gap = None

        rows.append(
            {
                "bin_id": bin_idx,
                "confidence_min": left,
                "confidence_max": right,
                "count": count,
                "fraction": fraction,
                "accuracy": accuracy,
                "confidence": confidence,
                "gap": gap,
            }
        )

    return {"ece": float(ece), "mce": float(mce), "bins": rows}


def per_class_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(metrics["per_class"])
