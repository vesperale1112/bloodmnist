"""Metric helpers for multiclass classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .constants import CLASS_NAMES, NUM_CLASSES


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
) -> dict[str, Any]:
    labels = list(range(NUM_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = []
    for idx, name in enumerate(class_names):
        per_class.append(
            {
                "class_id": idx,
                "class_name": name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def per_class_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(metrics["per_class"])

