"""Classification metrics."""
from __future__ import annotations

from typing import Dict

import numpy as np


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                                 precision_score, recall_score)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def format_metric(mean: float, std: float = 0.0) -> str:
    return f"{mean:.4f} +/- {std:.4f}"
