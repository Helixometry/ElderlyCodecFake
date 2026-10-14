"""Evaluation metrics for ElderlyCodecFake."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, equal_error_rate
from typing import Dict


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, confidences: np.ndarray = None) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "f1_macro": f1_score(targets, predictions, average='macro'),
        "f1_weighted": f1_score(targets, predictions, average='weighted'),
    }

    if confidences is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(targets, confidences)
        except:
            metrics["roc_auc"] = 0.0

        # Compute EER (Equal Error Rate)
        try:
            metrics["eer"] = equal_error_rate(targets, confidences)
        except:
            metrics["eer"] = 0.0

    return metrics


def print_metrics(metrics: Dict[str, float], language: str = "") -> None:
    """Pretty-print metrics."""
    if language:
        print(f"\n{language.upper()}")
        print("-" * 60)

    for metric_name, value in sorted(metrics.items()):
        print(f"  {metric_name:20s}: {value:.4f}")
