"""Reusable multi-label evaluation metrics."""

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Structured multi-label evaluation result."""

    threshold: float
    labels: tuple[str, ...]
    aggregate_metrics: dict[str, float]
    per_label_metrics: dict[str, dict[str, float | int]]
    roc_auc: dict[str, Any]
    average_precision: dict[str, Any]
    confusion_matrices: dict[str, dict[str, int]]
    error_counts: dict[str, dict[str, int]]


def evaluate_multilabel(
    y_true,
    probabilities,
    labels: Sequence[str],
    threshold: float = 0.5,
) -> EvaluationResult:
    """Evaluate multi-label probability outputs at a fixed threshold."""
    label_order = tuple(labels)
    _validate_threshold(threshold)
    y_true_array, probability_array = _validate_inputs(
        y_true,
        probabilities,
        label_order,
    )
    predictions = (probability_array >= threshold).astype(int)

    aggregate_metrics = _calculate_aggregate_metrics(y_true_array, predictions)
    per_label_metrics = _calculate_per_label_metrics(
        y_true_array,
        predictions,
        label_order,
    )
    roc_auc = _calculate_roc_auc(y_true_array, probability_array, label_order)
    average_precision = _calculate_average_precision(
        y_true_array,
        probability_array,
        label_order,
    )
    confusion_matrices = _calculate_confusion_matrices(
        y_true_array,
        predictions,
        label_order,
    )
    error_counts = {
        label: {
            "false_positives": counts["FP"],
            "false_negatives": counts["FN"],
        }
        for label, counts in confusion_matrices.items()
    }

    return EvaluationResult(
        threshold=float(threshold),
        labels=label_order,
        aggregate_metrics=aggregate_metrics,
        per_label_metrics=per_label_metrics,
        roc_auc=roc_auc,
        average_precision=average_precision,
        confusion_matrices=confusion_matrices,
        error_counts=error_counts,
    )


def _calculate_aggregate_metrics(y_true, predictions) -> dict[str, float]:
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="micro",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }


def _calculate_per_label_metrics(
    y_true,
    predictions,
    labels: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        average=None,
        zero_division=0,
    )

    return {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }


def _calculate_roc_auc(y_true, probabilities, labels: Sequence[str]) -> dict[str, Any]:
    per_label = {}
    for index, label in enumerate(labels):
        per_label[label] = _safe_roc_auc(
            y_true[:, index],
            probabilities[:, index],
        )

    defined_values = [value for value in per_label.values() if value is not None]

    return {
        "per_label": per_label,
        "macro": _mean_or_none(defined_values),
        "micro": _safe_roc_auc(y_true.ravel(), probabilities.ravel()),
    }


def _calculate_average_precision(
    y_true,
    probabilities,
    labels: Sequence[str],
) -> dict[str, Any]:
    per_label = {}
    for index, label in enumerate(labels):
        per_label[label] = _safe_average_precision(
            y_true[:, index],
            probabilities[:, index],
        )

    defined_values = [value for value in per_label.values() if value is not None]

    return {
        "per_label": per_label,
        "macro": _mean_or_none(defined_values),
        "micro": _safe_average_precision(y_true.ravel(), probabilities.ravel()),
    }


def _calculate_confusion_matrices(
    y_true,
    predictions,
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    matrices = {}
    for index, label in enumerate(labels):
        tn, fp, fn, tp = confusion_matrix(
            y_true[:, index],
            predictions[:, index],
            labels=[0, 1],
        ).ravel()
        matrices[label] = {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }

    return matrices


def _safe_roc_auc(y_true_column, probability_column) -> float | None:
    if set(np.asarray(y_true_column).astype(int).tolist()) != {0, 1}:
        return None

    try:
        score = float(roc_auc_score(y_true_column, probability_column))
    except ValueError:
        return None

    return _finite_or_none(score)


def _safe_average_precision(y_true_column, probability_column) -> float | None:
    try:
        return float(average_precision_score(y_true_column, probability_column))
    except ValueError:
        return None


def _mean_or_none(values: Sequence[float]) -> float | None:
    finite_values = [value for value in values if value is not None and isfinite(value)]
    if not finite_values:
        return None
    return float(fsum(finite_values) / len(finite_values))


def _finite_or_none(value: float) -> float | None:
    if not isfinite(value):
        return None
    return float(value)


def _validate_inputs(y_true, probabilities, labels: Sequence[str]):
    if not labels:
        raise ValueError("labels must not be empty")

    y_true_array = np.asarray(y_true)
    probability_array = np.asarray(probabilities)

    if y_true_array.ndim != 2:
        raise ValueError("y_true must be a 2D array-like structure")
    if probability_array.ndim != 2:
        raise ValueError("probabilities must be a 2D array-like structure")
    if y_true_array.shape != probability_array.shape:
        raise ValueError(
            "y_true and probabilities must have the same shape: "
            f"y_true={y_true_array.shape}, probabilities={probability_array.shape}"
        )
    if y_true_array.shape[1] != len(labels):
        raise ValueError(
            "Number of label columns must match labels: "
            f"columns={y_true_array.shape[1]}, labels={len(labels)}"
        )
    if not np.isin(y_true_array, [0, 1]).all():
        raise ValueError("y_true must contain only 0/1 values")
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must contain only finite values")
    if ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("probabilities must be between 0 and 1")

    return y_true_array.astype(int), probability_array.astype(float)


def _validate_threshold(threshold: float) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError(
            f"threshold must be a number between 0 and 1, got {type(threshold).__name__}"
        )
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1")
