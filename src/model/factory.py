"""Factory helpers for the XLM-R multi-label classifier."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch


PROBLEM_TYPE = "multi_label_classification"


def create_label_mappings(labels: Sequence[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Create deterministic label mappings from configured label order."""
    label_order = _validate_labels(labels)
    label2id = {label: index for index, label in enumerate(label_order)}
    id2label = {index: label for label, index in label2id.items()}

    return label2id, id2label


def create_multilabel_model(
    model_name: str,
    labels: Sequence[str],
    model_loader: Any | None = None,
):
    """Create an XLM-R sequence classifier configured for multi-label output."""
    label_order = _validate_labels(labels)
    label2id, id2label = create_label_mappings(label_order)
    loader = model_loader or _load_auto_model_for_sequence_classification()

    return loader.from_pretrained(
        model_name,
        num_labels=len(label_order),
        label2id=label2id,
        id2label=id2label,
        problem_type=PROBLEM_TYPE,
    )


def logits_to_probabilities(logits):
    """Convert raw multi-label logits to independent sigmoid probabilities."""
    return torch.sigmoid(logits)


def build_label_targets(row: Mapping[str, Any], labels: Sequence[str]) -> torch.Tensor:
    """Build a floating-point multi-label target tensor from a row-like mapping."""
    label_order = _validate_labels(labels)
    missing_labels = [label for label in label_order if label not in row]
    if missing_labels:
        missing = ", ".join(missing_labels)
        raise ValueError(f"Missing required label values: {missing}")

    values = []
    for label in label_order:
        value = row[label]
        if value not in (0, 1):
            raise ValueError(f"Label value for {label} must be 0 or 1, got {value}")
        values.append(float(value))

    return torch.tensor(values, dtype=torch.float32)


def _validate_labels(labels: Sequence[str]) -> tuple[str, ...]:
    if labels is None:
        raise TypeError("labels must be a non-empty sequence of strings")

    label_order = tuple(labels)
    if not label_order:
        raise ValueError("labels must not be empty")
    if any(not isinstance(label, str) or not label for label in label_order):
        raise ValueError("labels must contain non-empty strings")
    if len(set(label_order)) != len(label_order):
        raise ValueError("labels must be unique")

    return label_order


def _load_auto_model_for_sequence_classification():
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification

