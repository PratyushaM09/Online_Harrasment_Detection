"""Validation metrics for XLM-R smoke-training experiments."""

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from src.evaluation import evaluate_multilabel
from src.model import logits_to_probabilities


@dataclass(frozen=True)
class ValidationResult:
    """Structured validation outputs from a model evaluation pass."""

    loss: float
    logits: torch.Tensor
    probabilities: torch.Tensor
    labels: torch.Tensor
    aggregate_metrics: dict[str, float]
    threshold: float


def build_validation_result(
    validation_loss: float,
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_names: Sequence[str],
    threshold: float,
) -> ValidationResult:
    """Convert validation logits into probabilities and fixed-threshold metrics."""
    probabilities = logits_to_probabilities(logits)
    evaluation = evaluate_multilabel(
        labels.cpu().numpy(),
        probabilities.cpu().numpy(),
        label_names,
        threshold=threshold,
    )

    return ValidationResult(
        loss=float(validation_loss),
        logits=logits,
        probabilities=probabilities,
        labels=labels,
        aggregate_metrics=evaluation.aggregate_metrics,
        threshold=float(threshold),
    )


def get_model_output_value(output: Any, key: str):
    """Read a Hugging Face-style output value from an object or mapping."""
    if isinstance(output, dict):
        return output[key]
    return getattr(output, key)
