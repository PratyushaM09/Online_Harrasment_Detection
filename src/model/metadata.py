"""Lightweight model metadata."""

from dataclasses import dataclass
from typing import Sequence

from src.model.factory import PROBLEM_TYPE


@dataclass(frozen=True)
class ModelMetadata:
    """Configuration metadata for the XLM-R multi-label classifier."""

    model_name: str
    labels: tuple[str, ...]
    num_labels: int
    problem_type: str
    max_length: int


def create_model_metadata(
    model_name: str,
    labels: Sequence[str],
    max_length: int,
) -> ModelMetadata:
    """Create model metadata without trained performance values."""
    label_order = tuple(labels)

    return ModelMetadata(
        model_name=model_name,
        labels=label_order,
        num_labels=len(label_order),
        problem_type=PROBLEM_TYPE,
        max_length=max_length,
    )

