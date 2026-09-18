"""Deterministic sampling helpers for smoke-training experiments."""

from collections.abc import Sequence
import random

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.data import validate_required_columns


def sample_multilabel_dataframe(
    dataframe: pd.DataFrame,
    labels: Sequence[str],
    sample_size: int,
    random_seed: int,
) -> pd.DataFrame:
    """Return a deterministic multi-label-stratified sample."""
    _validate_sample_inputs(dataframe, labels, sample_size)

    if sample_size == len(dataframe):
        return dataframe.copy(deep=True).reset_index(drop=True)

    label_matrix = dataframe[list(labels)].eq(1).astype(int)
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_size,
        test_size=len(dataframe) - sample_size,
        random_state=random_seed,
    )
    sample_indices, _ = next(splitter.split(dataframe, label_matrix))
    sample_indices = _correct_sample_indices(
        selected_indices=list(sample_indices),
        row_count=len(dataframe),
        sample_size=sample_size,
        random_seed=random_seed,
    )

    return dataframe.iloc[sample_indices].copy(deep=True).reset_index(drop=True)


def _correct_sample_indices(
    selected_indices: list[int],
    row_count: int,
    sample_size: int,
    random_seed: int,
) -> list[int]:
    unique_selected = list(dict.fromkeys(int(index) for index in selected_indices))
    rng = random.Random(random_seed)

    if len(unique_selected) == sample_size:
        return unique_selected

    if len(unique_selected) < sample_size:
        selected_set = set(unique_selected)
        remaining_indices = [
            index for index in range(row_count) if index not in selected_set
        ]
        rng.shuffle(remaining_indices)
        deficit = sample_size - len(unique_selected)
        return unique_selected + remaining_indices[:deficit]

    return rng.sample(unique_selected, sample_size)


def _validate_sample_inputs(
    dataframe: pd.DataFrame,
    labels: Sequence[str],
    sample_size: int,
) -> None:
    if dataframe.empty:
        raise ValueError("dataframe must not be empty")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError(f"sample_size must be an int, got {type(sample_size).__name__}")
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")
    if sample_size > len(dataframe):
        raise ValueError(
            "sample_size must be less than or equal to source row count: "
            f"sample_size={sample_size}, rows={len(dataframe)}"
        )

    label_order = tuple(labels)
    if not label_order:
        raise ValueError("labels must not be empty")
    validate_required_columns(dataframe, label_order)
