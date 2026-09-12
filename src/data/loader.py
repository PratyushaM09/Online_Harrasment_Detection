"""Minimal Jigsaw dataset loading and validation."""

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.data.dataset_paths import get_training_data_path


EXPECTED_LABEL_COLUMNS: tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)

EXPECTED_COLUMNS: tuple[str, ...] = (
    "id",
    "comment_text",
    *EXPECTED_LABEL_COLUMNS,
)


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str] = EXPECTED_COLUMNS,
) -> None:
    """Validate that all required Jigsaw columns are present."""
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")


def load_training_data(
    csv_path: str | Path | None = None,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    """Load and validate the Jigsaw training CSV."""
    if limit_rows is not None and limit_rows < 0:
        raise ValueError("limit_rows must be greater than or equal to 0")

    resolved_path = Path(csv_path) if csv_path is not None else get_training_data_path()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Training dataset file not found: {resolved_path}")

    dataframe = pd.read_csv(resolved_path, nrows=limit_rows)
    validate_required_columns(dataframe)

    return dataframe


def get_dataset_metadata(
    dataframe: pd.DataFrame,
    label_names: Sequence[str] = EXPECTED_LABEL_COLUMNS,
) -> dict[str, Any]:
    """Return basic dataset metadata without modifying the data."""
    validate_required_columns(dataframe)

    return {
        "number_of_rows": int(dataframe.shape[0]),
        "number_of_columns": int(dataframe.shape[1]),
        "label_names": list(label_names),
        "missing_comment_text_count": int(dataframe["comment_text"].isna().sum()),
    }

