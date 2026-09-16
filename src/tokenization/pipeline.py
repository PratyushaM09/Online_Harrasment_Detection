"""Reusable DataFrame tokenization pipeline helpers."""

from typing import Any, Sequence

import pandas as pd

from src.data.loader import EXPECTED_LABEL_COLUMNS
from src.tokenization.tokenizer import tokenize_batch


DEFAULT_TEXT_COLUMN = "comment_text"


def tokenize_dataframe(
    dataframe: pd.DataFrame,
    tokenizer,
    max_length: int,
    text_column: str = DEFAULT_TEXT_COLUMN,
    batch_size: int = 1000,
    label_columns: Sequence[str] = EXPECTED_LABEL_COLUMNS,
) -> pd.DataFrame:
    """Tokenize a DataFrame text column without mutating source data."""
    if text_column not in dataframe.columns:
        raise ValueError(f"Text column not found in DataFrame: {text_column}")
    _validate_batch_size(batch_size)

    texts = dataframe[text_column].tolist()
    _validate_text_values(texts, text_column)

    input_ids: list[Any] = []
    attention_masks: list[Any] = []

    for start_index in range(0, len(texts), batch_size):
        batch = texts[start_index : start_index + batch_size]
        encoded = tokenize_batch(batch, tokenizer, max_length=max_length)
        input_ids.extend(encoded["input_ids"])
        attention_masks.extend(encoded["attention_mask"])

    output_columns = {}
    if "id" in dataframe.columns:
        output_columns["id"] = dataframe["id"].tolist()

    for label_column in label_columns:
        if label_column in dataframe.columns:
            output_columns[label_column] = dataframe[label_column].tolist()

    output_columns["input_ids"] = input_ids
    output_columns["attention_mask"] = attention_masks

    return pd.DataFrame(output_columns)


def _validate_text_values(texts: Sequence[str], text_column: str) -> None:
    for index, text in enumerate(texts):
        if text is None:
            raise TypeError(f"{text_column}[{index}] must be a string, got None")
        if not isinstance(text, str):
            raise TypeError(
                f"{text_column}[{index}] must be a string, got {type(text).__name__}"
            )


def _validate_batch_size(batch_size: int) -> None:
    if not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be an int, got {type(batch_size).__name__}")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
