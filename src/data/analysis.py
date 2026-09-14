"""Dataset analysis helpers for Jigsaw toxic comment data."""

from typing import Any, Sequence

import pandas as pd

from src.data.loader import EXPECTED_LABEL_COLUMNS, validate_required_columns


def analyze_label_distribution(
    dataframe: pd.DataFrame,
    label_names: Sequence[str] = EXPECTED_LABEL_COLUMNS,
) -> dict[str, dict[str, int | float | None]]:
    """Return per-label positive, negative, percentage, and imbalance statistics."""
    validate_required_columns(dataframe, label_names)
    row_count = len(dataframe)
    distribution = {}

    for label in label_names:
        positive_count = int(dataframe[label].eq(1).sum())
        negative_count = int(row_count - positive_count)
        positive_percentage = (
            (positive_count / row_count) * 100 if row_count > 0 else 0.0
        )
        imbalance_ratio = (
            None if positive_count == 0 else negative_count / positive_count
        )

        distribution[label] = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_percentage": positive_percentage,
            "imbalance_ratio": imbalance_ratio,
        }

    return distribution


def summarize_comment_labels(
    dataframe: pd.DataFrame,
    label_names: Sequence[str] = EXPECTED_LABEL_COLUMNS,
) -> dict[str, int | float]:
    """Return row-level multi-label summary statistics."""
    validate_required_columns(dataframe, label_names)
    row_count = len(dataframe)
    label_counts_per_row = dataframe[list(label_names)].eq(1).sum(axis=1)
    rows_with_toxicity = int(label_counts_per_row.gt(0).sum())
    rows_without_toxicity = int(row_count - rows_with_toxicity)

    return {
        "number_of_rows": int(row_count),
        "rows_with_toxicity": rows_with_toxicity,
        "rows_without_toxicity": rows_without_toxicity,
        "percentage_with_toxicity": (
            (rows_with_toxicity / row_count) * 100 if row_count > 0 else 0.0
        ),
        "average_positive_labels_per_row": (
            float(label_counts_per_row.mean()) if row_count > 0 else 0.0
        ),
        "maximum_simultaneous_positive_labels": (
            int(label_counts_per_row.max()) if row_count > 0 else 0
        ),
    }


def analyze_text_quality(
    dataframe: pd.DataFrame,
    text_column: str = "comment_text",
) -> dict[str, int]:
    """Return missing and blank text counts without modifying comment text."""
    validate_required_columns(dataframe, [text_column])
    comments = dataframe[text_column]

    missing_count = int(comments.isna().sum())
    blank_count = int(comments.dropna().astype(str).str.strip().eq("").sum())

    return {
        "missing_comment_text_count": missing_count,
        "blank_comment_text_count": blank_count,
    }


def calculate_label_cooccurrence(
    dataframe: pd.DataFrame,
    label_names: Sequence[str] = EXPECTED_LABEL_COLUMNS,
) -> pd.DataFrame:
    """Return counts for rows where each pair of labels is positive together."""
    validate_required_columns(dataframe, label_names)
    label_indicators = dataframe[list(label_names)].eq(1).astype(int)

    return label_indicators.T.dot(label_indicators)


def analyze_dataset(
    dataframe: pd.DataFrame,
    label_names: Sequence[str] = EXPECTED_LABEL_COLUMNS,
) -> dict[str, Any]:
    """Run all current dataset analysis helpers."""
    return {
        "label_distribution": analyze_label_distribution(dataframe, label_names),
        "comment_summary": summarize_comment_labels(dataframe, label_names),
        "text_quality": analyze_text_quality(dataframe),
        "label_cooccurrence": calculate_label_cooccurrence(dataframe, label_names),
    }

