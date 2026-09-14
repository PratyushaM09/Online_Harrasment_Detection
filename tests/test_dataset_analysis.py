import pandas as pd
import pytest

from src.data import (
    EXPECTED_LABEL_COLUMNS,
    analyze_dataset,
    analyze_label_distribution,
    analyze_text_quality,
    calculate_label_cooccurrence,
    summarize_comment_labels,
)


def _analysis_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "comment_text": [
                "A neutral comment.",
                "An insulting toxic comment.",
                "   ",
                None,
            ],
            "toxic": [0, 1, 0, 1],
            "severe_toxic": [0, 0, 0, 0],
            "obscene": [0, 0, 1, 0],
            "threat": [0, 0, 0, 0],
            "insult": [0, 1, 1, 0],
            "identity_hate": [0, 0, 0, 0],
        }
    )


def test_label_counts_are_correct():
    dataframe = _analysis_dataframe()
    distribution = analyze_label_distribution(dataframe)

    for label in EXPECTED_LABEL_COLUMNS:
        expected_positive = int(dataframe[label].eq(1).sum())
        expected_negative = len(dataframe) - expected_positive

        assert distribution[label]["positive_count"] == expected_positive
        assert distribution[label]["negative_count"] == expected_negative


def test_positive_percentages_are_correct():
    dataframe = _analysis_dataframe()
    distribution = analyze_label_distribution(dataframe)

    for label in EXPECTED_LABEL_COLUMNS:
        expected_percentage = dataframe[label].eq(1).sum() / len(dataframe) * 100

        assert distribution[label]["positive_percentage"] == pytest.approx(
            expected_percentage
        )


def test_zero_positive_label_does_not_crash_imbalance_calculation():
    distribution = analyze_label_distribution(_analysis_dataframe())

    assert distribution["threat"]["positive_count"] == 0
    assert distribution["threat"]["imbalance_ratio"] is None


def test_rows_with_and_without_toxicity_are_counted_correctly():
    dataframe = _analysis_dataframe()
    summary = summarize_comment_labels(dataframe)
    positive_rows = dataframe[list(EXPECTED_LABEL_COLUMNS)].eq(1).sum(axis=1).gt(0)

    assert summary["rows_with_toxicity"] == int(positive_rows.sum())
    assert summary["rows_without_toxicity"] == int((~positive_rows).sum())


def test_average_labels_per_row_is_correct():
    dataframe = _analysis_dataframe()
    summary = summarize_comment_labels(dataframe)
    labels_per_row = dataframe[list(EXPECTED_LABEL_COLUMNS)].eq(1).sum(axis=1)

    assert summary["average_positive_labels_per_row"] == pytest.approx(
        labels_per_row.mean()
    )
    assert summary["maximum_simultaneous_positive_labels"] == int(labels_per_row.max())


def test_missing_and_blank_comments_are_counted_separately():
    text_quality = analyze_text_quality(_analysis_dataframe())

    assert text_quality["missing_comment_text_count"] == 1
    assert text_quality["blank_comment_text_count"] == 1


def test_cooccurrence_counts_are_correct():
    dataframe = _analysis_dataframe()
    cooccurrence = calculate_label_cooccurrence(dataframe)

    assert cooccurrence.loc["toxic", "toxic"] == 2
    assert cooccurrence.loc["toxic", "insult"] == 1
    assert cooccurrence.loc["obscene", "insult"] == 1
    assert cooccurrence.loc["threat", "toxic"] == 0


def test_analysis_functions_do_not_mutate_input_dataframe():
    dataframe = _analysis_dataframe()
    original = dataframe.copy(deep=True)

    analyze_dataset(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)

