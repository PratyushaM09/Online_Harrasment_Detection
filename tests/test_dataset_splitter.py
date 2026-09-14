import pandas as pd
import pytest

from src.data import (
    EXPECTED_LABEL_COLUMNS,
    compare_label_distributions,
    split_dataset,
    validate_partition_integrity,
)


def _split_dataframe(row_count: int = 120) -> pd.DataFrame:
    rows = []

    for index in range(row_count):
        rows.append(
            {
                "id": f"id-{index:03d}",
                "comment_text": f"Original comment {index}",
                "toxic": int(index % 3 == 0),
                "severe_toxic": int(index % 10 == 0),
                "obscene": int(index % 5 == 0),
                "threat": int(index % 10 == 1),
                "insult": int(index % 4 == 0),
                "identity_hate": int(index % 10 == 2),
                "source_note": f"extra-{index}",
            }
        )

    return pd.DataFrame(rows)


def _run_split(dataframe: pd.DataFrame):
    return split_dataset(
        dataframe,
        labels=EXPECTED_LABEL_COLUMNS,
        train_ratio=0.80,
        validation_ratio=0.10,
        test_ratio=0.10,
        random_seed=42,
    )


def test_split_is_reproducible_with_same_seed():
    dataframe = _split_dataframe()

    first_split = _run_split(dataframe)
    second_split = _run_split(dataframe)

    for first_partition, second_partition in zip(first_split, second_split):
        pd.testing.assert_frame_equal(first_partition, second_partition)


def test_every_source_row_appears_exactly_once():
    dataframe = _split_dataframe()
    train, validation, test = _run_split(dataframe)
    combined_ids = pd.concat([train["id"], validation["id"], test["id"]])

    assert len(combined_ids) == len(dataframe)
    assert set(combined_ids) == set(dataframe["id"])
    assert combined_ids.is_unique


def test_no_id_appears_in_more_than_one_partition():
    dataframe = _split_dataframe()
    train, validation, test = _run_split(dataframe)

    validate_partition_integrity(dataframe, train, validation, test)
    assert set(train["id"]).isdisjoint(validation["id"])
    assert set(train["id"]).isdisjoint(test["id"])
    assert set(validation["id"]).isdisjoint(test["id"])


def test_partition_sizes_are_close_to_requested_ratios():
    dataframe = _split_dataframe()
    train, validation, test = _run_split(dataframe)

    assert len(train) / len(dataframe) == pytest.approx(0.80, abs=0.05)
    assert len(validation) / len(dataframe) == pytest.approx(0.10, abs=0.05)
    assert len(test) / len(dataframe) == pytest.approx(0.10, abs=0.05)


def test_split_does_not_mutate_input_dataframe():
    dataframe = _split_dataframe()
    original = dataframe.copy(deep=True)

    _run_split(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_split_preserves_all_original_columns():
    dataframe = _split_dataframe()
    train, validation, test = _run_split(dataframe)

    for partition in (train, validation, test):
        assert list(partition.columns) == list(dataframe.columns)


def test_rare_positive_labels_remain_represented_when_possible():
    dataframe = _split_dataframe()
    train, validation, test = _run_split(dataframe)

    for label in ("severe_toxic", "threat", "identity_hate"):
        assert train[label].eq(1).sum() > 0
        assert validation[label].eq(1).sum() > 0
        assert test[label].eq(1).sum() > 0


def test_invalid_ratios_are_rejected():
    dataframe = _split_dataframe()

    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        split_dataset(
            dataframe,
            labels=EXPECTED_LABEL_COLUMNS,
            train_ratio=0.70,
            validation_ratio=0.20,
            test_ratio=0.20,
            random_seed=42,
        )


def test_duplicate_ids_fail_clearly():
    dataframe = _split_dataframe()
    dataframe.loc[1, "id"] = dataframe.loc[0, "id"]

    with pytest.raises(ValueError, match="Duplicate dataset IDs"):
        _run_split(dataframe)


def test_distribution_comparison_reports_all_partitions():
    dataframe = _split_dataframe()
    train, validation, test = _run_split(dataframe)

    comparison = compare_label_distributions(dataframe, train, validation, test)

    assert list(comparison.index) == list(EXPECTED_LABEL_COLUMNS)
    assert list(comparison.columns) == ["full", "train", "validation", "test"]
