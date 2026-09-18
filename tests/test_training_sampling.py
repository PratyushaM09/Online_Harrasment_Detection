import pandas as pd
import pytest

from src.data import EXPECTED_LABEL_COLUMNS
from src.training import sample_multilabel_dataframe


def _frame(row_count=20):
    frame = pd.DataFrame(
        {
            "id": [f"id-{index}" for index in range(row_count)],
            "comment_text": [f"row {index}" for index in range(row_count)],
        }
    )
    for label_index, label in enumerate(EXPECTED_LABEL_COLUMNS):
        frame[label] = [
            int((index + label_index) % 3 == 0) for index in range(row_count)
        ]
    return frame


def test_sampling_is_deterministic_with_same_seed():
    first = sample_multilabel_dataframe(_frame(), EXPECTED_LABEL_COLUMNS, 8, 42)
    second = sample_multilabel_dataframe(_frame(), EXPECTED_LABEL_COLUMNS, 8, 42)

    assert first["id"].tolist() == second["id"].tolist()


@pytest.mark.parametrize("sample_size", [1, 7, 9])
def test_requested_sample_size_is_respected(sample_size):
    sample = sample_multilabel_dataframe(
        _frame(),
        EXPECTED_LABEL_COLUMNS,
        sample_size,
        42,
    )

    assert len(sample) == sample_size


def test_sample_contains_no_duplicate_rows():
    sample = sample_multilabel_dataframe(_frame(), EXPECTED_LABEL_COLUMNS, 7, 42)

    assert sample["id"].is_unique


def test_sample_size_equal_to_source_returns_full_copy():
    frame = _frame(row_count=6)

    sample = sample_multilabel_dataframe(frame, EXPECTED_LABEL_COLUMNS, 6, 42)

    assert len(sample) == 6
    assert sample["id"].tolist() == frame["id"].tolist()
    assert sample is not frame


def test_source_dataframe_is_not_changed_by_sampling():
    frame = _frame()
    original = frame.copy(deep=True)

    _ = sample_multilabel_dataframe(frame, EXPECTED_LABEL_COLUMNS, 8, 42)

    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("sample_size", [0, -1])
def test_invalid_sample_size_values_are_rejected(sample_size):
    with pytest.raises(ValueError, match="sample_size must be greater than 0"):
        sample_multilabel_dataframe(_frame(), EXPECTED_LABEL_COLUMNS, sample_size, 42)


@pytest.mark.parametrize("sample_size", [True, 1.5, "10"])
def test_invalid_sample_size_types_are_rejected(sample_size):
    with pytest.raises(TypeError, match="sample_size must be an int"):
        sample_multilabel_dataframe(_frame(), EXPECTED_LABEL_COLUMNS, sample_size, 42)


def test_sample_size_larger_than_source_is_rejected_clearly():
    with pytest.raises(ValueError, match="less than or equal to source row count"):
        sample_multilabel_dataframe(_frame(row_count=5), EXPECTED_LABEL_COLUMNS, 6, 42)
