from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    EXPECTED_COLUMNS,
    EXPECTED_LABEL_COLUMNS,
    get_dataset_metadata,
    load_training_data,
    validate_required_columns,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_toxic_comments.csv"


def test_valid_fixture_loads_successfully():
    dataframe = load_training_data(FIXTURE_PATH)

    assert list(dataframe.columns) == list(EXPECTED_COLUMNS)
    assert len(dataframe) == 3


def test_expected_required_columns_are_accepted():
    dataframe = pd.DataFrame(columns=EXPECTED_COLUMNS)

    validate_required_columns(dataframe)


def test_missing_required_column_raises_clear_error(tmp_path):
    broken_path = tmp_path / "missing_comment_text.csv"
    dataframe = pd.read_csv(FIXTURE_PATH).drop(columns=["comment_text"])
    dataframe.to_csv(broken_path, index=False)

    with pytest.raises(ValueError, match="Dataset is missing required columns"):
        load_training_data(broken_path)


def test_row_limiting_works():
    dataframe = load_training_data(FIXTURE_PATH, limit_rows=2)

    assert len(dataframe) == 2


def test_missing_file_raises_clear_error(tmp_path):
    missing_path = tmp_path / "train.csv"

    with pytest.raises(FileNotFoundError, match="Training dataset file not found"):
        load_training_data(missing_path)


def test_metadata_helper_reports_basic_counts():
    dataframe = load_training_data(FIXTURE_PATH)
    metadata = get_dataset_metadata(dataframe)

    assert metadata["number_of_rows"] == 3
    assert metadata["number_of_columns"] == len(EXPECTED_COLUMNS)
    assert metadata["label_names"] == list(EXPECTED_LABEL_COLUMNS)
    assert metadata["missing_comment_text_count"] == 0

