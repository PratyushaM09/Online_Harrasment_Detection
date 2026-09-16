import pandas as pd
import pytest

from src.baseline import (
    load_baseline_artifacts,
    save_baseline_artifacts,
    train_baseline,
)
from src.data import EXPECTED_LABEL_COLUMNS


def _baseline_dataframe() -> pd.DataFrame:
    rows = []
    for index in range(18):
        row = {
            "id": f"id-{index}",
            "comment_text": f"baseline sample text {index} repeated repeated",
        }
        for label_index, label in enumerate(EXPECTED_LABEL_COLUMNS):
            row[label] = int((index + label_index) % 4 == 0)
        rows.append(row)

    return pd.DataFrame(rows)


def test_train_baseline_returns_fitted_vectorizer_and_classifier():
    result = train_baseline(
        _baseline_dataframe(),
        labels=EXPECTED_LABEL_COLUMNS,
        random_seed=42,
    )

    assert hasattr(result.vectorizer, "vocabulary_")
    assert hasattr(result.classifier, "estimators_")
    assert result.labels == tuple(EXPECTED_LABEL_COLUMNS)


def test_train_baseline_does_not_mutate_input_dataframe():
    dataframe = _baseline_dataframe()
    original = dataframe.copy(deep=True)

    train_baseline(dataframe, labels=EXPECTED_LABEL_COLUMNS, random_seed=42)

    pd.testing.assert_frame_equal(dataframe, original)


def test_all_configured_labels_are_required():
    dataframe = _baseline_dataframe().drop(columns=["toxic"])

    with pytest.raises(ValueError, match="Missing required label columns"):
        train_baseline(dataframe, labels=EXPECTED_LABEL_COLUMNS, random_seed=42)


def test_invalid_label_values_fail_clearly():
    dataframe = _baseline_dataframe()
    dataframe.loc[0, "toxic"] = 2

    with pytest.raises(ValueError, match="Label columns must contain only 0/1"):
        train_baseline(dataframe, labels=EXPECTED_LABEL_COLUMNS, random_seed=42)


def test_tiny_valid_multilabel_fixture_can_train():
    result = train_baseline(
        _baseline_dataframe(),
        labels=EXPECTED_LABEL_COLUMNS,
        random_seed=42,
    )

    assert len(result.labels) == 6


def test_artifact_persistence_requires_overwrite(tmp_path):
    result = train_baseline(
        _baseline_dataframe(),
        labels=EXPECTED_LABEL_COLUMNS,
        random_seed=42,
    )
    save_baseline_artifacts(result, tmp_path, random_seed=42)

    with pytest.raises(FileExistsError, match="Baseline artifacts already exist"):
        save_baseline_artifacts(result, tmp_path, random_seed=42)


def test_artifact_persistence_round_trip(tmp_path):
    result = train_baseline(
        _baseline_dataframe(),
        labels=EXPECTED_LABEL_COLUMNS,
        random_seed=42,
    )
    save_baseline_artifacts(result, tmp_path, random_seed=42)

    loaded = load_baseline_artifacts(tmp_path)

    assert loaded.labels == result.labels
    assert loaded.default_threshold == result.default_threshold
