import pandas as pd
import pytest

from src.baseline import (
    create_baseline_classifier,
    fit_transform_training_text,
    get_label_matrix,
    predict_labels,
    predict_probabilities,
)
from src.baseline.model import fit_classifier
from src.baseline.vectorizer import create_tfidf_vectorizer
from src.data import EXPECTED_LABEL_COLUMNS


def _training_dataframe() -> pd.DataFrame:
    rows = []
    for index in range(12):
        row = {
            "comment_text": f"sample text {index} repeated",
        }
        for label_index, label in enumerate(EXPECTED_LABEL_COLUMNS):
            row[label] = int((index + label_index) % 3 == 0)
        rows.append(row)

    return pd.DataFrame(rows)


def _fit_fixture_classifier(random_seed: int = 42):
    dataframe = _training_dataframe()
    vectorizer = create_tfidf_vectorizer(min_df=1)
    _, features = fit_transform_training_text(dataframe, vectorizer=vectorizer)
    labels = get_label_matrix(dataframe, EXPECTED_LABEL_COLUMNS)
    classifier = create_baseline_classifier(random_seed=random_seed)
    fit_classifier(classifier, features, labels)

    return classifier, features, labels


def test_label_order_is_preserved():
    dataframe = _training_dataframe()
    label_matrix = get_label_matrix(dataframe, EXPECTED_LABEL_COLUMNS)

    assert label_matrix.shape == (len(dataframe), len(EXPECTED_LABEL_COLUMNS))
    assert label_matrix[0].tolist() == dataframe[list(EXPECTED_LABEL_COLUMNS)].iloc[0].tolist()


def test_probability_output_shape_is_correct():
    classifier, features, _ = _fit_fixture_classifier()

    probabilities = predict_probabilities(classifier, features)

    assert probabilities.shape == (features.shape[0], len(EXPECTED_LABEL_COLUMNS))


def test_predictions_are_binary():
    classifier, features, _ = _fit_fixture_classifier()

    predictions = predict_labels(classifier, features)

    assert set(predictions.ravel()).issubset({0, 1})


def test_threshold_parameter_changes_predictions():
    classifier, features, _ = _fit_fixture_classifier()

    low_threshold_predictions = predict_labels(classifier, features, threshold=0.1)
    high_threshold_predictions = predict_labels(classifier, features, threshold=0.9)

    assert low_threshold_predictions.sum() >= high_threshold_predictions.sum()


def test_same_seed_produces_reproducible_probabilities():
    first_classifier, features, _ = _fit_fixture_classifier(random_seed=42)
    second_classifier, _, _ = _fit_fixture_classifier(random_seed=42)

    first_probabilities = predict_probabilities(first_classifier, features)
    second_probabilities = predict_probabilities(second_classifier, features)

    assert first_probabilities.tolist() == second_probabilities.tolist()


def test_label_matrix_shape_is_validated():
    classifier, features, labels = _fit_fixture_classifier()

    with pytest.raises(ValueError, match="Feature row count must match label row count"):
        fit_classifier(classifier, features, labels[:-1])
