import math

import pytest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.evaluation import evaluate_multilabel


LABELS = ("toxic", "severe_toxic", "obscene")


def _fixture():
    y_true = [
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ]
    probabilities = [
        [0.9, 0.2, 0.8],
        [0.4, 0.7, 0.3],
        [0.6, 0.8, 0.2],
        [0.1, 0.4, 0.9],
    ]
    return y_true, probabilities


def test_aggregate_metrics_match_sklearn():
    y_true, probabilities = _fixture()
    result = evaluate_multilabel(y_true, probabilities, LABELS, threshold=0.5)
    predictions = [[int(value >= 0.5) for value in row] for row in probabilities]
    micro = precision_recall_fscore_support(
        y_true,
        predictions,
        average="micro",
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    assert result.aggregate_metrics["micro_precision"] == pytest.approx(micro[0])
    assert result.aggregate_metrics["micro_recall"] == pytest.approx(micro[1])
    assert result.aggregate_metrics["micro_f1"] == pytest.approx(micro[2])
    assert result.aggregate_metrics["macro_precision"] == pytest.approx(macro[0])
    assert result.aggregate_metrics["macro_recall"] == pytest.approx(macro[1])
    assert result.aggregate_metrics["macro_f1"] == pytest.approx(macro[2])


def test_per_label_metrics_preserve_label_order():
    y_true, probabilities = _fixture()
    result = evaluate_multilabel(y_true, probabilities, LABELS)

    assert tuple(result.per_label_metrics) == LABELS


def test_threshold_conversion_changes_predictions():
    y_true, probabilities = _fixture()
    low_threshold = evaluate_multilabel(y_true, probabilities, LABELS, threshold=0.3)
    high_threshold = evaluate_multilabel(y_true, probabilities, LABELS, threshold=0.8)
    low_true_positives = sum(
        matrix["TP"] for matrix in low_threshold.confusion_matrices.values()
    )
    high_true_positives = sum(
        matrix["TP"] for matrix in high_threshold.confusion_matrices.values()
    )

    assert low_true_positives >= high_true_positives


def test_confusion_matrix_counts_are_correct():
    y_true, probabilities = _fixture()
    result = evaluate_multilabel(y_true, probabilities, LABELS)
    predictions = [int(row[0] >= 0.5) for row in probabilities]
    tn, fp, fn, tp = confusion_matrix(
        [row[0] for row in y_true],
        predictions,
        labels=[0, 1],
    ).ravel()

    assert result.confusion_matrices["toxic"] == {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    assert result.error_counts["toxic"]["false_positives"] == int(fp)
    assert result.error_counts["toxic"]["false_negatives"] == int(fn)


def test_roc_auc_and_average_precision_use_probabilities():
    y_true, probabilities = _fixture()
    result = evaluate_multilabel(y_true, probabilities, LABELS)

    assert result.roc_auc["per_label"]["toxic"] == pytest.approx(
        roc_auc_score([row[0] for row in y_true], [row[0] for row in probabilities])
    )
    assert result.average_precision["per_label"]["toxic"] == pytest.approx(
        average_precision_score(
            [row[0] for row in y_true],
            [row[0] for row in probabilities],
        )
    )


def test_single_class_roc_auc_is_handled_safely():
    y_true = [[0, 1], [0, 0], [0, 1]]
    probabilities = [[0.1, 0.8], [0.2, 0.3], [0.4, 0.7]]
    result = evaluate_multilabel(y_true, probabilities, ("rare", "other"))

    assert result.roc_auc["per_label"]["rare"] is None
    assert result.roc_auc["per_label"]["other"] is not None


def test_all_zero_label_roc_auc_is_none():
    y_true = [[0, 1], [0, 0], [0, 1]]
    probabilities = [[0.1, 0.8], [0.2, 0.3], [0.4, 0.7]]
    result = evaluate_multilabel(y_true, probabilities, ("all_zero", "valid"))

    assert result.roc_auc["per_label"]["all_zero"] is None


def test_all_one_label_roc_auc_is_none():
    y_true = [[1, 1], [1, 0], [1, 1]]
    probabilities = [[0.1, 0.8], [0.2, 0.3], [0.4, 0.7]]
    result = evaluate_multilabel(y_true, probabilities, ("all_one", "valid"))

    assert result.roc_auc["per_label"]["all_one"] is None


def test_valid_two_class_label_roc_auc_is_float():
    y_true = [[0], [1], [0], [1]]
    probabilities = [[0.1], [0.8], [0.3], [0.9]]
    result = evaluate_multilabel(y_true, probabilities, ("valid",))

    assert isinstance(result.roc_auc["per_label"]["valid"], float)
    assert result.roc_auc["macro"] == pytest.approx(
        result.roc_auc["per_label"]["valid"]
    )


def test_roc_auc_structure_emits_no_nan():
    y_true = [[0, 1], [0, 0], [0, 1]]
    probabilities = [[0.1, 0.8], [0.2, 0.3], [0.4, 0.7]]
    result = evaluate_multilabel(y_true, probabilities, ("undefined", "valid"))

    roc_values = [
        result.roc_auc["macro"],
        result.roc_auc["micro"],
        *result.roc_auc["per_label"].values(),
    ]

    assert all(value is None or not math.isnan(value) for value in roc_values)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same shape"):
        evaluate_multilabel([[1, 0]], [[0.5, 0.5], [0.2, 0.8]], ("a", "b"))


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        evaluate_multilabel([[1]], [[0.5]], ("a",), threshold=1.5)


def test_non_finite_probabilities_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        evaluate_multilabel([[1]], [[math.nan]], ("a",))
