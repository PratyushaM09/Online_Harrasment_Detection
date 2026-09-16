from src.evaluation import evaluate_multilabel, format_evaluation_report


LABELS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)


def test_report_includes_threshold_aggregate_metrics_and_all_labels():
    y_true = [[1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]]
    probabilities = [[0.9, 0.1, 0.2, 0.1, 0.8, 0.2], [0.2, 0.1, 0.7, 0.1, 0.3, 0.2]]
    result = evaluate_multilabel(y_true, probabilities, LABELS)

    report = format_evaluation_report(result, title="TEST REPORT")

    assert "TEST REPORT" in report
    assert "Threshold: 0.50" in report
    assert "Aggregate Metrics" in report
    assert "Micro Precision" in report
    assert "Macro F1" in report
    for label in LABELS:
        assert label in report


def test_report_output_is_deterministic():
    y_true = [[1, 0], [0, 1]]
    probabilities = [[0.8, 0.2], [0.3, 0.7]]
    result = evaluate_multilabel(y_true, probabilities, ("a", "b"))

    assert format_evaluation_report(result) == format_evaluation_report(result)

