import json

import pandas as pd
import pytest

from src.evaluation import evaluate_multilabel, save_evaluation_results


def _result():
    return evaluate_multilabel(
        [[1, 0], [0, 1]],
        [[0.8, 0.2], [0.3, 0.7]],
        ("toxic", "insult"),
    )


def test_evaluation_persistence_writes_json_and_csv(tmp_path):
    result = _result()

    paths = save_evaluation_results({"validation": result}, tmp_path)

    assert tmp_path / "validation_metrics.json" in paths
    assert tmp_path / "per_label_metrics.csv" in paths
    assert (tmp_path / "validation_metrics.json").exists()
    assert (tmp_path / "per_label_metrics.csv").exists()


def test_evaluation_json_contains_structured_metrics(tmp_path):
    result = _result()
    save_evaluation_results({"validation": result}, tmp_path)
    payload = json.loads((tmp_path / "validation_metrics.json").read_text())

    assert payload["threshold"] == 0.5
    assert payload["labels"] == ["toxic", "insult"]
    assert "aggregate_metrics" in payload
    assert "roc_auc" in payload


def test_per_label_csv_contains_expected_rows(tmp_path):
    result = _result()
    save_evaluation_results({"validation": result, "test": result}, tmp_path)

    frame = pd.read_csv(tmp_path / "per_label_metrics.csv")

    assert set(frame["split"]) == {"validation", "test"}
    assert set(frame["label"]) == {"toxic", "insult"}
    assert {"false_positives", "false_negatives"}.issubset(frame.columns)


def test_evaluation_persistence_requires_overwrite(tmp_path):
    result = _result()
    save_evaluation_results({"validation": result}, tmp_path)

    with pytest.raises(FileExistsError, match="Evaluation artifacts already exist"):
        save_evaluation_results({"validation": result}, tmp_path)


def test_evaluation_persistence_allows_overwrite(tmp_path):
    result = _result()
    save_evaluation_results({"validation": result}, tmp_path)

    paths = save_evaluation_results({"validation": result}, tmp_path, overwrite=True)

    assert paths
