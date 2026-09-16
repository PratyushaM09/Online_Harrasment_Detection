"""Persistence helpers for evaluation results."""

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import EvaluationResult


def save_evaluation_results(
    results: dict[str, EvaluationResult],
    output_directory: str | Path,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Save evaluation metrics as JSON plus a compact per-label CSV."""
    output_path = Path(output_directory)
    json_paths = {
        split_name: output_path / f"{split_name}_metrics.json"
        for split_name in results
    }
    per_label_path = output_path / "per_label_metrics.csv"
    all_paths = (*json_paths.values(), per_label_path)
    existing_paths = [path for path in all_paths if path.exists()]

    if existing_paths and not overwrite:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Evaluation artifacts already exist. Use --overwrite to replace: "
            f"{existing}"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    for split_name, result in results.items():
        json_paths[split_name].write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    _build_per_label_frame(results).to_csv(per_label_path, index=False)

    return all_paths


def _build_per_label_frame(results: dict[str, EvaluationResult]) -> pd.DataFrame:
    rows = []
    for split_name, result in results.items():
        for label in result.labels:
            metrics = result.per_label_metrics[label]
            confusion = result.confusion_matrices[label]
            errors = result.error_counts[label]
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "support": metrics["support"],
                    "roc_auc": result.roc_auc["per_label"][label],
                    "average_precision": result.average_precision["per_label"][label],
                    "TN": confusion["TN"],
                    "FP": confusion["FP"],
                    "FN": confusion["FN"],
                    "TP": confusion["TP"],
                    "false_positives": errors["false_positives"],
                    "false_negatives": errors["false_negatives"],
                }
            )

    return pd.DataFrame(rows)

