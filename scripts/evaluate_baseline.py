"""Evaluate saved classical baseline artifacts on validation and test splits."""

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import (  # noqa: E402
    load_baseline_artifacts,
    predict_probabilities,
    transform_text,
)
from src.baseline.model import get_label_matrix  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import validate_required_columns  # noqa: E402
from src.evaluation import (  # noqa: E402
    evaluate_multilabel,
    format_evaluation_report,
    save_evaluation_results,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved TF-IDF + logistic regression baseline artifacts."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save evaluation metrics under artifacts/evaluation/baseline.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation artifacts when used with --save.",
    )
    return parser.parse_args()


def main() -> int:
    """Evaluate the saved baseline without retraining or threshold tuning."""
    args = _parse_args()
    config = load_config()
    artifact_directory = PROJECT_ROOT / "models" / "baseline"

    try:
        baseline = _load_baseline_or_fail(artifact_directory)
        validation_dataframe = _load_processed_split("validation", config)
        test_dataframe = _load_processed_split("test", config)
        validation_result = _evaluate_split(
            validation_dataframe,
            baseline,
            "validation",
        )
        test_result = _evaluate_split(test_dataframe, baseline, "test")
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(error)
        return 1

    print(format_evaluation_report(validation_result, title="VALIDATION EVALUATION"))
    print()
    print(format_evaluation_report(test_result, title="TEST EVALUATION"))

    if args.save:
        output_directory = PROJECT_ROOT / "artifacts" / "evaluation" / "baseline"
        try:
            paths = save_evaluation_results(
                {"validation": validation_result, "test": test_result},
                output_directory,
                overwrite=args.overwrite,
            )
        except FileExistsError as error:
            print(error)
            return 1

        print()
        print("saved evaluation artifacts:")
        for path in paths:
            print(path)

    return 0


def _load_baseline_or_fail(artifact_directory: Path):
    required_paths = (
        artifact_directory / "vectorizer.joblib",
        artifact_directory / "classifier.joblib",
        artifact_directory / "metadata.json",
    )
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Saved baseline artifacts were not found. "
            "Run `python scripts/train_baseline.py --save` first. Missing: "
            f"{missing}"
        )

    return load_baseline_artifacts(artifact_directory)


def _load_processed_split(split_name: str, config) -> pd.DataFrame:
    path = PROJECT_ROOT / config.paths.processed_data / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}")

    return pd.read_csv(path)


def _evaluate_split(dataframe: pd.DataFrame, baseline, split_name: str):
    validate_required_columns(dataframe, ["id", "comment_text", *baseline.labels])
    features = transform_text(baseline.vectorizer, dataframe)
    probabilities = predict_probabilities(baseline.classifier, features)
    y_true = get_label_matrix(dataframe, baseline.labels)

    return evaluate_multilabel(
        y_true,
        probabilities,
        labels=baseline.labels,
        threshold=baseline.default_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())

