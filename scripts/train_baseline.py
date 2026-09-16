"""Train the classical TF-IDF + logistic regression baseline."""

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import (  # noqa: E402
    predict_labels,
    save_baseline_artifacts,
    train_baseline,
    transform_text,
)
from src.baseline.model import get_label_matrix  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import validate_required_columns  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the TF-IDF + logistic regression baseline."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional deterministic sample size for train/validation splits.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save trained artifacts under models/baseline.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing saved artifacts when used with --save.",
    )
    return parser.parse_args()


def main() -> int:
    """Train the baseline and print a small sanity report."""
    args = _parse_args()
    config = load_config()

    try:
        train_dataframe = _load_processed_split("train", config)
        validation_dataframe = _load_processed_split("validation", config)
        test_dataframe = _load_processed_split("test", config)
        _validate_split(train_dataframe, config.labels)
        _validate_split(validation_dataframe, config.labels)
        _validate_split(test_dataframe, config.labels)

        if args.sample_size is not None:
            train_dataframe = _sample_dataframe(
                train_dataframe,
                args.sample_size,
                config.project.random_seed,
                "train",
            )
            validation_dataframe = _sample_dataframe(
                validation_dataframe,
                args.sample_size,
                config.project.random_seed,
                "validation",
            )

        result = train_baseline(
            train_dataframe,
            labels=config.labels,
            random_seed=config.project.random_seed,
        )
        validation_features = transform_text(
            result.vectorizer,
            validation_dataframe,
        )
        validation_true = get_label_matrix(validation_dataframe, result.labels)
        validation_predictions = predict_labels(
            result.classifier,
            validation_features,
            threshold=result.default_threshold,
        )
        validation_micro_f1 = f1_score(
            validation_true,
            validation_predictions,
            average="micro",
            zero_division=0,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(error)
        return 1

    print("BASELINE TRAINING COMPLETE")
    print(f"training rows: {len(train_dataframe)}")
    print(f"validation rows: {len(validation_dataframe)}")
    print(f"test rows loaded: {len(test_dataframe)}")
    print(f"TF-IDF feature count: {len(result.vectorizer.vocabulary_)}")
    print(f"labels: {', '.join(result.labels)}")
    print(f"validation micro-F1: {validation_micro_f1:.4f}")

    if args.save:
        output_directory = PROJECT_ROOT / "models" / "baseline"
        try:
            paths = save_baseline_artifacts(
                result,
                output_directory,
                random_seed=config.project.random_seed,
                overwrite=args.overwrite,
            )
        except FileExistsError as error:
            print(error)
            return 1

        print("saved artifacts:")
        for path in paths:
            print(path)

    return 0


def _load_processed_split(split_name: str, config) -> pd.DataFrame:
    path = PROJECT_ROOT / config.paths.processed_data / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}")

    return pd.read_csv(path)


def _validate_split(dataframe: pd.DataFrame, labels) -> None:
    validate_required_columns(dataframe, ["id", "comment_text", *labels])


def _sample_dataframe(
    dataframe: pd.DataFrame,
    sample_size: int,
    random_seed: int,
    split_name: str,
) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample-size must be greater than 0")

    if sample_size >= len(dataframe):
        return dataframe.copy(deep=True).reset_index(drop=True)

    return dataframe.sample(n=sample_size, random_state=random_seed).reset_index(
        drop=True
    )


if __name__ == "__main__":
    raise SystemExit(main())

