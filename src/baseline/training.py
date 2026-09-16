"""Training and persistence orchestration for the classical baseline."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import joblib

from src.baseline.model import (
    DEFAULT_THRESHOLD,
    create_baseline_classifier,
    fit_classifier,
    get_label_matrix,
)
from src.baseline.vectorizer import (
    DEFAULT_TEXT_COLUMN,
    fit_transform_training_text,
)


@dataclass(frozen=True)
class BaselineTrainingResult:
    """Fitted baseline artifacts and metadata."""

    vectorizer: Any
    classifier: Any
    labels: tuple[str, ...]
    default_threshold: float = DEFAULT_THRESHOLD


def train_baseline(
    train_dataframe,
    labels: Sequence[str],
    random_seed: int,
    text_column: str = DEFAULT_TEXT_COLUMN,
) -> BaselineTrainingResult:
    """Train the TF-IDF + one-vs-rest logistic regression baseline."""
    label_order = tuple(labels)
    vectorizer, train_features = fit_transform_training_text(
        train_dataframe,
        text_column=text_column,
    )
    label_matrix = get_label_matrix(train_dataframe, label_order)
    classifier = create_baseline_classifier(random_seed=random_seed)
    fit_classifier(classifier, train_features, label_matrix)

    return BaselineTrainingResult(
        vectorizer=vectorizer,
        classifier=classifier,
        labels=label_order,
    )


def save_baseline_artifacts(
    result: BaselineTrainingResult,
    output_directory: str | Path,
    random_seed: int,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Save fitted baseline artifacts without silently overwriting."""
    output_path = Path(output_directory)
    vectorizer_path = output_path / "vectorizer.joblib"
    classifier_path = output_path / "classifier.joblib"
    metadata_path = output_path / "metadata.json"
    artifact_paths = (vectorizer_path, classifier_path, metadata_path)
    existing_paths = [path for path in artifact_paths if path.exists()]

    if existing_paths and not overwrite:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Baseline artifacts already exist. Use --overwrite to replace: "
            f"{existing}"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.vectorizer, vectorizer_path)
    joblib.dump(result.classifier, classifier_path)
    metadata = {
        "labels": list(result.labels),
        "random_seed": random_seed,
        "default_threshold": result.default_threshold,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return artifact_paths


def load_baseline_artifacts(
    artifact_directory: str | Path,
) -> BaselineTrainingResult:
    """Load saved baseline artifacts from disk."""
    artifact_path = Path(artifact_directory)
    vectorizer = joblib.load(artifact_path / "vectorizer.joblib")
    classifier = joblib.load(artifact_path / "classifier.joblib")
    metadata = json.loads(
        (artifact_path / "metadata.json").read_text(encoding="utf-8")
    )

    return BaselineTrainingResult(
        vectorizer=vectorizer,
        classifier=classifier,
        labels=tuple(metadata["labels"]),
        default_threshold=float(metadata["default_threshold"]),
    )

