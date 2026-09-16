"""Classical baseline helpers."""

from src.baseline.model import (
    create_baseline_classifier,
    get_label_matrix,
    predict_labels,
    predict_probabilities,
)
from src.baseline.training import (
    BaselineTrainingResult,
    load_baseline_artifacts,
    save_baseline_artifacts,
    train_baseline,
)
from src.baseline.vectorizer import (
    create_tfidf_vectorizer,
    fit_transform_training_text,
    transform_text,
)

__all__ = [
    "BaselineTrainingResult",
    "create_baseline_classifier",
    "create_tfidf_vectorizer",
    "fit_transform_training_text",
    "get_label_matrix",
    "load_baseline_artifacts",
    "predict_labels",
    "predict_probabilities",
    "save_baseline_artifacts",
    "train_baseline",
    "transform_text",
]

