"""Multi-label logistic regression baseline helpers."""

from collections.abc import Sequence

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


DEFAULT_THRESHOLD = 0.5


def create_baseline_classifier(random_seed: int) -> OneVsRestClassifier:
    """Create a one-vs-rest logistic regression classifier."""
    base_classifier = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight=None,
        random_state=random_seed,
    )

    return OneVsRestClassifier(base_classifier)


def get_label_matrix(dataframe, labels: Sequence[str]):
    """Return the multi-label target matrix in configured label order."""
    missing_labels = [label for label in labels if label not in dataframe.columns]
    if missing_labels:
        missing = ", ".join(missing_labels)
        raise ValueError(f"Missing required label columns: {missing}")

    label_matrix = dataframe[list(labels)]
    invalid_labels = [
        label
        for label in labels
        if not label_matrix[label].isin([0, 1]).all()
    ]
    if invalid_labels:
        invalid = ", ".join(invalid_labels)
        raise ValueError(f"Label columns must contain only 0/1 values: {invalid}")

    return label_matrix.astype(int).to_numpy()


def fit_classifier(classifier, features, label_matrix):
    """Fit a multi-label classifier after validating feature/label shape."""
    _validate_feature_label_rows(features, label_matrix)

    return classifier.fit(features, label_matrix)


def predict_probabilities(classifier, features):
    """Predict one probability per label for each row."""
    return classifier.predict_proba(features)


def predict_labels(classifier, features, threshold: float = DEFAULT_THRESHOLD):
    """Predict binary labels using an explicit probability threshold."""
    _validate_threshold(threshold)
    probabilities = predict_probabilities(classifier, features)

    return (probabilities >= threshold).astype(int)


def _validate_feature_label_rows(features, label_matrix) -> None:
    feature_rows = features.shape[0]
    label_rows = label_matrix.shape[0]

    if feature_rows != label_rows:
        raise ValueError(
            "Feature row count must match label row count: "
            f"features={feature_rows}, labels={label_rows}"
        )


def _validate_threshold(threshold: float) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError(
            f"threshold must be a number between 0 and 1, got {type(threshold).__name__}"
        )
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1")

