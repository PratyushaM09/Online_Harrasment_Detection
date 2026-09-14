"""Dataset loading and validation helpers."""

from src.data.dataset_paths import TRAIN_FILENAME, get_training_data_path
from src.data.loader import (
    EXPECTED_COLUMNS,
    EXPECTED_LABEL_COLUMNS,
    get_dataset_metadata,
    load_training_data,
    validate_required_columns,
)
from src.data.analysis import (
    analyze_dataset,
    analyze_label_distribution,
    analyze_text_quality,
    calculate_label_cooccurrence,
    summarize_comment_labels,
)

__all__ = [
    "EXPECTED_COLUMNS",
    "EXPECTED_LABEL_COLUMNS",
    "TRAIN_FILENAME",
    "analyze_dataset",
    "analyze_label_distribution",
    "analyze_text_quality",
    "calculate_label_cooccurrence",
    "get_dataset_metadata",
    "get_training_data_path",
    "load_training_data",
    "summarize_comment_labels",
    "validate_required_columns",
]
