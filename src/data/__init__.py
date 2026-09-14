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
from src.data.splitter import (
    compare_label_distributions,
    save_split_partitions,
    split_dataset,
    validate_partition_integrity,
    validate_split_ratios,
)

__all__ = [
    "EXPECTED_COLUMNS",
    "EXPECTED_LABEL_COLUMNS",
    "TRAIN_FILENAME",
    "analyze_dataset",
    "analyze_label_distribution",
    "analyze_text_quality",
    "calculate_label_cooccurrence",
    "compare_label_distributions",
    "get_dataset_metadata",
    "get_training_data_path",
    "load_training_data",
    "save_split_partitions",
    "split_dataset",
    "summarize_comment_labels",
    "validate_partition_integrity",
    "validate_required_columns",
    "validate_split_ratios",
]
