"""Deterministic multi-label dataset splitting utilities."""

from pathlib import Path
from typing import Sequence

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.data.analysis import analyze_label_distribution
from src.data.loader import EXPECTED_LABEL_COLUMNS, validate_required_columns


ID_COLUMN = "id"


def split_dataset(
    dataframe: pd.DataFrame,
    labels: Sequence[str],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test partitions."""
    validate_split_ratios(train_ratio, validation_ratio, test_ratio)
    validate_required_columns(dataframe, [ID_COLUMN, *labels])
    _validate_unique_ids(dataframe)

    temp_ratio = validation_ratio + test_ratio
    label_matrix = dataframe[list(labels)].eq(1).astype(int)

    first_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=temp_ratio,
        random_state=random_seed,
    )
    train_indices, temp_indices = next(first_splitter.split(dataframe, label_matrix))

    train = dataframe.iloc[train_indices].copy().reset_index(drop=True)
    temp = dataframe.iloc[temp_indices].copy().reset_index(drop=True)

    temp_label_matrix = temp[list(labels)].eq(1).astype(int)
    test_fraction_of_temp = test_ratio / temp_ratio
    second_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_fraction_of_temp,
        random_state=random_seed,
    )
    validation_indices, test_indices = next(
        second_splitter.split(temp, temp_label_matrix)
    )

    validation = temp.iloc[validation_indices].copy().reset_index(drop=True)
    test = temp.iloc[test_indices].copy().reset_index(drop=True)

    validate_partition_integrity(dataframe, train, validation, test)
    return train, validation, test


def validate_split_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    tolerance: float = 1e-6,
) -> None:
    """Validate positive split ratios that sum to 1.0."""
    ratios = (train_ratio, validation_ratio, test_ratio)

    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("Split ratios must be greater than 0")

    if abs(sum(ratios) - 1.0) > tolerance:
        raise ValueError(
            "Split ratios must sum to 1.0: "
            f"train={train_ratio}, validation={validation_ratio}, test={test_ratio}"
        )


def validate_partition_integrity(
    original: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    id_column: str = ID_COLUMN,
) -> None:
    """Validate completeness and ID separation across split partitions."""
    for name, partition in (
        ("original", original),
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        validate_required_columns(partition, [id_column])
        _validate_unique_ids(partition, name=name, id_column=id_column)

    if len(train) + len(validation) + len(test) != len(original):
        raise ValueError("Split partitions do not add up to the original row count")

    original_ids = set(original[id_column])
    train_ids = set(train[id_column])
    validation_ids = set(validation[id_column])
    test_ids = set(test[id_column])

    overlaps = {
        "train/validation": train_ids & validation_ids,
        "train/test": train_ids & test_ids,
        "validation/test": validation_ids & test_ids,
    }
    overlapping_pairs = [name for name, values in overlaps.items() if values]
    if overlapping_pairs:
        raise ValueError(
            "Split partitions contain overlapping IDs: "
            + ", ".join(overlapping_pairs)
        )

    combined_ids = train_ids | validation_ids | test_ids
    if combined_ids != original_ids:
        raise ValueError("Every original dataset ID must appear exactly once")


def compare_label_distributions(
    full: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    labels: Sequence[str] = EXPECTED_LABEL_COLUMNS,
) -> pd.DataFrame:
    """Compare per-label positive percentages across dataset partitions."""
    partitions = {
        "full": full,
        "train": train,
        "validation": validation,
        "test": test,
    }
    values: dict[str, dict[str, float]] = {}

    for partition_name, partition in partitions.items():
        distribution = analyze_label_distribution(partition, labels)
        values[partition_name] = {
            label: float(distribution[label]["positive_percentage"])
            for label in labels
        }

    return pd.DataFrame(values, index=list(labels))


def save_split_partitions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_directory: str | Path,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Save split partitions as CSV files without silently overwriting."""
    output_path = Path(output_directory)
    partition_paths = (
        output_path / "train.csv",
        output_path / "validation.csv",
        output_path / "test.csv",
    )
    existing_paths = [path for path in partition_paths if path.exists()]

    if existing_paths and not overwrite:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Processed split files already exist. Use --overwrite to replace: "
            f"{existing}"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    train.to_csv(partition_paths[0], index=False)
    validation.to_csv(partition_paths[1], index=False)
    test.to_csv(partition_paths[2], index=False)

    return partition_paths


def _validate_unique_ids(
    dataframe: pd.DataFrame,
    name: str = "dataset",
    id_column: str = ID_COLUMN,
) -> None:
    if dataframe[id_column].duplicated().any():
        raise ValueError(f"Duplicate dataset IDs found in {name}; cannot split safely")
