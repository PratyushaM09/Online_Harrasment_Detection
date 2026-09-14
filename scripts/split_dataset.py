"""Run and optionally persist reproducible multi-label dataset splits."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data import (  # noqa: E402
    compare_label_distributions,
    load_training_data,
    save_split_partitions,
    split_dataset,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic multi-label train/validation/test splits."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist splits to the configured processed data directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed split CSV files when used with --save.",
    )
    return parser.parse_args()


def main() -> int:
    """Split the configured dataset and print validation details."""
    args = _parse_args()
    config = load_config()

    try:
        dataframe = load_training_data()
        train, validation, test = split_dataset(
            dataframe=dataframe,
            labels=config.labels,
            train_ratio=config.dataset.train_ratio,
            validation_ratio=config.dataset.validation_ratio,
            test_ratio=config.dataset.test_ratio,
            random_seed=config.project.random_seed,
        )
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    print("SPLIT SUMMARY")
    print(f"Full rows: {len(dataframe)}")
    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Test rows: {len(test)}")

    print("\nLABEL DISTRIBUTION PERCENTAGES")
    distribution = compare_label_distributions(
        dataframe,
        train,
        validation,
        test,
        labels=config.labels,
    )
    print(distribution.to_string(float_format=lambda value: f"{value:.2f}"))

    if args.save:
        output_directory = PROJECT_ROOT / config.paths.processed_data
        try:
            paths = save_split_partitions(
                train,
                validation,
                test,
                output_directory,
                overwrite=args.overwrite,
            )
        except FileExistsError as error:
            print(error)
            return 1

        print("\nSAVED SPLITS")
        for path in paths:
            print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

