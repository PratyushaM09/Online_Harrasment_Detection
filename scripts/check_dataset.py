"""Validate the locally downloaded Jigsaw training dataset."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_dataset_metadata, load_training_data  # noqa: E402


def main() -> int:
    """Load the configured dataset and print basic metadata."""
    try:
        dataframe = load_training_data()
        metadata = get_dataset_metadata(dataframe)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    print("Jigsaw Toxic Comment Classification dataset is valid.")
    print(f"Rows: {metadata['number_of_rows']}")
    print(f"Columns: {metadata['number_of_columns']}")
    print(f"Labels: {', '.join(metadata['label_names'])}")
    print(f"Missing comment_text values: {metadata['missing_comment_text_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

