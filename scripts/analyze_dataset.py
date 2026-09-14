"""Print analysis statistics for the locally downloaded Jigsaw dataset."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import analyze_dataset, load_training_data  # noqa: E402


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{value:.2f}"


def main() -> int:
    """Load the configured training data and print analysis results."""
    try:
        dataframe = load_training_data()
        analysis = analyze_dataset(dataframe)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    comment_summary = analysis["comment_summary"]
    text_quality = analysis["text_quality"]

    print("DATASET SUMMARY")
    print(f"Rows: {comment_summary['number_of_rows']}")
    print(f"Columns: {len(dataframe.columns)}")

    print("\nLABEL DISTRIBUTION")
    for label, stats in analysis["label_distribution"].items():
        print(f"{label}:")
        print(f"  positive: {stats['positive_count']}")
        print(f"  negative: {stats['negative_count']}")
        print(f"  percentage: {stats['positive_percentage']:.2f}%")
        print(f"  imbalance ratio: {_format_ratio(stats['imbalance_ratio'])}")

    print("\nCOMMENT SUMMARY")
    print(f"rows with toxicity: {comment_summary['rows_with_toxicity']}")
    print(f"rows without toxicity: {comment_summary['rows_without_toxicity']}")
    print(
        "percentage with toxicity: "
        f"{comment_summary['percentage_with_toxicity']:.2f}%"
    )
    print(
        "average labels per row: "
        f"{comment_summary['average_positive_labels_per_row']:.4f}"
    )
    print(
        "maximum simultaneous labels: "
        f"{comment_summary['maximum_simultaneous_positive_labels']}"
    )

    print("\nTEXT QUALITY")
    print(f"missing comments: {text_quality['missing_comment_text_count']}")
    print(f"blank comments: {text_quality['blank_comment_text_count']}")

    print("\nLABEL CO-OCCURRENCE")
    print(analysis["label_cooccurrence"].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

