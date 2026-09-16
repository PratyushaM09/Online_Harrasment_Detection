"""Analyze XLM-R tokenized sequence lengths for dataset splits."""

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data import validate_required_columns  # noqa: E402
from src.tokenization import (  # noqa: E402
    DEFAULT_TRUNCATION_LIMITS,
    analyze_truncation_impact,
    calculate_token_lengths,
    load_tokenizer,
    summarize_token_lengths,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze unpadded, untruncated XLM-R token lengths."
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "raw"),
        default="train",
        help="Dataset split to analyze. Defaults to the processed training split.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional deterministic sample size for quick local experiments.",
    )
    return parser.parse_args()


def main() -> int:
    """Load a split, tokenize without padding/truncation, and print length stats."""
    args = _parse_args()
    config = load_config()

    try:
        dataset = _load_split(args.split, config)
        validate_required_columns(dataset, ["comment_text"])
        if args.sample_size is not None:
            dataset = _sample_dataset(dataset, args.sample_size, config.project.random_seed)

        tokenizer = load_tokenizer(config.model.name)
        token_lengths = calculate_token_lengths(dataset["comment_text"].tolist(), tokenizer)
        summary = summarize_token_lengths(token_lengths)
        truncation = analyze_truncation_impact(
            token_lengths,
            candidate_limits=DEFAULT_TRUNCATION_LIMITS,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(error)
        return 1

    print("TOKEN LENGTH ANALYSIS")
    print(f"Split analyzed: {args.split}")
    print(f"Rows analyzed: {summary['count']}")
    if args.sample_size is not None:
        print(f"Sample size: {args.sample_size}")

    print("\nTOKEN LENGTH STATISTICS")
    print(f"Minimum: {summary['minimum']}")
    print(f"Mean: {summary['mean']:.2f}")
    print(f"Median: {summary['median']:.2f}")
    print(f"P90: {summary['p90']:.2f}")
    print(f"P95: {summary['p95']:.2f}")
    print(f"P99: {summary['p99']:.2f}")
    print(f"Maximum: {summary['maximum']}")

    print("\nTRUNCATION IMPACT")
    for result in truncation:
        print(f"max_length={result['max_length']}")
        print(f"comments truncated: {result['comments_exceeding_limit']}")
        print(f"percentage: {result['percentage_exceeding_limit']:.2f}%")
        print()

    return 0


def _load_split(split_name: str, config) -> pd.DataFrame:
    if split_name == "raw":
        path = PROJECT_ROOT / config.paths.raw_data / "train.csv"
    else:
        path = PROJECT_ROOT / config.paths.processed_data / f"{split_name}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    return pd.read_csv(path)


def _sample_dataset(
    dataframe: pd.DataFrame,
    sample_size: int,
    random_seed: int,
) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample-size must be greater than 0")
    if sample_size > len(dataframe):
        raise ValueError(
            f"sample-size {sample_size} exceeds available rows {len(dataframe)}"
        )

    return dataframe.sample(n=sample_size, random_state=random_seed).reset_index(
        drop=True
    )


if __name__ == "__main__":
    raise SystemExit(main())

