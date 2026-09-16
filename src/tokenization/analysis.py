"""Token length analysis helpers."""

from collections.abc import Sequence
from statistics import fmean, median


DEFAULT_TRUNCATION_LIMITS: tuple[int, ...] = (64, 128, 256, 512)


def calculate_token_lengths(
    texts: Sequence[str],
    tokenizer,
    batch_size: int = 1000,
) -> tuple[int, ...]:
    """Calculate unpadded, untruncated tokenizer lengths for each text."""
    _validate_texts(texts)
    _validate_batch_size(batch_size)
    text_list = list(texts)

    if not text_list:
        return ()

    token_lengths: list[int] = []

    for start_index in range(0, len(text_list), batch_size):
        batch = text_list[start_index : start_index + batch_size]
        encoded = tokenizer(
            batch,
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        token_lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])

    return tuple(token_lengths)


def summarize_token_lengths(token_lengths: Sequence[int]) -> dict[str, int | float]:
    """Summarize token lengths with common percentile statistics."""
    _validate_token_lengths(token_lengths)
    length_list = list(token_lengths)

    if not length_list:
        raise ValueError("token_lengths must not be empty")

    ordered_lengths = sorted(int(length) for length in length_list)

    return {
        "count": len(ordered_lengths),
        "minimum": ordered_lengths[0],
        "maximum": ordered_lengths[-1],
        "mean": float(fmean(ordered_lengths)),
        "median": float(median(ordered_lengths)),
        "p90": _percentile(ordered_lengths, 90),
        "p95": _percentile(ordered_lengths, 95),
        "p99": _percentile(ordered_lengths, 99),
    }


def analyze_truncation_impact(
    token_lengths: Sequence[int],
    candidate_limits: Sequence[int] = DEFAULT_TRUNCATION_LIMITS,
) -> list[dict[str, int | float]]:
    """Report how many tokenized comments exceed each candidate max length."""
    _validate_token_lengths(token_lengths)
    _validate_candidate_limits(candidate_limits)
    length_list = list(token_lengths)
    row_count = len(length_list)

    results = []
    for max_length in candidate_limits:
        exceeding_count = sum(length > max_length for length in length_list)
        percentage = (exceeding_count / row_count) * 100 if row_count else 0.0
        results.append(
            {
                "max_length": int(max_length),
                "comments_exceeding_limit": int(exceeding_count),
                "percentage_exceeding_limit": float(percentage),
            }
        )

    return results


def _percentile(sorted_values: Sequence[int], percentile: int) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * (percentile / 100)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index

    return float(
        sorted_values[lower_index] * (1 - weight)
        + sorted_values[upper_index] * weight
    )


def _validate_texts(texts: Sequence[str]) -> None:
    if texts is None:
        raise TypeError("texts must be a sequence of strings, got None")
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of strings, not a single string")

    for index, text in enumerate(texts):
        if text is None:
            raise TypeError(f"texts[{index}] must be a string, got None")
        if not isinstance(text, str):
            raise TypeError(
                f"texts[{index}] must be a string, got {type(text).__name__}"
            )


def _validate_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(
            f"batch_size must be an int greater than 0, got {type(batch_size).__name__}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")


def _validate_token_lengths(token_lengths: Sequence[int]) -> None:
    if token_lengths is None:
        raise TypeError("token_lengths must be a sequence of integers, got None")

    for index, length in enumerate(token_lengths):
        if not isinstance(length, int):
            raise TypeError(
                f"token_lengths[{index}] must be an int, got {type(length).__name__}"
            )
        if length < 0:
            raise ValueError(f"token_lengths[{index}] must be greater than or equal to 0")


def _validate_candidate_limits(candidate_limits: Sequence[int]) -> None:
    if candidate_limits is None:
        raise TypeError("candidate_limits must be a sequence of integers, got None")

    for index, limit in enumerate(candidate_limits):
        if not isinstance(limit, int):
            raise TypeError(
                f"candidate_limits[{index}] must be an int, got {type(limit).__name__}"
            )
        if limit <= 0:
            raise ValueError(f"candidate_limits[{index}] must be greater than 0")
