import pytest

from src.tokenization import (
    analyze_truncation_impact,
    calculate_token_lengths,
    summarize_token_lengths,
)


class LengthFakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, padding, truncation, add_special_tokens):
        self.calls.append(
            {
                "texts": texts,
                "padding": padding,
                "truncation": truncation,
                "add_special_tokens": add_special_tokens,
            }
        )
        return {
            "input_ids": [
                list(range(len(text.split()) + 2)) if text else [0, 1]
                for text in texts
            ]
        }


def test_calculate_token_lengths_returns_one_length_per_text():
    tokenizer = LengthFakeTokenizer()
    texts = ["one", "two words", ""]

    lengths = calculate_token_lengths(texts, tokenizer)

    assert lengths == (3, 4, 2)
    assert len(lengths) == len(texts)


def test_length_analysis_preserves_input_order():
    tokenizer = LengthFakeTokenizer()

    lengths = calculate_token_lengths(["a", "a b c", "a b"], tokenizer)

    assert lengths == (3, 5, 4)


def test_length_analysis_requests_no_padding_or_truncation():
    tokenizer = LengthFakeTokenizer()

    calculate_token_lengths(["hello world"], tokenizer)

    assert tokenizer.calls[0]["padding"] is False
    assert tokenizer.calls[0]["truncation"] is False
    assert tokenizer.calls[0]["add_special_tokens"] is True


def test_summary_statistics_are_calculated_correctly():
    summary = summarize_token_lengths([10, 20, 30, 40, 50])

    assert summary["count"] == 5
    assert summary["minimum"] == 10
    assert summary["maximum"] == 50
    assert summary["mean"] == pytest.approx(30.0)
    assert summary["median"] == 30.0


def test_percentiles_are_sensible():
    summary = summarize_token_lengths([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    assert summary["p90"] == pytest.approx(9.1)
    assert summary["p95"] == pytest.approx(9.55)
    assert summary["p99"] == pytest.approx(9.91)


def test_truncation_percentages_are_calculated_correctly():
    results = analyze_truncation_impact([10, 20, 70, 130], candidate_limits=[64, 128])

    assert results == [
        {
            "max_length": 64,
            "comments_exceeding_limit": 2,
            "percentage_exceeding_limit": 50.0,
        },
        {
            "max_length": 128,
            "comments_exceeding_limit": 1,
            "percentage_exceeding_limit": 25.0,
        },
    ]


def test_empty_text_input_returns_empty_lengths():
    assert calculate_token_lengths([], LengthFakeTokenizer()) == ()


def test_empty_lengths_summary_fails_clearly():
    with pytest.raises(ValueError, match="token_lengths must not be empty"):
        summarize_token_lengths([])


def test_invalid_text_is_rejected():
    with pytest.raises(TypeError, match=r"texts\[1\] must be a string"):
        calculate_token_lengths(["valid", None], LengthFakeTokenizer())


def test_input_texts_are_not_mutated():
    tokenizer = LengthFakeTokenizer()
    texts = ["one", "two words"]
    original = list(texts)

    calculate_token_lengths(texts, tokenizer)

    assert texts == original


def test_batching_preserves_alignment():
    tokenizer = LengthFakeTokenizer()

    lengths = calculate_token_lengths(["a", "a b", "a b c"], tokenizer, batch_size=2)

    assert lengths == (3, 4, 5)
    assert len(tokenizer.calls) == 2


@pytest.mark.parametrize("invalid_batch_size", [1.5, "10", True])
def test_invalid_batch_size_type_is_rejected(invalid_batch_size):
    with pytest.raises(TypeError, match="batch_size must be an int"):
        calculate_token_lengths(
            ["hello"],
            LengthFakeTokenizer(),
            batch_size=invalid_batch_size,
        )


@pytest.mark.parametrize("invalid_batch_size", [0, -1])
def test_invalid_batch_size_value_is_rejected(invalid_batch_size):
    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        calculate_token_lengths(
            ["hello"],
            LengthFakeTokenizer(),
            batch_size=invalid_batch_size,
        )
