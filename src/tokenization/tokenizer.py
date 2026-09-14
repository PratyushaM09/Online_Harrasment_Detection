"""Small Hugging Face tokenizer wrapper for XLM-R experiments."""

from collections.abc import Sequence
from typing import Any


def load_tokenizer(model_name: str):
    """Load a Hugging Face tokenizer by model name."""
    if model_name is None:
        raise TypeError("model_name must be a string, got None")
    if not isinstance(model_name, str):
        raise TypeError(f"model_name must be a string, got {type(model_name).__name__}")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def tokenize_text(
    text: str,
    tokenizer,
    max_length: int,
) -> dict[str, Any]:
    """Tokenize one text with deterministic padding and truncation behavior."""
    _validate_text(text)
    _validate_max_length(max_length)

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def tokenize_batch(
    texts: Sequence[str],
    tokenizer,
    max_length: int,
) -> dict[str, Any]:
    """Tokenize a batch of texts while preserving input order."""
    if texts is None:
        raise TypeError("texts must be a sequence of strings, got None")
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of strings, not a single string")
    for text in texts:
        _validate_text(text)
    _validate_max_length(max_length)

    encoded = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def inspect_tokens(text: str, tokenizer) -> dict[str, list[int] | list[str]]:
    """Return readable token strings and token IDs for learning/debugging."""
    _validate_text(text)

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return {
        "tokens": tokens,
        "token_ids": token_ids,
    }


def _validate_text(text: str) -> None:
    if text is None:
        raise TypeError("text must be a string, got None")
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")


def _validate_max_length(max_length: int) -> None:
    if not isinstance(max_length, int):
        raise TypeError(f"max_length must be an int, got {type(max_length).__name__}")
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0")

