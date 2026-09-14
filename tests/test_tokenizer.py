import pytest

from src.tokenization import inspect_tokens, tokenize_batch, tokenize_text


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text_or_texts, truncation, padding, max_length):
        self.calls.append(
            {
                "text_or_texts": text_or_texts,
                "truncation": truncation,
                "padding": padding,
                "max_length": max_length,
            }
        )

        if isinstance(text_or_texts, list):
            return {
                "input_ids": [
                    self._ids_for_text(text, max_length) for text in text_or_texts
                ],
                "attention_mask": [
                    self._mask_for_text(text, max_length) for text in text_or_texts
                ],
            }

        return {
            "input_ids": self._ids_for_text(text_or_texts, max_length),
            "attention_mask": self._mask_for_text(text_or_texts, max_length),
        }

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return [len(token) for token in tokens]

    def _ids_for_text(self, text, max_length):
        ids = [len(token) for token in text.split()]
        return (ids + [0] * max_length)[:max_length]

    def _mask_for_text(self, text, max_length):
        token_count = min(len(text.split()), max_length)
        return [1] * token_count + [0] * (max_length - token_count)


def test_tokenize_text_rejects_none():
    with pytest.raises(TypeError, match="text must be a string"):
        tokenize_text(None, FakeTokenizer(), max_length=8)


def test_tokenize_text_rejects_non_string_values():
    with pytest.raises(TypeError, match="text must be a string"):
        tokenize_text(123, FakeTokenizer(), max_length=8)


def test_empty_string_is_allowed():
    encoded = tokenize_text("", FakeTokenizer(), max_length=4)

    assert encoded["input_ids"] == [0, 0, 0, 0]
    assert encoded["attention_mask"] == [0, 0, 0, 0]


def test_max_length_and_tokenizer_options_are_forwarded():
    tokenizer = FakeTokenizer()

    encoded = tokenize_text("hello world", tokenizer, max_length=6)

    assert set(encoded) == {"input_ids", "attention_mask"}
    assert tokenizer.calls[0]["max_length"] == 6
    assert tokenizer.calls[0]["truncation"] is True
    assert tokenizer.calls[0]["padding"] == "max_length"


def test_invalid_max_length_is_rejected():
    with pytest.raises(ValueError, match="max_length must be greater than 0"):
        tokenize_text("hello", FakeTokenizer(), max_length=0)


def test_tokenize_batch_preserves_order():
    tokenizer = FakeTokenizer()
    texts = ["short", "two words", "three small words"]

    encoded = tokenize_batch(texts, tokenizer, max_length=5)

    assert tokenizer.calls[0]["text_or_texts"] == texts
    assert encoded["input_ids"][0][0] == len("short")
    assert encoded["input_ids"][1][:2] == [len("two"), len("words")]
    assert encoded["input_ids"][2][:3] == [5, 5, 5]


def test_tokenize_batch_rejects_single_string():
    with pytest.raises(TypeError, match="not a single string"):
        tokenize_batch("hello", FakeTokenizer(), max_length=5)


def test_tokenize_batch_rejects_non_string_member():
    with pytest.raises(TypeError, match="text must be a string"):
        tokenize_batch(["hello", 123], FakeTokenizer(), max_length=5)


def test_inspect_tokens_returns_readable_tokens_and_ids():
    result = inspect_tokens("hello world", FakeTokenizer())

    assert result == {
        "tokens": ["hello", "world"],
        "token_ids": [5, 5],
    }
