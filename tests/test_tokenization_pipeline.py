import pandas as pd
import pytest

from src.data import EXPECTED_LABEL_COLUMNS
from src.tokenization import tokenize_dataframe


class PipelineFakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, truncation, padding, max_length):
        self.calls.append(
            {
                "texts": texts,
                "truncation": truncation,
                "padding": padding,
                "max_length": max_length,
            }
        )
        return {
            "input_ids": [self._ids(text, max_length) for text in texts],
            "attention_mask": [self._mask(text, max_length) for text in texts],
        }

    def _ids(self, text, max_length):
        ids = [len(token) for token in text.split()]
        return (ids + [0] * max_length)[:max_length]

    def _mask(self, text, max_length):
        count = min(len(text.split()), max_length)
        return [1] * count + [0] * (max_length - count)


def _pipeline_dataframe() -> pd.DataFrame:
    dataframe = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "comment_text": ["first text", "second sample", "third row"],
            "normalized_text": ["first text", "second sample", "third row"],
        }
    )

    for label in EXPECTED_LABEL_COLUMNS:
        dataframe[label] = [0, 1, 0]

    return dataframe


def test_pipeline_returns_input_ids_and_attention_mask():
    tokenized = tokenize_dataframe(
        _pipeline_dataframe(),
        PipelineFakeTokenizer(),
        max_length=4,
    )

    assert "input_ids" in tokenized.columns
    assert "attention_mask" in tokenized.columns


def test_pipeline_preserves_ids_and_six_labels_when_present():
    dataframe = _pipeline_dataframe()
    tokenized = tokenize_dataframe(dataframe, PipelineFakeTokenizer(), max_length=4)

    assert tokenized["id"].tolist() == dataframe["id"].tolist()
    for label in EXPECTED_LABEL_COLUMNS:
        assert tokenized[label].tolist() == dataframe[label].tolist()


def test_pipeline_does_not_mutate_input_dataframe():
    dataframe = _pipeline_dataframe()
    original = dataframe.copy(deep=True)

    tokenize_dataframe(dataframe, PipelineFakeTokenizer(), max_length=4)

    pd.testing.assert_frame_equal(dataframe, original)


def test_pipeline_forwards_max_length_and_padding_options():
    tokenizer = PipelineFakeTokenizer()

    tokenize_dataframe(_pipeline_dataframe(), tokenizer, max_length=6)

    assert tokenizer.calls[0]["max_length"] == 6
    assert tokenizer.calls[0]["truncation"] is True
    assert tokenizer.calls[0]["padding"] == "max_length"


def test_pipeline_preserves_row_order():
    dataframe = _pipeline_dataframe()

    tokenized = tokenize_dataframe(dataframe, PipelineFakeTokenizer(), max_length=4)

    assert tokenized["id"].tolist() == ["a", "b", "c"]


def test_configurable_text_column_works():
    dataframe = _pipeline_dataframe()
    dataframe.loc[0, "normalized_text"] = "custom source text"

    tokenized = tokenize_dataframe(
        dataframe,
        PipelineFakeTokenizer(),
        text_column="normalized_text",
        max_length=4,
    )

    assert tokenized.loc[0, "input_ids"][:3] == [
        len("custom"),
        len("source"),
        len("text"),
    ]


def test_missing_text_column_raises_clear_error():
    with pytest.raises(ValueError, match="Text column not found"):
        tokenize_dataframe(
            _pipeline_dataframe(),
            PipelineFakeTokenizer(),
            text_column="missing_text",
            max_length=4,
        )


def test_batch_behavior_preserves_row_alignment():
    tokenizer = PipelineFakeTokenizer()
    dataframe = _pipeline_dataframe()

    tokenized = tokenize_dataframe(
        dataframe,
        tokenizer,
        max_length=4,
        batch_size=2,
    )

    assert len(tokenizer.calls) == 2
    assert tokenized["id"].tolist() == dataframe["id"].tolist()
    assert len(tokenized) == len(dataframe)


def test_non_string_text_value_is_rejected():
    dataframe = _pipeline_dataframe()
    dataframe.loc[1, "comment_text"] = None

    with pytest.raises(TypeError, match=r"comment_text\[1\] must be a string"):
        tokenize_dataframe(dataframe, PipelineFakeTokenizer(), max_length=4)

