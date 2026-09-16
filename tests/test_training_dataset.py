import pandas as pd
import pytest
import torch

from src.data import EXPECTED_LABEL_COLUMNS
from src.training import ToxicCommentDataset


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, truncation, padding, max_length):
        self.calls.append(
            {
                "text": text,
                "truncation": truncation,
                "padding": padding,
                "max_length": max_length,
            }
        )
        token_values = [len(token) for token in text.split()]
        input_ids = (token_values + [0] * max_length)[:max_length]
        attention_mask = [1 if index < len(token_values) else 0 for index in range(max_length)]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _dataset_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "comment_text": ["first text", "second sample", "third row"],
        }
    )
    for label_index, label in enumerate(EXPECTED_LABEL_COLUMNS):
        frame[label] = [int((row + label_index) % 2 == 0) for row in range(len(frame))]

    return frame


_DEFAULT_TOKENIZER = object()


def _dataset(frame=None, tokenizer=_DEFAULT_TOKENIZER, max_length=8):
    return ToxicCommentDataset(
        dataframe=frame if frame is not None else _dataset_frame(),
        tokenizer=FakeTokenizer() if tokenizer is _DEFAULT_TOKENIZER else tokenizer,
        text_column="comment_text",
        label_columns=EXPECTED_LABEL_COLUMNS,
        max_length=max_length,
    )


def test_dataset_length_and_row_order_are_preserved():
    tokenizer = FakeTokenizer()
    dataset = _dataset(tokenizer=tokenizer)

    assert len(dataset) == 3
    _ = dataset[1]
    assert tokenizer.calls[0]["text"] == "second sample"


def test_dataset_returns_expected_tensors():
    item = _dataset()[0]

    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    assert item["input_ids"].shape == torch.Size([8])
    assert item["attention_mask"].shape == torch.Size([8])
    assert item["labels"].shape == torch.Size([6])
    assert item["labels"].dtype == torch.float32


def test_label_order_matches_configured_labels():
    frame = _dataset_frame()
    dataset = _dataset(frame=frame)

    assert dataset[0]["labels"].tolist() == [
        float(frame.loc[0, label]) for label in EXPECTED_LABEL_COLUMNS
    ]


def test_tokenizer_receives_max_length_truncation_and_padding():
    tokenizer = FakeTokenizer()
    dataset = _dataset(tokenizer=tokenizer, max_length=5)

    _ = dataset[0]

    assert tokenizer.calls[0]["max_length"] == 5
    assert tokenizer.calls[0]["truncation"] is True
    assert tokenizer.calls[0]["padding"] == "max_length"


def test_missing_text_column_is_rejected():
    frame = _dataset_frame().drop(columns=["comment_text"])

    with pytest.raises(ValueError, match="Text column not found"):
        _dataset(frame=frame)


def test_missing_label_column_is_rejected():
    frame = _dataset_frame().drop(columns=["toxic"])

    with pytest.raises(ValueError, match="Missing required label columns"):
        _dataset(frame=frame)


def test_missing_text_value_is_rejected():
    frame = _dataset_frame()
    frame.loc[1, "comment_text"] = None

    with pytest.raises(TypeError, match=r"comment_text\[1\] must be a string"):
        _dataset(frame=frame)


def test_source_dataframe_is_not_mutated():
    frame = _dataset_frame()
    original = frame.copy(deep=True)

    dataset = _dataset(frame=frame)
    _ = dataset[0]

    pd.testing.assert_frame_equal(frame, original)


def test_invalid_max_length_is_rejected():
    with pytest.raises(ValueError, match="max_length must be greater than 0"):
        _dataset(max_length=0)


def test_missing_tokenizer_is_rejected():
    with pytest.raises(TypeError, match="tokenizer is required"):
        _dataset(tokenizer=None)
