import pandas as pd
import pytest
import torch

from src.data import EXPECTED_LABEL_COLUMNS
from src.training import ToxicCommentDataset, create_dataloader, create_split_dataloaders


class FakeTokenizer:
    def __call__(self, text, truncation, padding, max_length):
        ids = [len(token) for token in text.split()]
        return {
            "input_ids": (ids + [0] * max_length)[:max_length],
            "attention_mask": [1 if index < len(ids) else 0 for index in range(max_length)],
        }


def _frame(row_count=5):
    frame = pd.DataFrame(
        {
            "id": [f"id-{index}" for index in range(row_count)],
            "comment_text": [f"row {index}" for index in range(row_count)],
        }
    )
    for label in EXPECTED_LABEL_COLUMNS:
        frame[label] = [index % 2 for index in range(row_count)]
    return frame


def _dataset(row_count=5):
    return ToxicCommentDataset(
        dataframe=_frame(row_count=row_count),
        tokenizer=FakeTokenizer(),
        text_column="comment_text",
        label_columns=EXPECTED_LABEL_COLUMNS,
        max_length=6,
    )


def test_dataloader_batch_shape_is_correct():
    dataloader = create_dataloader(_dataset(), batch_size=2, shuffle=False)
    batch = next(iter(dataloader))

    assert batch["input_ids"].shape == torch.Size([2, 6])
    assert batch["attention_mask"].shape == torch.Size([2, 6])
    assert batch["labels"].shape == torch.Size([2, 6])


@pytest.mark.parametrize("invalid_batch_size", [0, -1])
def test_batch_size_value_validation(invalid_batch_size):
    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        create_dataloader(_dataset(), batch_size=invalid_batch_size, shuffle=False)


@pytest.mark.parametrize("invalid_batch_size", [True, 1.5, "2"])
def test_batch_size_type_validation(invalid_batch_size):
    with pytest.raises(TypeError, match="batch_size must be an int"):
        create_dataloader(_dataset(), batch_size=invalid_batch_size, shuffle=False)


def test_train_validation_test_shuffle_settings_are_created():
    loaders = create_split_dataloaders(
        _dataset(),
        _dataset(),
        _dataset(),
        batch_size=2,
    )

    assert set(loaders) == {"train", "validation", "test"}


def test_batch_labels_align_with_inputs_when_not_shuffled():
    dataset = _dataset(row_count=3)
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))

    assert batch["labels"][0].tolist() == dataset[0]["labels"].tolist()
    assert batch["labels"][1].tolist() == dataset[1]["labels"].tolist()


def test_final_partial_batch_works_correctly():
    dataloader = create_dataloader(_dataset(row_count=5), batch_size=2, shuffle=False)
    batches = list(dataloader)

    assert batches[-1]["input_ids"].shape[0] == 1

