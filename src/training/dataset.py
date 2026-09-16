"""PyTorch Dataset for tokenized toxic comment rows."""

from collections.abc import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.model import build_label_targets


DEFAULT_TEXT_COLUMN = "comment_text"


class ToxicCommentDataset(Dataset):
    """Convert comment rows into XLM-R model-ready tensors."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        text_column: str,
        label_columns: Sequence[str],
        max_length: int,
        include_id: bool = False,
    ) -> None:
        self._validate_inputs(
            dataframe=dataframe,
            tokenizer=tokenizer,
            text_column=text_column,
            label_columns=label_columns,
            max_length=max_length,
        )
        self.dataframe = dataframe.copy(deep=True).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_columns = tuple(label_columns)
        self.max_length = max_length
        self.include_id = include_id

    def __len__(self) -> int:
        """Return number of rows in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one tokenized row and its floating-point label vector."""
        row = self.dataframe.iloc[index]
        encoded = self.tokenizer(
            row[self.text_column],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        item = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encoded["attention_mask"],
                dtype=torch.long,
            ),
            "labels": build_label_targets(row, self.label_columns),
        }

        if self.include_id and "id" in self.dataframe.columns:
            item["id"] = row["id"]

        return item

    @staticmethod
    def _validate_inputs(
        dataframe: pd.DataFrame,
        tokenizer,
        text_column: str,
        label_columns: Sequence[str],
        max_length: int,
    ) -> None:
        if tokenizer is None:
            raise TypeError("tokenizer is required")
        if dataframe.empty:
            raise ValueError("dataframe must not be empty")
        if text_column not in dataframe.columns:
            raise ValueError(f"Text column not found in DataFrame: {text_column}")
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError(f"max_length must be an int, got {type(max_length).__name__}")
        if max_length <= 0:
            raise ValueError("max_length must be greater than 0")

        missing_labels = [
            label for label in label_columns if label not in dataframe.columns
        ]
        if missing_labels:
            missing = ", ".join(missing_labels)
            raise ValueError(f"Missing required label columns: {missing}")

        for index, text in enumerate(dataframe[text_column].tolist()):
            if text is None or pd.isna(text):
                raise TypeError(f"{text_column}[{index}] must be a string, got missing")
            if not isinstance(text, str):
                raise TypeError(
                    f"{text_column}[{index}] must be a string, got {type(text).__name__}"
                )

