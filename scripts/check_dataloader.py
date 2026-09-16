"""Inspect one PyTorch DataLoader batch for the XLM-R data pipeline."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.tokenization import load_tokenizer  # noqa: E402
from src.training import ToxicCommentDataset, create_dataloader  # noqa: E402


def main() -> int:
    """Load a few rows and print one DataLoader batch shape summary."""
    config = load_config()
    train_path = PROJECT_ROOT / config.paths.processed_data / "train.csv"
    if not train_path.exists():
        print(f"Processed training split not found: {train_path}")
        return 1

    dataframe = pd.read_csv(train_path, nrows=8)
    tokenizer = load_tokenizer(config.model.name)
    dataset = ToxicCommentDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        text_column="comment_text",
        label_columns=config.labels,
        max_length=config.tokenization.max_length,
    )
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(dataloader))

    print(f"batch size: {batch['input_ids'].shape[0]}")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    print(f"label dtype: {batch['labels'].dtype}")
    print(f"first label vector: {batch['labels'][0].tolist()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

