"""DataLoader factory helpers."""

from torch.utils.data import DataLoader


def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a standard PyTorch DataLoader with basic validation."""
    _validate_batch_size(batch_size)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_split_dataloaders(
    train_dataset,
    validation_dataset,
    test_dataset,
    batch_size: int,
) -> dict[str, DataLoader]:
    """Create train/validation/test DataLoaders with standard shuffle behavior."""
    return {
        "train": create_dataloader(train_dataset, batch_size=batch_size, shuffle=True),
        "validation": create_dataloader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
        "test": create_dataloader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def _validate_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be an int, got {type(batch_size).__name__}")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

