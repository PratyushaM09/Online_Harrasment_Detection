"""Device selection helpers."""

import torch


def select_device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

