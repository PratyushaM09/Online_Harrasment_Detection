"""Reproducibility utilities."""

import os
import random


def set_seed(seed: int) -> None:
    """Seed standard-library randomness used by the project."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an int")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

