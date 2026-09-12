import random

from src.utils import set_seed


def test_set_seed_repeats_standard_library_random_sequence():
    set_seed(42)
    first_sequence = [random.random() for _ in range(5)]

    set_seed(42)
    second_sequence = [random.random() for _ in range(5)]

    assert first_sequence == second_sequence
