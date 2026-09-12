from dataclasses import replace

import pytest

from src.config import load_config
from src.data import get_training_data_path


def test_configured_missing_training_file_raises_clear_error():
    config = load_config()
    missing_paths = replace(
        config.paths,
        raw_data="tests/fixtures/missing_raw_data_for_path_test",
    )
    missing_config = replace(config, paths=missing_paths)

    with pytest.raises(FileNotFoundError, match="Training dataset not found"):
        get_training_data_path(missing_config)


def test_absolute_raw_data_path_is_rejected():
    config = load_config()
    absolute_paths = replace(config.paths, raw_data="C:/datasets/jigsaw")
    absolute_config = replace(config, paths=absolute_paths)

    with pytest.raises(ValueError, match="project-relative"):
        get_training_data_path(absolute_config)
