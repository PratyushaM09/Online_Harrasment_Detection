from pathlib import Path

import pytest

from src.config import load_config


def test_valid_config_loads_successfully():
    config = load_config()

    assert config.project.name == "Deep Learning Model for Online Harassment Detection"
    assert config.project.random_seed == 42
    assert config.labels == (
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    )


def test_missing_config_raises_clear_error(tmp_path):
    missing_config = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(missing_config)


def test_invalid_yaml_raises_clear_error(tmp_path):
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text("project: [broken", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid YAML configuration"):
        load_config(Path(invalid_config))

