from pathlib import Path

import pytest

from src.config import load_config


def test_valid_config_loads_successfully():
    config = load_config()

    assert config.project.name == "Deep Learning Model for Online Harassment Detection"
    assert config.project.random_seed == 42
    assert config.dataset.train_ratio == 0.80
    assert config.dataset.validation_ratio == 0.10
    assert config.dataset.test_ratio == 0.10
    assert config.model.name == "xlm-roberta-base"
    assert config.tokenization.max_length == 256
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


def test_invalid_split_ratios_raise_clear_error(tmp_path):
    invalid_config = tmp_path / "invalid_split_ratios.yaml"
    invalid_config.write_text(
        """
project:
  name: "Deep Learning Model for Online Harassment Detection"
  random_seed: 42
labels:
  - toxic
  - severe_toxic
  - obscene
  - threat
  - insult
  - identity_hate
dataset:
  train_ratio: 0.70
  validation_ratio: 0.20
  test_ratio: 0.20
model:
  name: "xlm-roberta-base"
tokenization:
  max_length: 256
paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  models: "models"
  checkpoints: "checkpoints"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Dataset split ratios must sum to 1.0"):
        load_config(invalid_config)


def test_invalid_tokenization_max_length_raises_clear_error(tmp_path):
    invalid_config = tmp_path / "invalid_tokenization.yaml"
    invalid_config.write_text(
        """
project:
  name: "Deep Learning Model for Online Harassment Detection"
  random_seed: 42
labels:
  - toxic
  - severe_toxic
  - obscene
  - threat
  - insult
  - identity_hate
dataset:
  train_ratio: 0.80
  validation_ratio: 0.10
  test_ratio: 0.10
model:
  name: "xlm-roberta-base"
tokenization:
  max_length: 0
paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  models: "models"
  checkpoints: "checkpoints"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Tokenization max_length"):
        load_config(invalid_config)
