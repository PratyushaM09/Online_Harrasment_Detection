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
    assert config.training.learning_rate == 0.00002
    assert config.training.weight_decay == 0.01
    assert config.training.epochs == 1
    assert config.training.batch_size == 8
    assert config.training.gradient_accumulation_steps == 1
    assert config.training.train_sample_size == 10000
    assert config.training.validation_sample_size == 2000
    assert config.training.default_threshold == 0.5
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
training:
  learning_rate: 0.00002
  weight_decay: 0.01
  epochs: 1
  batch_size: 8
  gradient_accumulation_steps: 1
  train_sample_size: 10000
  validation_sample_size: 2000
  default_threshold: 0.5
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
training:
  learning_rate: 0.00002
  weight_decay: 0.01
  epochs: 1
  batch_size: 8
  gradient_accumulation_steps: 1
  train_sample_size: 10000
  validation_sample_size: 2000
  default_threshold: 0.5
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


def test_invalid_training_batch_size_raises_clear_error(tmp_path):
    invalid_config = tmp_path / "invalid_training.yaml"
    invalid_config.write_text(
        _valid_config_text().replace("batch_size: 8", "batch_size: 0"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training.batch_size"):
        load_config(invalid_config)


def test_boolean_training_integer_is_rejected(tmp_path):
    invalid_config = tmp_path / "invalid_training_bool.yaml"
    invalid_config.write_text(
        _valid_config_text().replace("epochs: 1", "epochs: true"),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="training.epochs must be an int"):
        load_config(invalid_config)


def test_invalid_training_threshold_raises_clear_error(tmp_path):
    invalid_config = tmp_path / "invalid_training_threshold.yaml"
    invalid_config.write_text(
        _valid_config_text().replace("default_threshold: 0.5", "default_threshold: 1.5"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training.default_threshold"):
        load_config(invalid_config)


def _valid_config_text() -> str:
    return """
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
  max_length: 256
training:
  learning_rate: 0.00002
  weight_decay: 0.01
  epochs: 1
  batch_size: 8
  gradient_accumulation_steps: 1
  train_sample_size: 10000
  validation_sample_size: 2000
  default_threshold: 0.5
paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  models: "models"
  checkpoints: "checkpoints"
""".strip()
