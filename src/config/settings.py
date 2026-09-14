"""Project configuration loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@dataclass(frozen=True)
class ProjectConfig:
    """Project-level configuration values."""

    name: str
    random_seed: int


@dataclass(frozen=True)
class PathsConfig:
    """Project path configuration values."""

    raw_data: str
    processed_data: str
    models: str
    checkpoints: str


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset split configuration values."""

    train_ratio: float
    validation_ratio: float
    test_ratio: float


@dataclass(frozen=True)
class ModelConfig:
    """Transformer model configuration values."""

    name: str


@dataclass(frozen=True)
class TokenizationConfig:
    """Tokenizer configuration values."""

    max_length: int


@dataclass(frozen=True)
class AppConfig:
    """Structured application configuration."""

    project: ProjectConfig
    labels: tuple[str, ...]
    dataset: DatasetConfig
    model: ModelConfig
    tokenization: TokenizationConfig
    paths: PathsConfig


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load project configuration from a YAML file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML configuration in {path}") from error

    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")

    return _parse_config(raw_config, path)


def _parse_config(raw_config: dict[str, Any], path: Path) -> AppConfig:
    project = raw_config.get("project")
    labels = raw_config.get("labels")
    dataset = raw_config.get("dataset")
    model = raw_config.get("model")
    tokenization = raw_config.get("tokenization")
    paths = raw_config.get("paths")

    if not isinstance(project, dict):
        raise ValueError(f"Missing or invalid 'project' section in {path}")
    if not isinstance(labels, list):
        raise ValueError(f"Missing or invalid 'labels' section in {path}")
    if not isinstance(dataset, dict):
        raise ValueError(f"Missing or invalid 'dataset' section in {path}")
    if not isinstance(model, dict):
        raise ValueError(f"Missing or invalid 'model' section in {path}")
    if not isinstance(tokenization, dict):
        raise ValueError(f"Missing or invalid 'tokenization' section in {path}")
    if not isinstance(paths, dict):
        raise ValueError(f"Missing or invalid 'paths' section in {path}")

    try:
        dataset_config = DatasetConfig(
            train_ratio=float(dataset["train_ratio"]),
            validation_ratio=float(dataset["validation_ratio"]),
            test_ratio=float(dataset["test_ratio"]),
        )
        _validate_split_ratios(dataset_config, path)
        tokenization_config = TokenizationConfig(
            max_length=int(tokenization["max_length"]),
        )
        _validate_tokenization_config(tokenization_config, path)

        return AppConfig(
            project=ProjectConfig(
                name=str(project["name"]),
                random_seed=int(project["random_seed"]),
            ),
            labels=tuple(str(label) for label in labels),
            dataset=dataset_config,
            model=ModelConfig(name=str(model["name"])),
            tokenization=tokenization_config,
            paths=PathsConfig(
                raw_data=str(paths["raw_data"]),
                processed_data=str(paths["processed_data"]),
                models=str(paths["models"]),
                checkpoints=str(paths["checkpoints"]),
            ),
        )
    except KeyError as error:
        raise ValueError(f"Missing required configuration key in {path}: {error}") from error


def _validate_split_ratios(dataset_config: DatasetConfig, path: Path) -> None:
    ratios = (
        dataset_config.train_ratio,
        dataset_config.validation_ratio,
        dataset_config.test_ratio,
    )

    if any(ratio <= 0 for ratio in ratios):
        raise ValueError(f"Dataset split ratios must be greater than 0 in {path}")

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(
            "Dataset split ratios must sum to 1.0: "
            f"train={dataset_config.train_ratio}, "
            f"validation={dataset_config.validation_ratio}, "
            f"test={dataset_config.test_ratio}"
        )


def _validate_tokenization_config(
    tokenization_config: TokenizationConfig,
    path: Path,
) -> None:
    if tokenization_config.max_length <= 0:
        raise ValueError(f"Tokenization max_length must be greater than 0 in {path}")
