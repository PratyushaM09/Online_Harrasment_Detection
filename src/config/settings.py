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
class AppConfig:
    """Structured application configuration."""

    project: ProjectConfig
    labels: tuple[str, ...]
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
    paths = raw_config.get("paths")

    if not isinstance(project, dict):
        raise ValueError(f"Missing or invalid 'project' section in {path}")
    if not isinstance(labels, list):
        raise ValueError(f"Missing or invalid 'labels' section in {path}")
    if not isinstance(paths, dict):
        raise ValueError(f"Missing or invalid 'paths' section in {path}")

    try:
        return AppConfig(
            project=ProjectConfig(
                name=str(project["name"]),
                random_seed=int(project["random_seed"]),
            ),
            labels=tuple(str(label) for label in labels),
            paths=PathsConfig(
                raw_data=str(paths["raw_data"]),
                processed_data=str(paths["processed_data"]),
                models=str(paths["models"]),
                checkpoints=str(paths["checkpoints"]),
            ),
        )
    except KeyError as error:
        raise ValueError(f"Missing required configuration key in {path}: {error}") from error

