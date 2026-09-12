"""Dataset path resolution helpers."""

from pathlib import Path

from src.config import AppConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILENAME = "train.csv"


def get_training_data_path(config: AppConfig | None = None) -> Path:
    """Return the configured path to the Jigsaw training CSV."""
    app_config = config or load_config()
    raw_data_path = Path(app_config.paths.raw_data)

    if raw_data_path.is_absolute():
        raise ValueError(
            "Configured raw data path must be project-relative, "
            f"got absolute path: {raw_data_path}"
        )

    train_path = (PROJECT_ROOT / raw_data_path / TRAIN_FILENAME).resolve()

    try:
        train_path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(
            f"Configured training data path resolves outside the project: {train_path}"
        ) from error

    if not train_path.exists():
        raise FileNotFoundError(
            "Training dataset not found. Download the Jigsaw Toxic Comment "
            f"Classification Challenge train.csv and place it at: {train_path}"
        )

    return train_path

