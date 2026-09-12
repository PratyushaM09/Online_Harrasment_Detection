"""Lightweight project entry point."""

from src.config import load_config


READY_MESSAGE = "Online Harassment Detection project environment is ready."


def main() -> None:
    """Print a simple readiness message using centralized configuration."""
    config = load_config()

    print(READY_MESSAGE)
    print(f"Project: {config.project.name}")
    print(f"Labels configured: {len(config.labels)}")


if __name__ == "__main__":
    main()
