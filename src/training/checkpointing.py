"""Checkpoint persistence for smoke-training experiments."""

from dataclasses import asdict, is_dataclass
import json
from math import isfinite
from pathlib import Path
import shutil
from typing import Any


METADATA_FILENAME = "smoke_training_metadata.json"


def save_smoke_checkpoint(
    checkpoint_dir: str | Path,
    model,
    tokenizer,
    metadata: dict[str, Any],
    overwrite: bool = False,
) -> Path:
    """Persist a smoke-test checkpoint without silently overwriting outputs."""
    output_dir = Path(checkpoint_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Smoke-test checkpoint already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata_path = output_dir / METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return output_dir


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return float(value) if isfinite(value) else None
    return value
