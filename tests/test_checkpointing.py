import json

import pytest

from src.training.checkpointing import METADATA_FILENAME, save_smoke_checkpoint


class FakePretrainedObject:
    def __init__(self, marker):
        self.marker = marker
        self.saved_paths = []

    def save_pretrained(self, output_dir):
        self.saved_paths.append(output_dir)
        (output_dir / self.marker).write_text("saved", encoding="utf-8")


def test_checkpoint_directory_and_metadata_are_written(tmp_path):
    checkpoint_dir = tmp_path / "smoke-test"
    model = FakePretrainedObject("model.bin")
    tokenizer = FakePretrainedObject("tokenizer.json")

    save_smoke_checkpoint(
        checkpoint_dir,
        model=model,
        tokenizer=tokenizer,
        metadata={"labels": ["toxic", "insult"], "metric": 0.5},
    )

    metadata = json.loads((checkpoint_dir / METADATA_FILENAME).read_text())
    assert checkpoint_dir.exists()
    assert (checkpoint_dir / "model.bin").exists()
    assert (checkpoint_dir / "tokenizer.json").exists()
    assert metadata["labels"] == ["toxic", "insult"]


def test_checkpoint_overwrite_is_blocked_by_default(tmp_path):
    checkpoint_dir = tmp_path / "smoke-test"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "existing.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Use --overwrite"):
        save_smoke_checkpoint(
            checkpoint_dir,
            model=FakePretrainedObject("model.bin"),
            tokenizer=FakePretrainedObject("tokenizer.json"),
            metadata={"labels": ["toxic"]},
        )


def test_checkpoint_overwrite_is_explicitly_allowed(tmp_path):
    checkpoint_dir = tmp_path / "smoke-test"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "existing.txt").write_text("old", encoding="utf-8")

    save_smoke_checkpoint(
        checkpoint_dir,
        model=FakePretrainedObject("model.bin"),
        tokenizer=FakePretrainedObject("tokenizer.json"),
        metadata={"labels": ["toxic"]},
        overwrite=True,
    )

    metadata = json.loads((checkpoint_dir / METADATA_FILENAME).read_text())
    assert not (checkpoint_dir / "existing.txt").exists()
    assert metadata["labels"] == ["toxic"]
