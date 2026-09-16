"""Inspect the XLM-R multi-label model configuration and sanity output."""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.model import (  # noqa: E402
    create_label_mappings,
    create_model_metadata,
    create_multilabel_model,
    logits_to_probabilities,
    select_device,
)
from src.tokenization import load_tokenizer  # noqa: E402


def main() -> int:
    """Load the model and run one pre-fine-tuning sanity forward pass."""
    config = load_config()
    device = select_device()
    metadata = create_model_metadata(
        model_name=config.model.name,
        labels=config.labels,
        max_length=config.tokenization.max_length,
    )
    label2id, id2label = create_label_mappings(config.labels)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device}")
    print(f"Model name: {metadata.model_name}")
    print(f"Num labels: {metadata.num_labels}")
    print(f"Problem type: {metadata.problem_type}")
    print(f"Label2ID: {label2id}")
    print(f"ID2Label: {id2label}")

    tokenizer = load_tokenizer(config.model.name)
    model = create_multilabel_model(config.model.name, config.labels)
    model.to(device)
    model.eval()

    encoded = tokenizer(
        "You are rude.",
        truncation=True,
        padding="max_length",
        max_length=config.tokenization.max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = logits_to_probabilities(outputs.logits)

    print("UNTRAINED / PRE-FINE-TUNING SANITY OUTPUT")
    print(f"Logits shape: {tuple(outputs.logits.shape)}")
    print(f"Probabilities: {probabilities.cpu().tolist()[0]}")
    print("These values are not meaningful harassment predictions yet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

