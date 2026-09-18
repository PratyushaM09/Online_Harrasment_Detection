"""Run a small XLM-R smoke fine-tuning experiment."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.model import create_multilabel_model, select_device  # noqa: E402
from src.tokenization import load_tokenizer  # noqa: E402
from src.training import (  # noqa: E402
    ToxicCommentDataset,
    create_dataloader,
    sample_multilabel_dataframe,
    save_smoke_checkpoint,
    train_one_epoch,
    validate_model,
)
from src.utils import set_seed  # noqa: E402


TEXT_COLUMN = "comment_text"


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Run the Milestone 6A XLM-R smoke-training experiment."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the smoke-test checkpoint after training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing smoke-test checkpoint when used with --save.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    set_seed(config.project.random_seed)

    train_path = PROJECT_ROOT / config.paths.processed_data / "train.csv"
    validation_path = PROJECT_ROOT / config.paths.processed_data / "validation.csv"
    if not train_path.exists():
        print(f"Processed training split not found: {train_path}")
        return 1
    if not validation_path.exists():
        print(f"Processed validation split not found: {validation_path}")
        return 1

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: not available")

    train_frame = pd.read_csv(train_path)
    validation_frame = pd.read_csv(validation_path)
    train_sample = sample_multilabel_dataframe(
        train_frame,
        labels=config.labels,
        sample_size=config.training.train_sample_size,
        random_seed=config.project.random_seed,
    )
    validation_sample = sample_multilabel_dataframe(
        validation_frame,
        labels=config.labels,
        sample_size=config.training.validation_sample_size,
        random_seed=config.project.random_seed,
    )

    tokenizer = load_tokenizer(config.model.name)
    train_dataset = ToxicCommentDataset(
        dataframe=train_sample,
        tokenizer=tokenizer,
        text_column=TEXT_COLUMN,
        label_columns=config.labels,
        max_length=config.tokenization.max_length,
    )
    validation_dataset = ToxicCommentDataset(
        dataframe=validation_sample,
        tokenizer=tokenizer,
        text_column=TEXT_COLUMN,
        label_columns=config.labels,
        max_length=config.tokenization.max_length,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    model = create_multilabel_model(config.model.name, config.labels)
    device = select_device()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    print(f"training sample size: {len(train_sample)}")
    print(f"validation sample size: {len(validation_sample)}")
    print(f"batch size: {config.training.batch_size}")
    print(f"max length: {config.tokenization.max_length}")
    print(f"epochs: {config.training.epochs}")
    print(f"device: {device}")

    start_time = perf_counter()
    last_train_result = None
    last_validation_result = None
    for epoch in range(1, config.training.epochs + 1):
        last_train_result = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        )
        last_validation_result = validate_model(
            model=model,
            dataloader=validation_loader,
            device=device,
            labels=config.labels,
            threshold=config.training.default_threshold,
        )
        metrics = last_validation_result.aggregate_metrics

        print(f"Epoch {epoch}/{config.training.epochs}")
        print(f"training loss: {last_train_result.loss:.6f}")
        print(f"validation loss: {last_validation_result.loss:.6f}")
        print(f"micro precision: {metrics['micro_precision']:.6f}")
        print(f"micro recall: {metrics['micro_recall']:.6f}")
        print(f"micro F1: {metrics['micro_f1']:.6f}")
        print(f"macro precision: {metrics['macro_precision']:.6f}")
        print(f"macro recall: {metrics['macro_recall']:.6f}")
        print(f"macro F1: {metrics['macro_f1']:.6f}")
        print(f"elapsed time: {last_train_result.elapsed_seconds:.2f}s")
        print(f"AMP used: {last_train_result.amp_used}")

    total_elapsed = perf_counter() - start_time
    checkpoint_path = PROJECT_ROOT / config.paths.models / "xlmr" / "smoke-test"
    if args.save:
        metadata = {
            "model_name": config.model.name,
            "labels": list(config.labels),
            "max_length": config.tokenization.max_length,
            "train_sample_size": config.training.train_sample_size,
            "validation_sample_size": config.training.validation_sample_size,
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "random_seed": config.project.random_seed,
            "threshold": config.training.default_threshold,
            "validation_loss": last_validation_result.loss,
            "validation_metrics": last_validation_result.aggregate_metrics,
            "training_loss": last_train_result.loss,
            "elapsed_seconds": total_elapsed,
            "amp_used": last_train_result.amp_used,
            "device": str(device),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        saved_path = save_smoke_checkpoint(
            checkpoint_path,
            model=model,
            tokenizer=tokenizer,
            metadata=metadata,
            overwrite=args.overwrite,
        )
        print(f"checkpoint path: {saved_path}")
    else:
        print("checkpoint path: not saved; rerun with --save to persist")

    print(f"total elapsed time: {total_elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
