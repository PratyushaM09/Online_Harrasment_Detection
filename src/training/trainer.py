"""Small native PyTorch trainer for the Milestone 6A smoke experiment."""

from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import torch

from src.training.metrics import (
    ValidationResult,
    build_validation_result,
    get_model_output_value,
)


@dataclass(frozen=True)
class TrainingEpochResult:
    """Summary of one training epoch."""

    loss: float
    optimizer_steps: int
    elapsed_seconds: float
    amp_used: bool


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = True,
) -> TrainingEpochResult:
    """Train for one epoch with optional CUDA AMP and gradient accumulation."""
    _validate_gradient_accumulation_steps(gradient_accumulation_steps)
    amp_enabled = _amp_enabled(device, use_amp)
    scaler = _create_grad_scaler(amp_enabled)
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    batch_count = 0
    optimizer_steps = 0
    pending_gradients = False
    start_time = perf_counter()

    for batch_index, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).float()

        with _autocast_context(device, amp_enabled):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = get_model_output_value(output, "loss")
            scaled_loss = loss / gradient_accumulation_steps

        if scaler is None:
            scaled_loss.backward()
        else:
            scaler.scale(scaled_loss).backward()
        total_loss += float(loss.detach().cpu().item())
        batch_count += 1
        pending_gradients = True

        if batch_index % gradient_accumulation_steps == 0:
            _optimizer_step(optimizer, scaler)
            optimizer.zero_grad()
            optimizer_steps += 1
            pending_gradients = False

    if pending_gradients:
        _optimizer_step(optimizer, scaler)
        optimizer.zero_grad()
        optimizer_steps += 1

    return TrainingEpochResult(
        loss=total_loss / max(batch_count, 1),
        optimizer_steps=optimizer_steps,
        elapsed_seconds=perf_counter() - start_time,
        amp_used=amp_enabled,
    )


def validate_model(
    model,
    dataloader,
    device: torch.device,
    labels: Sequence[str],
    threshold: float = 0.5,
    use_amp: bool = True,
) -> ValidationResult:
    """Run validation without gradient tracking."""
    amp_enabled = _amp_enabled(device, use_amp)
    model.eval()

    total_loss = 0.0
    batch_count = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_tensor = batch["labels"].to(device).float()

            with _autocast_context(device, amp_enabled):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=label_tensor,
                )
                loss = get_model_output_value(output, "loss")
                logits = get_model_output_value(output, "logits")

            total_loss += float(loss.detach().cpu().item())
            batch_count += 1
            all_logits.append(logits.detach().cpu())
            all_labels.append(label_tensor.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    true_labels = torch.cat(all_labels, dim=0)

    return build_validation_result(
        validation_loss=total_loss / max(batch_count, 1),
        logits=logits,
        labels=true_labels,
        label_names=labels,
        threshold=threshold,
    )


def _amp_enabled(device: torch.device, use_amp: bool) -> bool:
    return bool(use_amp and device.type == "cuda" and torch.cuda.is_available())


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, enabled=True)


def _create_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.amp.GradScaler(enabled=True)


def _optimizer_step(optimizer, scaler) -> None:
    if scaler is None:
        optimizer.step()
        return

    scaler.step(optimizer)
    scaler.update()


def _validate_gradient_accumulation_steps(gradient_accumulation_steps: int) -> None:
    if isinstance(gradient_accumulation_steps, bool) or not isinstance(
        gradient_accumulation_steps,
        int,
    ):
        raise TypeError(
            "gradient_accumulation_steps must be an int, got "
            f"{type(gradient_accumulation_steps).__name__}"
        )
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0")
