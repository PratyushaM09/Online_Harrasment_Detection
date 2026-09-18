from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training import train_one_epoch, validate_model


LABELS = ("toxic", "severe_toxic")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, len(LABELS))
        self.last_label_dtype = None

    def forward(self, input_ids, attention_mask, labels):
        self.last_label_dtype = labels.dtype
        features = input_ids.float() * attention_mask.float()
        logits = self.linear(features)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return SimpleNamespace(loss=loss, logits=logits)


class CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.1)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure=closure)


def _loader(row_count=4, batch_size=1):
    rows = []
    for index in range(row_count):
        rows.append(
            {
                "input_ids": torch.tensor([index + 1, 2, 3], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                "labels": torch.tensor(
                    [float(index % 2), float((index + 1) % 2)],
                    dtype=torch.float32,
                ),
            }
        )
    return DataLoader(rows, batch_size=batch_size)


def test_train_one_epoch_updates_model_and_returns_finite_loss():
    model = TinyModel()
    optimizer = CountingSGD(model.parameters())
    before = model.linear.weight.detach().clone()

    result = train_one_epoch(
        model,
        _loader(row_count=3),
        optimizer,
        torch.device("cpu"),
    )

    assert model.training is True
    assert optimizer.step_count == 3
    assert torch.isfinite(torch.tensor(result.loss))
    assert not torch.equal(model.linear.weight.detach(), before)
    assert model.last_label_dtype == torch.float32


def test_gradient_accumulation_steps_optimizer_less_often():
    model = TinyModel()
    optimizer = CountingSGD(model.parameters())

    result = train_one_epoch(
        model,
        _loader(row_count=4),
        optimizer,
        torch.device("cpu"),
        gradient_accumulation_steps=2,
    )

    assert optimizer.step_count == 2
    assert result.optimizer_steps == 2


def test_final_partial_accumulation_window_steps_optimizer():
    model = TinyModel()
    optimizer = CountingSGD(model.parameters())

    result = train_one_epoch(
        model,
        _loader(row_count=5),
        optimizer,
        torch.device("cpu"),
        gradient_accumulation_steps=2,
    )

    assert optimizer.step_count == 3
    assert result.optimizer_steps == 3


def test_invalid_gradient_accumulation_is_rejected():
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        train_one_epoch(
            TinyModel(),
            _loader(),
            CountingSGD(TinyModel().parameters()),
            torch.device("cpu"),
            gradient_accumulation_steps=0,
        )


def test_validation_uses_eval_mode_sigmoid_and_threshold_metrics():
    model = TinyModel()

    result = validate_model(
        model,
        _loader(row_count=4, batch_size=2),
        torch.device("cpu"),
        labels=LABELS,
        threshold=0.5,
    )

    assert model.training is False
    assert result.threshold == 0.5
    assert result.logits.shape == torch.Size([4, 2])
    assert result.probabilities.shape == torch.Size([4, 2])
    assert result.labels.shape == torch.Size([4, 2])
    assert torch.all(result.probabilities >= 0)
    assert torch.all(result.probabilities <= 1)
    assert set(result.aggregate_metrics) == {
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    }


def test_validation_does_not_create_training_gradients():
    model = TinyModel()

    _ = validate_model(
        model,
        _loader(row_count=3),
        torch.device("cpu"),
        labels=LABELS,
    )

    assert all(parameter.grad is None for parameter in model.parameters())
