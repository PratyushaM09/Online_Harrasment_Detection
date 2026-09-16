import pandas as pd
import pytest
import torch

from src.model import (
    build_label_targets,
    create_label_mappings,
    create_multilabel_model,
    logits_to_probabilities,
)
from src.model.factory import PROBLEM_TYPE


LABELS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)


class FakeModelLoader:
    calls = []

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.calls.append({"model_name": model_name, **kwargs})
        return {"model_name": model_name, **kwargs}


def test_model_factory_forwards_multilabel_configuration():
    FakeModelLoader.calls = []

    model = create_multilabel_model("xlm-roberta-base", LABELS, model_loader=FakeModelLoader)

    assert model["num_labels"] == 6
    assert model["problem_type"] == PROBLEM_TYPE
    assert model["label2id"]["toxic"] == 0
    assert model["id2label"][5] == "identity_hate"


def test_label_mappings_preserve_order():
    label2id, id2label = create_label_mappings(LABELS)

    assert label2id == {
        "toxic": 0,
        "severe_toxic": 1,
        "obscene": 2,
        "threat": 3,
        "insult": 4,
        "identity_hate": 5,
    }
    assert id2label[0] == "toxic"
    assert id2label[5] == "identity_hate"


def test_empty_label_list_is_rejected():
    with pytest.raises(ValueError, match="labels must not be empty"):
        create_label_mappings(())


def test_duplicate_labels_are_rejected():
    with pytest.raises(ValueError, match="labels must be unique"):
        create_label_mappings(("toxic", "toxic"))


def test_logits_to_probabilities_uses_sigmoid_and_preserves_shape():
    logits = torch.tensor([[0.0, 1.0, -1.0]])

    probabilities = logits_to_probabilities(logits)

    assert probabilities.shape == logits.shape
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)
    assert probabilities[0, 0].item() == pytest.approx(0.5)


def test_build_label_targets_preserves_order_and_float_dtype():
    row = {
        "toxic": 1,
        "severe_toxic": 0,
        "obscene": 1,
        "threat": 0,
        "insult": 1,
        "identity_hate": 0,
    }

    targets = build_label_targets(row, LABELS)

    assert targets.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert targets.dtype == torch.float32


def test_build_label_targets_rejects_missing_labels():
    with pytest.raises(ValueError, match="Missing required label values"):
        build_label_targets({"toxic": 1}, LABELS)


def test_build_label_targets_rejects_invalid_values():
    row = {label: 0 for label in LABELS}
    row["toxic"] = 2

    with pytest.raises(ValueError, match="must be 0 or 1"):
        build_label_targets(row, LABELS)


def test_build_label_targets_does_not_mutate_source_row():
    frame = pd.DataFrame([{label: 0 for label in LABELS}])
    original = frame.copy(deep=True)

    build_label_targets(frame.iloc[0], LABELS)

    pd.testing.assert_frame_equal(frame, original)

