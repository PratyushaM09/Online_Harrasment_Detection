from src.model import create_model_metadata
from src.model.factory import PROBLEM_TYPE


def test_model_metadata_preserves_configuration_values():
    labels = ("toxic", "insult")

    metadata = create_model_metadata(
        model_name="xlm-roberta-base",
        labels=labels,
        max_length=256,
    )

    assert metadata.model_name == "xlm-roberta-base"
    assert metadata.labels == labels
    assert metadata.num_labels == 2
    assert metadata.problem_type == PROBLEM_TYPE
    assert metadata.max_length == 256

