import pandas as pd
import pytest

from src.baseline import (
    create_tfidf_vectorizer,
    fit_transform_training_text,
    transform_text,
)


def test_vectorizer_fits_on_training_text():
    dataframe = pd.DataFrame(
        {"comment_text": ["hello world", "hello there", "world there"]}
    )
    vectorizer = create_tfidf_vectorizer(min_df=1)

    fitted_vectorizer, features = fit_transform_training_text(dataframe, vectorizer=vectorizer)

    assert features.shape[0] == len(dataframe)
    assert "hello" in fitted_vectorizer.vocabulary_


def test_vectorizer_transforms_validation_text_without_refitting():
    train_dataframe = pd.DataFrame(
        {"comment_text": ["hello world", "hello there", "world there"]}
    )
    validation_dataframe = pd.DataFrame({"comment_text": ["unseen validation text"]})
    vectorizer = create_tfidf_vectorizer(min_df=1)
    fitted_vectorizer, _ = fit_transform_training_text(
        train_dataframe,
        vectorizer=vectorizer,
    )
    vocabulary_before = dict(fitted_vectorizer.vocabulary_)

    validation_features = transform_text(fitted_vectorizer, validation_dataframe)

    assert validation_features.shape[0] == len(validation_dataframe)
    assert fitted_vectorizer.vocabulary_ == vocabulary_before


def test_missing_text_column_fails_clearly():
    dataframe = pd.DataFrame({"text": ["hello"]})

    with pytest.raises(ValueError, match="Text column not found"):
        fit_transform_training_text(dataframe, vectorizer=create_tfidf_vectorizer())


def test_missing_text_value_fails_clearly():
    dataframe = pd.DataFrame({"comment_text": ["hello", None]})

    with pytest.raises(TypeError, match=r"comment_text\[1\] must be a string"):
        fit_transform_training_text(dataframe, vectorizer=create_tfidf_vectorizer())

