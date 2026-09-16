"""TF-IDF vectorizer helpers for the classical baseline."""

from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_TEXT_COLUMN = "comment_text"


def create_tfidf_vectorizer(
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: int = 100000,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Create the baseline TF-IDF vectorizer."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )


def fit_transform_training_text(
    dataframe,
    text_column: str = DEFAULT_TEXT_COLUMN,
    vectorizer: TfidfVectorizer | None = None,
):
    """Fit a vectorizer on training text only and return features."""
    texts = _get_texts(dataframe, text_column)
    fitted_vectorizer = vectorizer or create_tfidf_vectorizer()
    features = fitted_vectorizer.fit_transform(texts)

    return fitted_vectorizer, features


def transform_text(
    vectorizer: TfidfVectorizer,
    dataframe,
    text_column: str = DEFAULT_TEXT_COLUMN,
):
    """Transform text with an already-fitted vectorizer."""
    texts = _get_texts(dataframe, text_column)

    return vectorizer.transform(texts)


def _get_texts(dataframe, text_column: str) -> list[str]:
    if text_column not in dataframe.columns:
        raise ValueError(f"Text column not found in DataFrame: {text_column}")

    texts = dataframe[text_column].tolist()
    _validate_texts(texts, text_column)
    return texts


def _validate_texts(texts: Sequence[str], text_column: str) -> None:
    for index, text in enumerate(texts):
        if text is None:
            raise TypeError(f"{text_column}[{index}] must be a string, got None")
        if not isinstance(text, str):
            raise TypeError(
                f"{text_column}[{index}] must be a string, got {type(text).__name__}"
            )

