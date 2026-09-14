"""Conservative text normalization for noisy social-media text."""

import re
import unicodedata

import pandas as pd

from src.preprocessing.models import PreprocessingResult


UNICODE_NORMALIZED = "unicode_normalized"
LINE_ENDINGS_NORMALIZED = "line_endings_normalized"
TRIMMED_WHITESPACE = "trimmed_whitespace"
INTERNAL_WHITESPACE_NORMALIZED = "internal_whitespace_normalized"
URL_REPLACED = "url_replaced"
MENTION_REPLACED = "mention_replaced"

URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<![\w.])@[A-Za-z0-9_]+")
INTERNAL_HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[ \t\f\v]+")


def normalize_text(text: str) -> PreprocessingResult:
    """Normalize text conservatively while preserving stylistic evidence."""
    if text is None:
        raise TypeError("text must be a string, got None")
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    original_text = text
    normalized_text = text
    events: list[str] = []

    unicode_normalized = unicodedata.normalize("NFC", normalized_text)
    if unicode_normalized != normalized_text:
        events.append(UNICODE_NORMALIZED)
        normalized_text = unicode_normalized

    line_endings_normalized = normalized_text.replace("\r\n", "\n").replace("\r", "\n")
    if line_endings_normalized != normalized_text:
        events.append(LINE_ENDINGS_NORMALIZED)
        normalized_text = line_endings_normalized

    trimmed_text = normalized_text.strip()
    if trimmed_text != normalized_text:
        events.append(TRIMMED_WHITESPACE)
        normalized_text = trimmed_text

    whitespace_normalized = INTERNAL_HORIZONTAL_WHITESPACE_PATTERN.sub(
        " ", normalized_text
    )
    if whitespace_normalized != normalized_text:
        events.append(INTERNAL_WHITESPACE_NORMALIZED)
        normalized_text = whitespace_normalized

    url_normalized = URL_PATTERN.sub("<URL>", normalized_text)
    if url_normalized != normalized_text:
        events.append(URL_REPLACED)
        normalized_text = url_normalized

    mention_normalized = MENTION_PATTERN.sub("<USER>", normalized_text)
    if mention_normalized != normalized_text:
        events.append(MENTION_REPLACED)
        normalized_text = mention_normalized

    return PreprocessingResult(
        original_text=original_text,
        normalized_text=normalized_text,
        normalization_events=tuple(events),
    )


def preprocess_dataframe(
    dataframe: pd.DataFrame,
    text_column: str = "comment_text",
) -> pd.DataFrame:
    """Return a new DataFrame with conservative preprocessing columns added."""
    if text_column not in dataframe.columns:
        raise ValueError(f"Text column not found in DataFrame: {text_column}")

    preprocessed = dataframe.copy(deep=True)
    results = preprocessed[text_column].apply(normalize_text)

    preprocessed["original_text"] = results.map(lambda result: result.original_text)
    preprocessed["normalized_text"] = results.map(lambda result: result.normalized_text)
    preprocessed["normalization_events"] = results.map(
        lambda result: result.normalization_events
    )

    return preprocessed

