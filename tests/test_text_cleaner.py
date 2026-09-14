import pandas as pd
import pytest

from src.preprocessing import normalize_text, preprocess_dataframe


def test_plain_text_remains_unchanged():
    text = "This is plain text."

    result = normalize_text(text)

    assert result.original_text == text
    assert result.normalized_text == text
    assert result.normalization_events == ()


def test_leading_and_trailing_whitespace_is_trimmed():
    result = normalize_text("   hello world   ")

    assert result.normalized_text == "hello world"
    assert result.normalization_events == ("trimmed_whitespace",)


def test_repeated_internal_spaces_normalize_correctly():
    result = normalize_text("you    are\tawful")

    assert result.normalized_text == "you are awful"
    assert "internal_whitespace_normalized" in result.normalization_events


def test_internal_whitespace_preserves_line_boundaries():
    result = normalize_text("line one    here\nline two\tthere")

    assert result.normalized_text == "line one here\nline two there"


def test_line_endings_are_normalized():
    result = normalize_text("first\r\nsecond\rthird")

    assert result.normalized_text == "first\nsecond\nthird"
    assert result.normalization_events == ("line_endings_normalized",)


def test_urls_become_placeholder():
    result = normalize_text("go to https://example.com now")

    assert result.normalized_text == "go to <URL> now"
    assert "url_replaced" in result.normalization_events


def test_mentions_become_placeholder():
    result = normalize_text("@alex you are awful")

    assert result.normalized_text == "<USER> you are awful"
    assert "mention_replaced" in result.normalization_events


def test_email_addresses_are_not_treated_as_mentions():
    text = "send mail to alex@example.com"

    result = normalize_text(text)

    assert result.normalized_text == text
    assert "mention_replaced" not in result.normalization_events


def test_emoji_are_preserved():
    text = "That was rude \U0001F92C"

    result = normalize_text(text)

    assert "\U0001F92C" in result.normalized_text


def test_punctuation_and_case_are_preserved():
    text = "YOU are such a b!tch!!!!"

    result = normalize_text(text)

    assert result.normalized_text == text


def test_obfuscated_terms_remain_unchanged():
    text = "b!tch k*ll h@te"

    result = normalize_text(text)

    assert result.normalized_text == text


def test_repeated_letters_remain_unchanged():
    text = "sooooo rude"

    result = normalize_text(text)

    assert result.normalized_text == text


def test_non_latin_unicode_remains_present():
    text = "\u0928\u092e\u0938\u094d\u0924\u0947 world"

    result = normalize_text(text)

    assert "\u0928\u092e\u0938\u094d\u0924\u0947" in result.normalized_text


def test_unicode_is_normalized_without_transliteration():
    text = "Cafe\u0301"

    result = normalize_text(text)

    assert result.normalized_text == "Caf\u00e9"
    assert result.normalization_events == ("unicode_normalized",)


def test_none_is_rejected():
    with pytest.raises(TypeError, match="text must be a string"):
        normalize_text(None)


def test_non_string_values_are_rejected():
    with pytest.raises(TypeError, match="text must be a string"):
        normalize_text(123)


def test_empty_string_is_handled():
    result = normalize_text("")

    assert result.normalized_text == ""
    assert result.normalization_events == ()


def test_whitespace_only_string_normalizes_predictably():
    result = normalize_text("   \t   ")

    assert result.normalized_text == ""
    assert result.normalization_events == ("trimmed_whitespace",)


def test_normalization_events_appear_only_when_relevant():
    result = normalize_text("  @alex   visit https://example.com  ")

    assert result.normalization_events == (
        "trimmed_whitespace",
        "internal_whitespace_normalized",
        "url_replaced",
        "mention_replaced",
    )


def test_preprocess_dataframe_adds_columns_without_mutating_source():
    dataframe = pd.DataFrame(
        {
            "id": ["1", "2"],
            "comment_text": ["  @alex hi  ", "YOU are such a b!tch!!!!"],
            "toxic": [0, 1],
        }
    )
    original = dataframe.copy(deep=True)

    preprocessed = preprocess_dataframe(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)
    assert "original_text" in preprocessed.columns
    assert "normalized_text" in preprocessed.columns
    assert "normalization_events" in preprocessed.columns
    assert preprocessed.loc[0, "original_text"] == "  @alex hi  "
    assert preprocessed.loc[0, "normalized_text"] == "<USER> hi"
    assert preprocessed.loc[1, "normalized_text"] == "YOU are such a b!tch!!!!"
