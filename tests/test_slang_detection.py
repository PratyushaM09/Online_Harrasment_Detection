import pytest

from src.preprocessing import (
    detect_flagged_expressions,
    detect_obfuscations,
    detect_slang,
    normalize_text,
)


def test_kys_is_detected():
    detections = detect_slang("you should kys")

    assert detections[0].surface == "kys"
    assert detections[0].canonical == "kill yourself"
    assert detections[0].kind == "slang"


def test_unalive_is_detected():
    detections = detect_slang("do not unalive anyone")

    assert detections[0].surface == "unalive"
    assert detections[0].canonical == "kill"


def test_h8_is_detected():
    detections = detect_slang("i h8 this")

    assert detections[0].surface == "h8"
    assert detections[0].canonical == "hate"


def test_slang_detection_is_case_insensitive():
    detections = detect_slang("KYS now")

    assert detections[0].surface == "KYS"
    assert detections[0].canonical == "kill yourself"


def test_slang_is_not_matched_inside_larger_words():
    text = "akysb unaliveness high8"

    assert detect_slang(text) == ()


def test_bitch_obfuscation_is_detected():
    detections = detect_obfuscations("stop saying b!tch")

    assert detections[0].surface == "b!tch"
    assert detections[0].canonical == "bitch"
    assert detections[0].kind == "obfuscation"


def test_hate_obfuscation_is_detected():
    detections = detect_obfuscations("h@te is explicit")

    assert detections[0].surface == "h@te"
    assert detections[0].canonical == "hate"


def test_kill_obfuscation_is_detected():
    detections = detect_obfuscations("do not write k*ll")

    assert detections[0].surface == "k*ll"
    assert detections[0].canonical == "kill"


def test_harmless_text_gives_empty_result():
    assert detect_flagged_expressions("this is fine") == ()


def test_multiple_detections_preserve_order():
    text = "kys then h@te then unalive then b!tch"

    detections = detect_flagged_expressions(text)

    assert [detection.surface for detection in detections] == [
        "kys",
        "h@te",
        "unalive",
        "b!tch",
    ]


def test_offsets_are_correct():
    text = "go kys now"
    detection = detect_flagged_expressions(text)[0]

    assert detection.start == 3
    assert detection.end == 6
    assert text[detection.start : detection.end] == detection.surface


def test_original_text_is_unchanged():
    text = "you should kys"
    original = text[:]

    detect_flagged_expressions(text)

    assert text == original


def test_no_automatic_replacement_occurs_in_normalization():
    text = "you should kys and say b!tch"

    result = normalize_text(text)

    assert result.original_text == text
    assert result.normalized_text == text


def test_detection_output_is_deterministic():
    text = "KYS h8 h@te k*ll unalive"

    assert detect_flagged_expressions(text) == detect_flagged_expressions(text)


def test_non_string_input_is_rejected_clearly():
    with pytest.raises(TypeError, match="text must be a string"):
        detect_flagged_expressions(123)


def test_none_input_is_rejected_clearly():
    with pytest.raises(TypeError, match="text must be a string"):
        detect_flagged_expressions(None)


def test_detection_does_not_make_toxicity_decisions():
    text = "This game says 'unalive' as a meme."
    detections = detect_flagged_expressions(text)

    assert detections[0].surface == "unalive"
    assert not hasattr(detections[0], "prediction")
    assert not hasattr(detections[0], "toxicity_score")

