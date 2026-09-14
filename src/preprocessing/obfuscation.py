"""Explicit obfuscated-expression detection helpers."""

import re

from src.preprocessing.models import DetectedExpression
from src.preprocessing.slang_dictionary import detect_slang


OBFUSCATION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("b!tch", "bitch"),
    ("h@te", "hate"),
    ("k*ll", "kill"),
)


def detect_obfuscations(text: str) -> tuple[DetectedExpression, ...]:
    """Detect explicit obfuscation patterns without rewriting text."""
    _validate_text(text)
    detections: list[DetectedExpression] = []

    for surface_form, canonical in OBFUSCATION_PATTERNS:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(surface_form)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            detections.append(
                DetectedExpression(
                    surface=match.group(0),
                    canonical=canonical,
                    kind="obfuscation",
                    start=match.start(),
                    end=match.end(),
                )
            )

    return tuple(
        sorted(
            detections,
            key=lambda detection: (
                detection.start,
                detection.end,
                detection.kind,
                detection.surface,
            ),
        )
    )


def detect_flagged_expressions(text: str) -> tuple[DetectedExpression, ...]:
    """Combine slang and obfuscation detections in deterministic text order."""
    _validate_text(text)
    detections = (*detect_slang(text), *detect_obfuscations(text))

    return tuple(
        sorted(
            detections,
            key=lambda detection: (
                detection.start,
                detection.end,
                detection.kind,
                detection.surface,
            ),
        )
    )


def _validate_text(text: str) -> None:
    if text is None:
        raise TypeError("text must be a string, got None")
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

