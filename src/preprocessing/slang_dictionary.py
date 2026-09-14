"""Small auditable slang dictionary and detection helpers."""

import re

from src.preprocessing.models import DetectedExpression


SLANG_TERMS: dict[str, str] = {
    "kys": "kill yourself",
    "unalive": "kill",
    "h8": "hate",
}


def detect_slang(text: str) -> tuple[DetectedExpression, ...]:
    """Detect curated internet slang as metadata without rewriting text."""
    _validate_text(text)
    detections: list[DetectedExpression] = []

    for surface_form, canonical in SLANG_TERMS.items():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(surface_form)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            detections.append(
                DetectedExpression(
                    surface=match.group(0),
                    canonical=canonical,
                    kind="slang",
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


def _validate_text(text: str) -> None:
    if text is None:
        raise TypeError("text must be a string, got None")
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

