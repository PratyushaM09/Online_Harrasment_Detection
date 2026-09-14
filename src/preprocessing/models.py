"""Preprocessing result models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingResult:
    """Immutable result from conservative text normalization."""

    original_text: str
    normalized_text: str
    normalization_events: tuple[str, ...]


@dataclass(frozen=True)
class DetectedExpression:
    """Immutable metadata for a detected slang or obfuscated expression."""

    surface: str
    canonical: str
    kind: str
    start: int
    end: int
