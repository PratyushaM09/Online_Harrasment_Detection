"""Demonstrate slang and obfuscation detection metadata."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import detect_flagged_expressions  # noqa: E402


EXAMPLES = (
    "kys",
    "you should unalive yourself",
    "i h8 this",
    "b!tch",
    "k*ll",
    "h@te",
    "this is fine",
    "KYS and h@te are detected, but the text is not rewritten",
)


def main() -> int:
    """Print detection metadata for curated examples."""
    for text in EXAMPLES:
        print("INPUT")
        print(text)
        print("DETECTED EXPRESSIONS")

        detections = detect_flagged_expressions(text)
        if not detections:
            print("none")
        for detection in detections:
            print(f"surface: {detection.surface}")
            print(f"canonical: {detection.canonical}")
            print(f"kind: {detection.kind}")
            print(f"offsets: {detection.start}-{detection.end}")

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

