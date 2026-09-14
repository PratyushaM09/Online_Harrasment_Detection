"""Demonstrate conservative preprocessing examples."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import normalize_text  # noqa: E402


EXAMPLES = {
    "extra whitespace": "   you    are\tawful   ",
    "URL": "go to https://example.com now",
    "mention": "@alex you are awful",
    "emoji": "YOU are rude!!! \U0001F92C",
    "obfuscation": "b!tch k*ll h@te",
    "multilingual Unicode": "\u0928\u092e\u0938\u094d\u0924\u0947 world",
    "repeated letters": "sooooo rude",
    "mixed casing": "YOU are such a b!tch!!!!",
    "combined": "@alex   YOU are sooooo rude!!! \U0001F92C https://example.com",
}


def main() -> int:
    """Print before/after examples for manual inspection."""
    for label, text in EXAMPLES.items():
        result = normalize_text(text)
        print(f"EXAMPLE: {label}")
        print("INPUT:")
        print(result.original_text)
        print("OUTPUT:")
        print(result.normalized_text)
        print("EVENTS:")
        if result.normalization_events:
            for event in result.normalization_events:
                print(event)
        else:
            print("none")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

