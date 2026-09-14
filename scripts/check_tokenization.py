"""Demonstrate XLM-R tokenizer behavior on multilingual and noisy examples."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.tokenization import inspect_tokens, load_tokenizer, tokenize_text  # noqa: E402


EXAMPLES = (
    "You are rude.",
    "YOU are sooooo rude!!!",
    "b!tch",
    "k*ll",
    "h@te",
    "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
    "Bonjour, tu es impoli.",
    "Hola, eres grosero.",
)


def main() -> int:
    """Print readable tokenizer output for demonstration examples."""
    config = load_config()
    tokenizer = load_tokenizer(config.model.name)
    max_length = config.tokenization.max_length

    for text in EXAMPLES:
        token_info = inspect_tokens(text, tokenizer)
        encoded = tokenize_text(text, tokenizer, max_length=max_length)
        non_padding_count = int(sum(encoded["attention_mask"]))

        print("INPUT")
        print(text)
        print("TOKENS")
        print(token_info["tokens"])
        print("INPUT IDS")
        print(encoded["input_ids"][:non_padding_count])
        print("ATTENTION MASK")
        print(encoded["attention_mask"][:non_padding_count])
        print("NON-PADDING TOKEN COUNT")
        print(non_padding_count)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

