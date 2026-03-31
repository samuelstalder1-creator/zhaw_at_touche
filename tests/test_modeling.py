from __future__ import annotations

import unittest
from unittest.mock import patch

from zhaw_at_touche import modeling


class TokenizerLoadingTests(unittest.TestCase):
    def test_load_tokenizer_wraps_missing_backend_errors(self) -> None:
        backend_errors = [
            ImportError(
                "SentencePieceExtractor requires the SentencePiece library but it was not found "
                "in your environment."
            ),
            ValueError(
                "`tiktoken` is required to read a `tiktoken` file. Install it with "
                "`pip install tiktoken`."
            ),
        ]

        for error in backend_errors:
            with self.subTest(error=type(error).__name__):
                with patch.object(modeling.AutoTokenizer, "from_pretrained", side_effect=error):
                    with self.assertRaisesRegex(RuntimeError, "uv sync"):
                        modeling.load_tokenizer_from_pretrained("microsoft/deberta-v3-base")

    def test_load_tokenizer_preserves_unrelated_errors(self) -> None:
        with patch.object(
            modeling.AutoTokenizer,
            "from_pretrained",
            side_effect=ValueError("Tokenizer config is invalid."),
        ):
            with self.assertRaisesRegex(ValueError, "Tokenizer config is invalid."):
                modeling.load_tokenizer_from_pretrained("broken/model")


if __name__ == "__main__":
    unittest.main()
