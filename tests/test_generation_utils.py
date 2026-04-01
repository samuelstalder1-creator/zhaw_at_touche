from __future__ import annotations

import unittest

from zhaw_at_touche.cli.generate_neutral import build_parser, resolve_backend, resolve_model
from zhaw_at_touche.generation_utils import (
    _openai_compatible_text,
    build_chat_messages,
    clean_response_text,
    default_backend_for_provider,
    get_openai_compatible_usage_counts,
    get_transformers_usage_counts,
    model_alias,
)


class GenerationUtilsTests(unittest.TestCase):
    def test_clean_response_text_flattens_bullets(self) -> None:
        text = "\\u2022 First point\\n- Second point\\n\\nThird line"
        cleaned = clean_response_text(text)
        self.assertEqual(cleaned, "First point Second point Third line")

    def test_model_alias(self) -> None:
        self.assertEqual(model_alias("gemini-2.5-flash-lite"), "gemini25flashlite")
        self.assertEqual(model_alias("Qwen/Qwen2.5-1.5B-Instruct"), "qwen")
        self.assertEqual(model_alias("custom/model-v1"), "custommodelv1")

    def test_default_backend_for_provider(self) -> None:
        self.assertEqual(default_backend_for_provider("gemini"), "gemini")
        self.assertEqual(default_backend_for_provider("qwen"), "transformers")

    def test_transformers_generation_helpers(self) -> None:
        messages = build_chat_messages("What is a dehumidifier?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Write a helpful, factual answer", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "What is a dehumidifier?"})
        self.assertEqual(
            get_transformers_usage_counts(input_tokens=13, output_tokens=21),
            {
                "input_tokens": 13,
                "cached_input_tokens": 0,
                "output_tokens": 21,
                "total_tokens": 34,
            },
        )

    def test_openai_compatible_response_helpers(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "First sentence."},
                            {"type": "text", "text": "Second sentence."},
                        ]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 17,
                "total_tokens": 28,
            },
        }

        self.assertEqual(_openai_compatible_text(payload), "First sentence. Second sentence.")
        self.assertEqual(
            get_openai_compatible_usage_counts(payload),
            {
                "input_tokens": 11,
                "cached_input_tokens": 0,
                "output_tokens": 17,
                "total_tokens": 28,
            },
        )

    def test_generate_neutral_cli_defaults_for_qwen(self) -> None:
        args = build_parser().parse_args(["--provider", "qwen", "--split", "validation"])

        self.assertEqual(resolve_backend(args), "transformers")
        self.assertEqual(resolve_model(args), "Qwen/Qwen2.5-1.5B-Instruct")


if __name__ == "__main__":
    unittest.main()
