from __future__ import annotations

import sys
import unittest
from types import ModuleType, SimpleNamespace

from zhaw_at_touche.cli.generated_stats import build_parser
from zhaw_at_touche.generated_stats import GeneratedRow, build_histogram, token_length_analysis


class GeneratedStatsTests(unittest.TestCase):
    def test_build_histogram_limits_bin_count_to_value_range(self) -> None:
        counts, ranges, min_value, max_value = build_histogram([2, 2, 3, 4], requested_bins=10)

        self.assertEqual(sum(counts), 4)
        self.assertEqual(len(counts), 3)
        self.assertEqual(min_value, 2)
        self.assertEqual(max_value, 4)
        self.assertEqual(ranges[0][0], 2.0)

    def test_token_length_analysis_keeps_legacy_metric_names(self) -> None:
        class FakeLocalTokenizer:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def compute_tokens(self, texts: list[str]) -> SimpleNamespace:
                return SimpleNamespace(
                    tokens_info=[
                        SimpleNamespace(token_ids=list(range(len(text.split()))))
                        for text in texts
                    ]
                )

        google_module = ModuleType("google")
        genai_module = ModuleType("google.genai")
        local_tokenizer_module = ModuleType("google.genai.local_tokenizer")
        local_tokenizer_module.LocalTokenizer = FakeLocalTokenizer
        google_module.genai = genai_module
        genai_module.local_tokenizer = local_tokenizer_module

        rows = [
            GeneratedRow(
                row_id="row-1",
                query="one two",
                response="three four five",
                generated_response="six",
            ),
            GeneratedRow(
                row_id="row-2",
                query="alpha",
                response="beta gamma",
                generated_response="delta epsilon zeta",
            ),
        ]

        original_google = sys.modules.get("google")
        original_google_genai = sys.modules.get("google.genai")
        original_local_tokenizer = sys.modules.get("google.genai.local_tokenizer")
        sys.modules["google"] = google_module
        sys.modules["google.genai"] = genai_module
        sys.modules["google.genai.local_tokenizer"] = local_tokenizer_module
        try:
            analysis = token_length_analysis(rows, "gemini-2.5-flash-lite", batch_size=2)
        finally:
            if original_google is None:
                sys.modules.pop("google", None)
            else:
                sys.modules["google"] = original_google
            if original_google_genai is None:
                sys.modules.pop("google.genai", None)
            else:
                sys.modules["google.genai"] = original_google_genai
            if original_local_tokenizer is None:
                sys.modules.pop("google.genai.local_tokenizer", None)
            else:
                sys.modules["google.genai.local_tokenizer"] = original_local_tokenizer

        self.assertEqual(
            list(analysis.metric_values),
            [
                "query_tokens",
                "response_tokens",
                "neutral_response_tokens",
                "all_together_tokens",
            ],
        )
        self.assertEqual(
            [summary.name for summary in analysis.summaries],
            [
                "query_tokens",
                "response_tokens",
                "neutral_response_tokens",
                "all_together_tokens",
            ],
        )
        self.assertEqual(analysis.metric_values["neutral_response_tokens"], [1, 3])
        self.assertEqual(analysis.metric_values["all_together_tokens"], [6, 6])

    def test_generated_stats_cli_accepts_legacy_aliases(self) -> None:
        args = build_parser().parse_args(
            [
                "sample.jsonl",
                "--model",
                "gemini-2.5-flash-lite",
                "--neutral-field",
                "gemini25flashlite",
            ]
        )

        self.assertEqual(args.paths, ["sample.jsonl"])
        self.assertEqual(args.tokenizer_model, "gemini-2.5-flash-lite")
        self.assertEqual(args.generated_field, "gemini25flashlite")


if __name__ == "__main__":
    unittest.main()
