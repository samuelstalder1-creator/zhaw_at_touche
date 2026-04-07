from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from zhaw_at_touche.cli.pairwise_distances import parse_args
from zhaw_at_touche.jsonl import write_jsonl
from zhaw_at_touche.pairwise_distance import (
    FieldPair,
    merge_jsonl_records_by_id,
    parse_field_pair,
    summarize_pairwise_scores,
)


class PairwiseDistanceTests(unittest.TestCase):
    def test_parse_field_pair(self) -> None:
        pair = parse_field_pair("response:qwen")

        self.assertEqual(pair.left_field, "response")
        self.assertEqual(pair.right_field, "qwen")
        self.assertEqual(pair.key, "response__qwen")

    def test_parse_field_pair_rejects_invalid_value(self) -> None:
        with self.assertRaises(ValueError):
            parse_field_pair("response")

    def test_merge_jsonl_records_by_id_combines_provider_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            gemini_path = Path(tmp_dir) / "gemini.jsonl"
            qwen_path = Path(tmp_dir) / "qwen.jsonl"
            write_jsonl(
                gemini_path,
                [
                    {
                        "id": "a1",
                        "query": "query",
                        "response": "injected response",
                        "label": 1,
                        "gemini25flashlite": "gemini neutral",
                    }
                ],
            )
            write_jsonl(
                qwen_path,
                [
                    {
                        "id": "a1",
                        "query": "query",
                        "response": "injected response",
                        "label": 1,
                        "qwen": "qwen neutral",
                    }
                ],
            )

            records = merge_jsonl_records_by_id([gemini_path, qwen_path])

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["gemini25flashlite"], "gemini neutral")
            self.assertEqual(records[0]["qwen"], "qwen neutral")

    def test_merge_jsonl_records_by_id_rejects_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first_path = Path(tmp_dir) / "first.jsonl"
            second_path = Path(tmp_dir) / "second.jsonl"
            write_jsonl(
                first_path,
                [{"id": "a1", "response": "first response"}],
            )
            write_jsonl(
                second_path,
                [{"id": "a1", "response": "different response"}],
            )

            with self.assertRaises(ValueError):
                merge_jsonl_records_by_id([first_path, second_path])

    def test_summarize_pairwise_scores_includes_by_label_stats(self) -> None:
        rows = [
            {"label": 0, "pairwise_scores": {"response__qwen": 0.2}},
            {"label": 1, "pairwise_scores": {"response__qwen": 0.8}},
            {"label": 1, "pairwise_scores": {"response__qwen": 0.6}},
        ]

        summary = summarize_pairwise_scores(rows, [FieldPair("response", "qwen")])

        pair_summary = summary["pairs"]["response__qwen"]
        self.assertEqual(summary["records"], 3)
        self.assertAlmostEqual(pair_summary["mean"], 0.5333333333333333)
        self.assertAlmostEqual(pair_summary["by_label"]["0"]["mean"], 0.2)
        self.assertAlmostEqual(pair_summary["by_label"]["1"]["mean"], 0.7)

    def test_cli_parse_args_accepts_multiple_pairs(self) -> None:
        args = parse_args(
            [
                "--input-files",
                "data/generated/gemini/responses-test-with-neutral_gemini.jsonl",
                "data/generated/qwen/responses-test-with-neutral_qwen.jsonl",
                "--pair",
                "gemini25flashlite:qwen",
                "--pair",
                "response:qwen",
            ]
        )

        self.assertEqual(
            args.input_files,
            [
                "data/generated/gemini/responses-test-with-neutral_gemini.jsonl",
                "data/generated/qwen/responses-test-with-neutral_qwen.jsonl",
            ],
        )
        self.assertEqual(args.pair, ["gemini25flashlite:qwen", "response:qwen"])


if __name__ == "__main__":
    unittest.main()
