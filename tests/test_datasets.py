from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from zhaw_at_touche.datasets import build_model_input, detect_generated_text_field, merge_response_split


class DatasetTests(unittest.TestCase):
    def test_detect_generated_text_field(self) -> None:
        row = {
            "id": "1",
            "query": "q",
            "response": "r",
            "label": 1,
            "gemini25flashlite": "generated",
        }
        self.assertEqual(detect_generated_text_field(row, None), "gemini25flashlite")

    def test_build_model_input_with_neutral_reference(self) -> None:
        rendered = build_model_input(
            "How safe is this car?",
            "This answer contains product promotion.",
            input_format="query_neutral_response",
            reference_response="A neutral factual answer.",
            reference_label="GEMINI",
        )
        self.assertIn("USER QUERY: How safe is this car?", rendered)
        self.assertIn("NEUTRAL REFERENCE (GEMINI): A neutral factual answer.", rendered)
        self.assertIn("RESPONSE TO CLASSIFY: This answer contains product promotion.", rendered)

    def test_merge_response_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            response_path = root / "responses-train.jsonl"
            label_path = root / "responses-train-labels.jsonl"

            response_rows = [
                {
                    "id": "row-1",
                    "search_engine": "brave",
                    "meta_topic": "appliances",
                    "query": "query",
                    "response": "response",
                }
            ]
            label_rows = [
                {
                    "id": "row-1",
                    "label": 1,
                    "item": "item",
                    "advertiser": "advertiser",
                }
            ]

            response_path.write_text(
                "\n".join(json.dumps(row) for row in response_rows) + "\n",
                encoding="utf-8",
            )
            label_path.write_text(
                "\n".join(json.dumps(row) for row in label_rows) + "\n",
                encoding="utf-8",
            )

            merged = merge_response_split(response_path, label_path)

            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["label"], 1)
            self.assertEqual(merged[0]["item"], "item")
            self.assertEqual(merged[0]["advertiser"], "advertiser")


if __name__ == "__main__":
    unittest.main()
