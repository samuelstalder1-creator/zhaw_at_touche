from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from zhaw_at_touche.anchor_distance_threshold import derive_score_columns
from zhaw_at_touche.cli.anchor_distance_threshold import (
    apply_saved_state_defaults,
    parse_args,
)


class AnchorDistanceThresholdHelpersTests(unittest.TestCase):
    def test_derive_score_columns_computes_handcrafted_score(self) -> None:
        feature_columns = {
            "query__gemini25flashlite_distance": torch.tensor([0.1], dtype=torch.float32),
            "query__qwen_distance": torch.tensor([0.2], dtype=torch.float32),
            "gemini25flashlite__qwen_distance": torch.tensor([0.3], dtype=torch.float32),
            "query__response_distance": torch.tensor([0.7], dtype=torch.float32),
            "gemini25flashlite__response_distance": torch.tensor([0.8], dtype=torch.float32),
            "qwen__response_distance": torch.tensor([0.9], dtype=torch.float32),
        }

        derived = derive_score_columns(
            feature_columns=feature_columns,
            query_field="query",
            response_field="response",
            neutral_field="gemini25flashlite",
            aux_neutral_field="qwen",
        )

        self.assertAlmostEqual(float(derived["anchor_cohesion"][0]), 0.2)
        self.assertAlmostEqual(float(derived["response_drift"][0]), 0.8)
        self.assertAlmostEqual(float(derived["anchor_distance_score"][0]), 0.6)


class AnchorDistanceThresholdCliTests(unittest.TestCase):
    def test_parse_args_uses_setup_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "setup111.json").write_text(
                json.dumps(
                    {
                        "scoring_backend": "anchor_distance_threshold",
                        "model_dir": "models/setup111",
                        "input_files": ["gemini-test.jsonl"],
                        "aux_input_files": ["qwen-test.jsonl"],
                        "calibration_input_files": ["gemini-validation.jsonl"],
                        "aux_calibration_input_files": ["qwen-validation.jsonl"],
                        "embedding_model_name": "sentence-transformers/all-mpnet-base-v2",
                        "neutral_field": "gemini25flashlite",
                        "aux_neutral_field": "qwen",
                        "score_granularity": "response",
                        "batch_size": 64,
                    }
                ),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--setup-name",
                    "setup111",
                    "--setups-dir",
                    str(setups_dir),
                ]
            )

            self.assertEqual(args.model_dir, "models/setup111")
            self.assertEqual(args.input_files, ["gemini-test.jsonl"])
            self.assertEqual(args.aux_input_files, ["qwen-test.jsonl"])
            self.assertEqual(args.calibration_input_files, ["gemini-validation.jsonl"])
            self.assertEqual(args.aux_calibration_input_files, ["qwen-validation.jsonl"])
            self.assertEqual(args.embedding_model_name, "sentence-transformers/all-mpnet-base-v2")
            self.assertEqual(args.neutral_field, "gemini25flashlite")
            self.assertEqual(args.aux_neutral_field, "qwen")
            self.assertEqual(args.score_granularity, "response")
            self.assertEqual(args.batch_size, 64)

    def test_saved_state_defaults_override_stale_setup_values(self) -> None:
        args = parse_args(["--setup-name", "setup111"])
        args.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        args.aux_neutral_field = "other"
        args.max_length = 256

        updated = apply_saved_state_defaults(
            args,
            [],
            {
                "embedding_model_name": "sentence-transformers/all-mpnet-base-v2",
                "aux_neutral_field": "qwen",
                "max_length": 512,
            },
        )

        self.assertEqual(updated.embedding_model_name, "sentence-transformers/all-mpnet-base-v2")
        self.assertEqual(updated.aux_neutral_field, "qwen")
        self.assertEqual(updated.max_length, 512)


if __name__ == "__main__":
    unittest.main()
