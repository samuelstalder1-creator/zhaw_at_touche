from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from zhaw_at_touche.anchor_distance_classifier import (
    build_feature_columns,
    build_feature_rows,
    feature_pairs,
    pair_feature_name,
)
from zhaw_at_touche.cli.anchor_distance_classifier import (
    apply_saved_state_defaults,
    parse_args,
)


class AnchorDistanceClassifierHelpersTests(unittest.TestCase):
    def test_build_feature_columns_uses_pairwise_cosine_distance(self) -> None:
        embeddings_by_field = {
            "query": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "response": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            "gemini25flashlite": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "qwen": torch.tensor([[2**-0.5, 2**-0.5]], dtype=torch.float32),
        }
        pairs = feature_pairs(
            query_field="query",
            response_field="response",
            neutral_field="gemini25flashlite",
            aux_neutral_field="qwen",
        )

        columns = build_feature_columns(embeddings_by_field=embeddings_by_field, pairs=pairs)
        feature_names = [pair_feature_name(left, right) for left, right in pairs]
        rows = build_feature_rows(feature_columns=columns, feature_names=feature_names)

        self.assertEqual(
            feature_names,
            [
                "query__gemini25flashlite_distance",
                "query__qwen_distance",
                "gemini25flashlite__qwen_distance",
                "query__response_distance",
                "gemini25flashlite__response_distance",
                "qwen__response_distance",
            ],
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][0], 0.0)
        self.assertAlmostEqual(rows[0][1], 1.0 - (2**-0.5), places=6)
        self.assertAlmostEqual(rows[0][2], 1.0 - (2**-0.5), places=6)
        self.assertAlmostEqual(rows[0][3], 1.0)
        self.assertAlmostEqual(rows[0][4], 1.0)
        self.assertAlmostEqual(rows[0][5], 1.0 - (2**-0.5), places=6)


class AnchorDistanceClassifierCliTests(unittest.TestCase):
    def test_parse_args_uses_setup_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "setup110.json").write_text(
                json.dumps(
                    {
                        "scoring_backend": "anchor_distance_classifier",
                        "model_dir": "models/setup110",
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
                    "setup110",
                    "--setups-dir",
                    str(setups_dir),
                ]
            )

            self.assertEqual(args.model_dir, "models/setup110")
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
        args = parse_args(["--setup-name", "setup110"])
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
