from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from zhaw_at_touche.cli.embedding_divergence import parse_args
from zhaw_at_touche.embedding_divergence import (
    aggregate_sentence_distances,
    calibrate_threshold,
    greedy_sentence_alignment,
    split_sentences,
)


class EmbeddingDivergenceHelpersTests(unittest.TestCase):
    def test_split_sentences_handles_punctuation_and_newlines(self) -> None:
        text = "First sentence. Second sentence?\nThird sentence!"
        self.assertEqual(
            split_sentences(text),
            ["First sentence.", "Second sentence?", "Third sentence!"],
        )

    def test_greedy_sentence_alignment_marks_unmatched_response_sentences(self) -> None:
        response_sentences = ["kept sentence", "new ad sentence"]
        neutral_sentences = ["kept sentence"]
        response_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        neutral_embeddings = torch.tensor([[1.0, 0.0]])

        alignment = greedy_sentence_alignment(
            response_sentences=response_sentences,
            neutral_sentences=neutral_sentences,
            response_embeddings=response_embeddings,
            neutral_embeddings=neutral_embeddings,
        )

        self.assertEqual(alignment[0]["matched_neutral_sentence"], "kept sentence")
        self.assertAlmostEqual(float(alignment[0]["distance"]), 0.0)
        self.assertIsNone(alignment[1]["matched_neutral_sentence"])
        self.assertAlmostEqual(float(alignment[1]["distance"]), 1.0)
        self.assertAlmostEqual(aggregate_sentence_distances(alignment, "max"), 1.0)

    def test_calibrate_threshold_finds_perfect_split(self) -> None:
        threshold, summary = calibrate_threshold(
            scores=[0.1, 0.2, 0.8, 0.9],
            labels=[0, 0, 1, 1],
            threshold_metric="positive_f1",
        )

        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.8)
        self.assertEqual(summary["accuracy"], 1.0)


class EmbeddingDivergenceCliTests(unittest.TestCase):
    def test_parse_args_uses_setup_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "setup100.json").write_text(
                json.dumps(
                    {
                        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
                        "neutral_field": "gemini25flashlite",
                        "score_granularity": "sentence",
                        "sentence_agg": "max",
                        "threshold_metric": "positive_f1",
                        "batch_size": 64,
                    }
                ),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--setup-name",
                    "setup100",
                    "--setups-dir",
                    str(setups_dir),
                ]
            )

            self.assertEqual(args.embedding_model_name, "sentence-transformers/all-MiniLM-L6-v2")
            self.assertEqual(args.neutral_field, "gemini25flashlite")
            self.assertEqual(args.score_granularity, "sentence")
            self.assertEqual(args.sentence_agg, "max")
            self.assertEqual(args.threshold_metric, "positive_f1")
            self.assertEqual(args.batch_size, 64)


if __name__ == "__main__":
    unittest.main()
