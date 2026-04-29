from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from zhaw_at_touche.cli import embedding_lr_classifier as embedding_lr_cli
from zhaw_at_touche.embedding_lr_classifier import (
    EmbeddingLRConfig,
    EmbeddingLRPrediction,
    train_embedding_lr_classifier,
)
from zhaw_at_touche.evaluation_utils import metrics_dict


class FakePipeline:
    def __init__(self, _steps) -> None:
        self.named_steps = {"classifier": SimpleNamespace(intercept_=np.array([0.25]))}

    def fit(self, _features, _labels) -> None:
        return None

    def predict_proba(self, feature_matrix):
        first_value = float(feature_matrix[0][0])
        if first_value < 5.0:
            scores = [0.55, 0.35]
        else:
            scores = [0.60, 0.70, 0.40]
        return np.array([[1.0 - score, score] for score in scores], dtype=float)


class EmbeddingLRTrainerTests(unittest.TestCase):
    def test_validation_summary_uses_calibrated_threshold(self) -> None:
        train_records = [{"label": 1}, {"label": 0}]
        validation_records = [{"label": 1}, {"label": 0}, {"label": 0}]
        calibrated_summary = metrics_dict([1, 0, 0], [0, 1, 0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = EmbeddingLRConfig(
                trainer_type="embedding_residual_classifier",
                embedding_model_name="sentence-transformers/all-mpnet-base-v2",
                train_path=Path("train.jsonl"),
                validation_path=Path("validation.jsonl"),
                output_dir=Path(tmp_dir),
                max_length=32,
                batch_size=4,
                device="cpu",
                response_field="response",
                neutral_field="gemini25flashlite",
                threshold_metric="macro_f1",
            )

            with (
                patch("zhaw_at_touche.embedding_lr_classifier.load_embedding_model", return_value=("tok", "model")),
                patch(
                    "zhaw_at_touche.embedding_lr_classifier._load_records",
                    side_effect=[train_records, validation_records],
                ),
                patch(
                    "zhaw_at_touche.embedding_lr_classifier._embed_and_build",
                    side_effect=[
                        (["delta"], np.array([[1.0], [2.0]], dtype=float), {}),
                        (["delta"], np.array([[10.0], [20.0], [30.0]], dtype=float), {}),
                    ],
                ),
                patch("zhaw_at_touche.embedding_lr_classifier.Pipeline", FakePipeline),
                patch(
                    "zhaw_at_touche.embedding_lr_classifier.calibrate_threshold",
                    return_value=(0.65, calibrated_summary),
                ),
            ):
                summary = train_embedding_lr_classifier(config)

        expected = calibrated_summary
        self.assertEqual(summary["threshold"], 0.65)
        self.assertEqual(summary["validation_summary"], expected)


class EmbeddingLRValidationCliTests(unittest.TestCase):
    def test_metrics_summary_preserves_eval_and_calibration_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            model_dir = Path(tmp_dir) / "model"
            args = SimpleNamespace(
                setup_name="setup103",
                results_dir=str(results_dir),
                model_dir=str(model_dir),
                input_files=["eval.jsonl"],
                aux_input_files=None,
                calibration_input_files=["calibration.jsonl"],
                aux_calibration_input_files=None,
                embedding_model_name="sentence-transformers/all-mpnet-base-v2",
                response_field="response",
                neutral_field="gemini25flashlite",
                aux_neutral_field=None,
                query_field="query",
                threshold=None,
                threshold_metric="macro_f1",
                batch_size=8,
                max_length=128,
                device="cpu",
            )
            saved_state = {
                "trainer_type": "embedding_residual_classifier",
                "threshold": 0.61,
                "threshold_summary": {"metric": "macro_f1"},
                "embedding_model_name": args.embedding_model_name,
                "response_field": args.response_field,
                "neutral_field": args.neutral_field,
                "query_field": args.query_field,
                "threshold_metric": args.threshold_metric,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
            }
            records = [{
                "id": "row-1",
                "query": "query",
                "response": "response",
                "gemini25flashlite": "neutral",
                "label": 1,
            }]

            with (
                patch("zhaw_at_touche.cli.embedding_lr_classifier.parse_args", return_value=args),
                patch("zhaw_at_touche.cli.embedding_lr_classifier.resolve_device", return_value="cpu"),
                patch("zhaw_at_touche.cli.embedding_lr_classifier.load_state", return_value=saved_state),
                patch("zhaw_at_touche.cli.embedding_lr_classifier.load_bundle", return_value=object()),
                patch(
                    "zhaw_at_touche.cli.embedding_lr_classifier.load_embedding_model",
                    return_value=("tok", "model"),
                ),
                patch(
                    "zhaw_at_touche.cli.embedding_lr_classifier._load_records_for_eval",
                    return_value=records,
                ),
                patch(
                    "zhaw_at_touche.cli.embedding_lr_classifier.score_records",
                    return_value=[EmbeddingLRPrediction(label=1, score=0.9)],
                ),
                patch("zhaw_at_touche.cli.embedding_lr_classifier.save_confusion_matrix_image", return_value=None),
            ):
                embedding_lr_cli.main([])

            payload = json.loads((results_dir / "metrics_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["input_files"], ["eval.jsonl"])
            self.assertEqual(payload["calibration_input_files"], ["calibration.jsonl"])
            self.assertEqual(payload["aux_input_files"], [])
            self.assertEqual(payload["aux_calibration_input_files"], [])


if __name__ == "__main__":
    unittest.main()
