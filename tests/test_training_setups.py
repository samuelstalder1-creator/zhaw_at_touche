from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from zhaw_at_touche.cli.train_model import parse_args
from zhaw_at_touche.training_setups import load_setup_defaults


class TrainingSetupsTests(unittest.TestCase):
    def test_load_setup_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "setup6.json").write_text(
                json.dumps(
                    {
                        "description": "Merged Dagmar defaults.",
                        "epochs": 3,
                        "batch_size": 8,
                    }
                ),
                encoding="utf-8",
            )

            defaults = load_setup_defaults("setup6", setups_dir)

            self.assertEqual(defaults["epochs"], 3)
            self.assertEqual(defaults["batch_size"], 8)

    def test_parse_args_uses_setup_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "setup6.json").write_text(
                json.dumps(
                    {
                        "epochs": 3,
                        "batch_size": 8,
                        "train_file": "data/custom-train.jsonl",
                        "max_train_rows": 100,
                        "input_format": "query_reference_rag_response",
                        "reference_field": "gemini25flashlite",
                        "reference_label": "Unbiased Reference",
                        "validation_file": "data/custom-validation.jsonl",
                        "optimizer_eps": 1e-6,
                        "weight_decay": 0.01,
                        "lr_scheduler": "cosine_with_warmup",
                        "warmup_ratio": 0.05,
                        "max_grad_norm": 1.0,
                        "gradient_checkpointing": True,
                        "layerwise_lr_decay": 0.9,
                        "freeze_embeddings_epochs": 1,
                        "wandb_enabled": False,
                        "wandb_project": "local-test",
                    }
                ),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--setup-name",
                    "setup6",
                    "--setups-dir",
                    str(setups_dir),
                ]
            )

            self.assertEqual(args.epochs, 3)
            self.assertEqual(args.batch_size, 8)
            self.assertEqual(args.train_file, "data/custom-train.jsonl")
            self.assertEqual(args.max_train_rows, 100)
            self.assertEqual(args.input_format, "query_reference_rag_response")
            self.assertEqual(args.reference_field, "gemini25flashlite")
            self.assertEqual(args.reference_label, "Unbiased Reference")
            self.assertEqual(args.validation_file, "data/custom-validation.jsonl")
            self.assertEqual(args.optimizer_eps, 1e-6)
            self.assertEqual(args.weight_decay, 0.01)
            self.assertEqual(args.lr_scheduler, "cosine_with_warmup")
            self.assertEqual(args.warmup_ratio, 0.05)
            self.assertEqual(args.max_grad_norm, 1.0)
            self.assertTrue(args.gradient_checkpointing)
            self.assertEqual(args.layerwise_lr_decay, 0.9)
            self.assertEqual(args.freeze_embeddings_epochs, 1)
            self.assertFalse(args.wandb)
            self.assertEqual(args.wandb_project, "local-test")

    def test_parse_args_allows_cli_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "setup6.json").write_text(
                json.dumps({"epochs": 3, "batch_size": 8}),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--setup-name",
                    "setup6",
                    "--setups-dir",
                    str(setups_dir),
                    "--epochs",
                    "7",
                ]
            )

            self.assertEqual(args.epochs, 7)
            self.assertEqual(args.batch_size, 8)

    def test_parse_args_defaults_to_full_training_set(self) -> None:
        args = parse_args([])

        self.assertIsNone(args.max_train_rows)
        self.assertTrue(args.wandb)

    def test_repo_setup6_qwen_uses_qwen_generated_files(self) -> None:
        args = parse_args(["--setup-name", "setup6-qwen"])

        self.assertEqual(args.train_file, "data/generated/qwen/responses-train-with-neutral_qwen.jsonl")
        self.assertEqual(
            args.validation_file,
            "data/generated/qwen/responses-validation-with-neutral_qwen.jsonl",
        )
        self.assertEqual(args.model_name, "FacebookAI/roberta-base")

    def test_repo_setup7_qwen_uses_qwen_neutral_reference(self) -> None:
        args = parse_args(["--setup-name", "setup7-qwen"])

        self.assertEqual(args.train_file, "data/generated/qwen/responses-train-with-neutral_qwen.jsonl")
        self.assertEqual(
            args.validation_file,
            "data/generated/qwen/responses-validation-with-neutral_qwen.jsonl",
        )
        self.assertEqual(args.model_name, "allenai/longformer-base-4096")
        self.assertEqual(args.input_format, "query_neutral_response")
        self.assertEqual(args.reference_field, "qwen")
        self.assertEqual(args.reference_label, "QWEN")
        self.assertTrue(args.pad_to_max_length)

    def test_repo_setup104_qwen_uses_qwen_generated_files(self) -> None:
        args = parse_args(["--setup-name", "setup104-qwen"])

        self.assertEqual(args.trainer_type, "embedding_classifier")
        self.assertEqual(args.train_file, "data/generated/qwen/responses-train-with-neutral_qwen.jsonl")
        self.assertEqual(
            args.validation_file,
            "data/generated/qwen/responses-validation-with-neutral_qwen.jsonl",
        )
        self.assertEqual(args.model_name, "sentence-transformers/all-mpnet-base-v2")
        self.assertEqual(args.neutral_field, "qwen")

    def test_repo_setup103_qwen_uses_qwen_generated_files(self) -> None:
        args = parse_args(["--setup-name", "setup103-qwen"])

        self.assertEqual(args.trainer_type, "embedding_residual_classifier")
        self.assertEqual(args.train_file, "data/generated/qwen/responses-train-with-neutral_qwen.jsonl")
        self.assertEqual(
            args.validation_file,
            "data/generated/qwen/responses-validation-with-neutral_qwen.jsonl",
        )
        self.assertEqual(args.model_name, "sentence-transformers/all-mpnet-base-v2")
        self.assertEqual(args.neutral_field, "qwen")

    def test_existing_embedding_lr_setups_keep_legacy_defaults(self) -> None:
        residual_args = parse_args(["--setup-name", "setup103"])
        full_args = parse_args(["--setup-name", "setup104"])

        for args in (residual_args, full_args):
            self.assertEqual(args.delta_centering, "none")
            self.assertFalse(args.append_delta_abs)
            self.assertFalse(args.append_pairwise_cosine)
            self.assertFalse(args.append_delta_norm)
            self.assertEqual(args.lr_c_values, [1.0])
            self.assertEqual(args.lr_class_weight_options, ["balanced"])

    def test_repo_setup103_gemma_uses_gemma_generated_files(self) -> None:
        args = parse_args(["--setup-name", "setup103-gemma"])

        self.assertEqual(args.trainer_type, "embedding_residual_classifier")
        self.assertEqual(args.train_file, "data/generated/gemma4e4b/responses-train-with-neutral_gemma4e4b.jsonl")
        self.assertEqual(
            args.validation_file,
            "data/generated/gemma4e4b/responses-validation-with-neutral_gemma4e4b.jsonl",
        )
        self.assertEqual(args.model_name, "sentence-transformers/all-mpnet-base-v2")
        self.assertEqual(args.neutral_field, "gemma4_e4b")

    def test_repo_setup120_qwen_enables_centered_delta_feature_search(self) -> None:
        args = parse_args(["--setup-name", "setup120-qwen"])

        self.assertEqual(args.trainer_type, "embedding_residual_classifier")
        self.assertEqual(args.train_file, "data/generated/qwen/responses-train-with-neutral_qwen.jsonl")
        self.assertEqual(args.neutral_field, "qwen")
        self.assertEqual(args.delta_centering, "negative_mean")
        self.assertTrue(args.append_delta_abs)
        self.assertTrue(args.append_pairwise_cosine)
        self.assertTrue(args.append_delta_norm)
        self.assertEqual(args.lr_c_values, [0.25, 0.5, 1.0, 2.0, 4.0])
        self.assertEqual(args.lr_class_weight_options, ["none", "balanced"])

    def test_repo_setup121_gemma_enables_centered_full_embedding_search(self) -> None:
        args = parse_args(["--setup-name", "setup121-gemma"])

        self.assertEqual(args.trainer_type, "embedding_classifier")
        self.assertEqual(args.train_file, "data/generated/gemma4e4b/responses-train-with-neutral_gemma4e4b.jsonl")
        self.assertEqual(args.neutral_field, "gemma4_e4b")
        self.assertEqual(args.delta_centering, "negative_mean")
        self.assertTrue(args.append_delta_abs)
        self.assertTrue(args.append_pairwise_cosine)
        self.assertTrue(args.append_delta_norm)
        self.assertEqual(args.lr_c_values, [0.25, 0.5, 1.0, 2.0, 4.0])
        self.assertEqual(args.lr_class_weight_options, ["none", "balanced"])

    def test_parse_args_accepts_linear_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "setup10.json").write_text(
                json.dumps(
                    {
                        "model_name": "albert/albert-base-v2",
                        "lr_scheduler": "linear",
                        "warmup_ratio": 0.06,
                    }
                ),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--setup-name",
                    "setup10",
                    "--setups-dir",
                    str(setups_dir),
                ]
            )

            self.assertEqual(args.model_name, "albert/albert-base-v2")
            self.assertEqual(args.lr_scheduler, "linear")
            self.assertEqual(args.warmup_ratio, 0.06)

    def test_parse_args_supports_embedding_divergence_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "setup100.json").write_text(
                json.dumps(
                    {
                        "trainer_type": "embedding_divergence",
                        "train_file": "data/generated/gemini/responses-train-with-neutral_gemini.jsonl",
                        "validation_file": "data/generated/gemini/responses-validation-with-neutral_gemini.jsonl",
                        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                        "neutral_field": "gemini25flashlite",
                        "distance_metric": "cosine",
                        "score_granularity": "sentence",
                        "sentence_agg": "mean",
                        "threshold_metric": "macro_f1",
                        "batch_size": 32,
                        "max_length": 256,
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

            self.assertEqual(args.trainer_type, "embedding_divergence")
            self.assertEqual(args.model_name, "sentence-transformers/all-MiniLM-L6-v2")
            self.assertEqual(args.neutral_field, "gemini25flashlite")
            self.assertEqual(args.distance_metric, "cosine")
            self.assertEqual(args.score_granularity, "sentence")
            self.assertEqual(args.sentence_agg, "mean")
            self.assertEqual(args.threshold_metric, "macro_f1")
            self.assertEqual(args.batch_size, 32)
            self.assertEqual(args.max_length, 256)

    def test_parse_args_supports_topk_embedding_divergence_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "setup101.json").write_text(
                json.dumps(
                    {
                        "trainer_type": "embedding_divergence",
                        "train_file": "data/generated/gemini/responses-train-with-neutral_gemini.jsonl",
                        "validation_file": "data/generated/gemini/responses-validation-with-neutral_gemini.jsonl",
                        "model_name": "sentence-transformers/all-mpnet-base-v2",
                        "neutral_field": "gemini25flashlite",
                        "distance_metric": "cosine",
                        "score_granularity": "sentence",
                        "sentence_agg": "top3_mean",
                        "threshold_metric": "positive_f1",
                        "batch_size": 32,
                        "max_length": 384,
                    }
                ),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--setup-name",
                    "setup101",
                    "--setups-dir",
                    str(setups_dir),
                ]
            )

            self.assertEqual(args.trainer_type, "embedding_divergence")
            self.assertEqual(args.model_name, "sentence-transformers/all-mpnet-base-v2")
            self.assertEqual(args.sentence_agg, "top3_mean")
            self.assertEqual(args.threshold_metric, "positive_f1")
            self.assertEqual(args.max_length, 384)

    def test_repo_setup110_uses_anchor_distance_training_files(self) -> None:
        args = parse_args(["--setup-name", "setup110"])

        self.assertEqual(args.trainer_type, "anchor_distance_classifier")
        self.assertEqual(args.train_file, "data/generated/gemini/responses-train-with-neutral_gemini.jsonl")
        self.assertEqual(args.aux_train_file, "data/generated/qwen/responses-train-with-neutral_qwen.jsonl")
        self.assertEqual(
            args.validation_file,
            "data/generated/gemini/responses-validation-with-neutral_gemini.jsonl",
        )
        self.assertEqual(
            args.aux_validation_file,
            "data/generated/qwen/responses-validation-with-neutral_qwen.jsonl",
        )
        self.assertEqual(args.query_field, "query")
        self.assertEqual(args.response_field, "response")
        self.assertEqual(args.neutral_field, "gemini25flashlite")
        self.assertEqual(args.aux_neutral_field, "qwen")
        self.assertEqual(args.score_granularity, "response")

    def test_repo_setup111_uses_anchor_distance_threshold_training_files(self) -> None:
        args = parse_args(["--setup-name", "setup111"])

        self.assertEqual(args.trainer_type, "anchor_distance_threshold")
        self.assertEqual(args.train_file, "data/generated/gemini/responses-train-with-neutral_gemini.jsonl")
        self.assertEqual(args.aux_train_file, "data/generated/qwen/responses-train-with-neutral_qwen.jsonl")
        self.assertEqual(
            args.validation_file,
            "data/generated/gemini/responses-validation-with-neutral_gemini.jsonl",
        )
        self.assertEqual(
            args.aux_validation_file,
            "data/generated/qwen/responses-validation-with-neutral_qwen.jsonl",
        )
        self.assertEqual(args.query_field, "query")
        self.assertEqual(args.response_field, "response")
        self.assertEqual(args.neutral_field, "gemini25flashlite")
        self.assertEqual(args.aux_neutral_field, "qwen")
        self.assertEqual(args.score_granularity, "response")


if __name__ == "__main__":
    unittest.main()
