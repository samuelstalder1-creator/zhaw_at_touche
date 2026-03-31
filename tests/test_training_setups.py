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


if __name__ == "__main__":
    unittest.main()
