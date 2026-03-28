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


if __name__ == "__main__":
    unittest.main()
