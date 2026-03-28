from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from zhaw_at_touche.cli.validate_model import parse_args, resolve_default_eval_paths, resolve_model_source
from zhaw_at_touche.validation_setups import load_setup_defaults


class ValidationSetupsTests(unittest.TestCase):
    def test_load_setup_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "teamCMU.json").write_text(
                json.dumps(
                    {
                        "description": "Remote evaluation preset.",
                        "model_name": "teknology/ad-classifier-v0.4",
                        "batch_size": 32,
                    }
                ),
                encoding="utf-8",
            )

            defaults = load_setup_defaults("teamCMU", setups_dir)

            self.assertEqual(defaults["model_name"], "teknology/ad-classifier-v0.4")
            self.assertEqual(defaults["batch_size"], 32)

    def test_parse_args_uses_setup_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "teamCMU.json").write_text(
                json.dumps(
                    {
                        "model_name": "teknology/ad-classifier-v0.4",
                        "results_dir": "results/teamCMU",
                        "batch_size": 32,
                        "input_format": "query_neutral_response",
                        "reference_field": "gemini25flashlite",
                    }
                ),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--setup-name",
                    "teamCMU",
                    "--setups-dir",
                    str(setups_dir),
                ]
            )

            self.assertEqual(args.model_name, "teknology/ad-classifier-v0.4")
            self.assertEqual(args.results_dir, "results/teamCMU")
            self.assertEqual(args.batch_size, 32)
            self.assertEqual(args.eval_splits, ["test"])
            self.assertEqual(args.input_format, "query_neutral_response")
            self.assertEqual(args.reference_field, "gemini25flashlite")

    def test_resolve_model_source_prefers_model_name(self) -> None:
        args = parse_args(
            [
                "--setup-name",
                "teamCMU",
                "--model-name",
                "teknology/ad-classifier-v0.4",
            ]
        )

        self.assertEqual(resolve_model_source(args), "teknology/ad-classifier-v0.4")

    def test_cli_model_dir_overrides_setup_model_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "teamCMU.json").write_text(
                json.dumps(
                    {
                        "model_name": "teknology/ad-classifier-v0.4",
                    }
                ),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--setup-name",
                    "teamCMU",
                    "--setups-dir",
                    str(setups_dir),
                    "--model-dir",
                    "models/local-eval",
                ]
            )

            self.assertEqual(resolve_model_source(args), Path("models/local-eval"))

    def test_default_eval_uses_test_split_only(self) -> None:
        args = parse_args([])

        self.assertEqual(args.eval_splits, ["test"])
        self.assertEqual(
            args.input_files,
            [str(path) for path in resolve_default_eval_paths(["test"])],
        )

    def test_eval_splits_can_include_validation_and_test(self) -> None:
        args = parse_args(["--eval-splits", "validation", "test"])

        self.assertEqual(args.eval_splits, ["validation", "test"])
        self.assertEqual(
            args.input_files,
            [str(path) for path in resolve_default_eval_paths(["validation", "test"])],
        )

    def test_setup_input_files_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "custom.json").write_text(
                json.dumps(
                    {
                        "input_files": ["custom-a.jsonl", "custom-b.jsonl"],
                    }
                ),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--setup-name",
                    "custom",
                    "--setups-dir",
                    str(setups_dir),
                ]
            )

            self.assertEqual(args.input_files, ["custom-a.jsonl", "custom-b.jsonl"])


if __name__ == "__main__":
    unittest.main()
