from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from zhaw_at_touche.cli.validate_model import (
    parse_args,
    resolve_default_eval_paths,
    resolve_model_source,
    resolve_scoring_backend,
)
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
                        "input_format": "query_reference_rag_response",
                        "reference_field": "gemini25flashlite",
                        "reference_label": "Unbiased Reference",
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
            self.assertEqual(args.input_format, "query_reference_rag_response")
            self.assertEqual(args.reference_field, "gemini25flashlite")
            self.assertEqual(args.reference_label, "Unbiased Reference")

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

    def test_generated_provider_switches_default_eval_input_and_results_dir(self) -> None:
        args = parse_args(["--setup-name", "setup6", "--generated-provider", "qwen"])

        self.assertEqual(args.input_files, ["data/generated/qwen/responses-test-with-neutral_qwen.jsonl"])
        self.assertEqual(args.generated_field, "qwen")
        self.assertEqual(args.results_dir, "results/setup6-qwen")

    def test_generated_provider_updates_reference_aware_setup_defaults(self) -> None:
        args = parse_args(["--setup-name", "setup7", "--generated-provider", "qwen"])

        self.assertEqual(args.input_files, ["data/generated/qwen/responses-test-with-neutral_qwen.jsonl"])
        self.assertEqual(args.reference_field, "qwen")
        self.assertEqual(args.reference_label, "QWEN")
        self.assertEqual(args.results_dir, "results/setup7-qwen")

    def test_generated_provider_overrides_setup_input_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "custom.json").write_text(
                json.dumps(
                    {
                        "input_files": ["custom-a.jsonl"],
                        "results_dir": "results/custom",
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
                    "--generated-provider",
                    "qwen",
                ]
            )

            self.assertEqual(args.input_files, ["data/generated/qwen/responses-test-with-neutral_qwen.jsonl"])
            self.assertEqual(args.results_dir, "results/custom-qwen")

    def test_repo_setup6_qwen_uses_qwen_test_file(self) -> None:
        args = parse_args(["--setup-name", "setup6-qwen"])

        self.assertEqual(resolve_model_source(args), Path("models/setup6-qwen"))
        self.assertEqual(args.results_dir, "results/setup6-qwen")
        self.assertEqual(args.input_files, ["data/generated/qwen/responses-test-with-neutral_qwen.jsonl"])
        self.assertEqual(args.generated_field, "qwen")

    def test_repo_setup7_qwen_uses_qwen_test_file_and_reference(self) -> None:
        args = parse_args(["--setup-name", "setup7-qwen"])

        self.assertEqual(resolve_model_source(args), Path("models/setup7-qwen"))
        self.assertEqual(args.results_dir, "results/setup7-qwen")
        self.assertEqual(args.input_files, ["data/generated/qwen/responses-test-with-neutral_qwen.jsonl"])
        self.assertEqual(args.generated_field, "qwen")
        self.assertEqual(args.reference_field, "qwen")
        self.assertEqual(args.reference_label, "QWEN")
        self.assertTrue(args.pad_to_max_length)

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

    def test_load_setup_defaults_accepts_embedding_divergence_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "setup100.json").write_text(
                json.dumps(
                    {
                        "scoring_backend": "embedding_divergence",
                        "model_dir": "models/setup100",
                        "results_dir": "results/setup100",
                        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
                        "neutral_field": "gemini25flashlite",
                        "distance_metric": "cosine",
                        "score_granularity": "sentence",
                        "sentence_agg": "max",
                        "threshold_metric": "positive_f1",
                        "calibration_input_files": [
                            "data/generated/gemini/responses-validation-with-neutral_gemini.jsonl"
                        ],
                    }
                ),
                encoding="utf-8",
            )

            defaults = load_setup_defaults("setup100", setups_dir)

            self.assertEqual(defaults["scoring_backend"], "embedding_divergence")
            self.assertEqual(defaults["embedding_model_name"], "sentence-transformers/all-MiniLM-L6-v2")
            self.assertEqual(defaults["neutral_field"], "gemini25flashlite")

    def test_resolve_scoring_backend_detects_embedding_divergence_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "setup100.json").write_text(
                json.dumps(
                    {
                        "scoring_backend": "embedding_divergence",
                        "model_dir": "models/setup100",
                    }
                ),
                encoding="utf-8",
            )

            backend = resolve_scoring_backend(
                [
                    "--setup-name",
                    "setup100",
                    "--setups-dir",
                    str(setups_dir),
                ]
            )

            self.assertEqual(backend, "embedding_divergence")

    def test_load_setup_defaults_accepts_anchor_distance_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "setup110.json").write_text(
                json.dumps(
                    {
                        "scoring_backend": "anchor_distance_classifier",
                        "model_dir": "models/setup110",
                        "results_dir": "results/setup110",
                        "input_files": ["gemini-test.jsonl"],
                        "aux_input_files": ["qwen-test.jsonl"],
                        "calibration_input_files": ["gemini-validation.jsonl"],
                        "aux_calibration_input_files": ["qwen-validation.jsonl"],
                        "query_field": "query",
                        "response_field": "response",
                        "neutral_field": "gemini25flashlite",
                        "aux_neutral_field": "qwen",
                    }
                ),
                encoding="utf-8",
            )

            defaults = load_setup_defaults("setup110", setups_dir)

            self.assertEqual(defaults["scoring_backend"], "anchor_distance_classifier")
            self.assertEqual(defaults["aux_input_files"], ["qwen-test.jsonl"])
            self.assertEqual(defaults["aux_neutral_field"], "qwen")

    def test_resolve_scoring_backend_detects_anchor_distance_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "setup110.json").write_text(
                json.dumps(
                    {
                        "scoring_backend": "anchor_distance_classifier",
                        "model_dir": "models/setup110",
                    }
                ),
                encoding="utf-8",
            )

            backend = resolve_scoring_backend(
                [
                    "--setup-name",
                    "setup110",
                    "--setups-dir",
                    str(setups_dir),
                ]
            )

            self.assertEqual(backend, "anchor_distance_classifier")

    def test_resolve_scoring_backend_detects_anchor_distance_threshold_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setups_dir = Path(tmp_dir)
            (setups_dir / "setup111.json").write_text(
                json.dumps(
                    {
                        "scoring_backend": "anchor_distance_threshold",
                        "model_dir": "models/setup111",
                    }
                ),
                encoding="utf-8",
            )

            backend = resolve_scoring_backend(
                [
                    "--setup-name",
                    "setup111",
                    "--setups-dir",
                    str(setups_dir),
                ]
            )

            self.assertEqual(backend, "anchor_distance_threshold")


if __name__ == "__main__":
    unittest.main()
