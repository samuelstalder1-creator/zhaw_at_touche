from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from zhaw_at_touche.cli import train_model


class TrainModelCliTests(unittest.TestCase):
    def test_dual_neutral_validation_requires_aux_validation_file(self) -> None:
        args = SimpleNamespace(
            model_dir=None,
            setup_name="setup116",
            trainer_type="classifier",
            input_format="query_dual_neutral_response",
            train_file="train.jsonl",
            aux_train_file="aux-train.jsonl",
            validation_file="validation.jsonl",
            aux_validation_file=None,
        )

        with patch("zhaw_at_touche.cli.train_model.parse_args", return_value=args):
            with self.assertRaisesRegex(ValueError, "--aux-validation-file"):
                train_model.main()


if __name__ == "__main__":
    unittest.main()
