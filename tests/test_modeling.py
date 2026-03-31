from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from zhaw_at_touche import modeling


class TokenizerLoadingTests(unittest.TestCase):
    def test_load_tokenizer_wraps_missing_backend_errors(self) -> None:
        backend_errors = [
            ImportError(
                "SentencePieceExtractor requires the SentencePiece library but it was not found "
                "in your environment."
            ),
            ValueError(
                "`tiktoken` is required to read a `tiktoken` file. Install it with "
                "`pip install tiktoken`."
            ),
        ]

        for error in backend_errors:
            with self.subTest(error=type(error).__name__):
                with patch.object(modeling.AutoTokenizer, "from_pretrained", side_effect=error):
                    with self.assertRaisesRegex(RuntimeError, "uv sync"):
                        modeling.load_tokenizer_from_pretrained("microsoft/deberta-v3-base")

    def test_load_tokenizer_preserves_unrelated_errors(self) -> None:
        with patch.object(
            modeling.AutoTokenizer,
            "from_pretrained",
            side_effect=ValueError("Tokenizer config is invalid."),
        ):
            with self.assertRaisesRegex(ValueError, "Tokenizer config is invalid."):
                modeling.load_tokenizer_from_pretrained("broken/model")


class FakeEncoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)


class FakeBaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(32, 4)
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList([FakeEncoderLayer(), FakeEncoderLayer()])


class FakeClassifierModel(torch.nn.Module):
    base_model_prefix = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.fake = FakeBaseModel()
        self.classifier = torch.nn.Linear(4, 2)


class OptimizerHelpersTests(unittest.TestCase):
    def test_optimizer_param_groups_apply_layerwise_decay_and_weight_decay(self) -> None:
        model = FakeClassifierModel()
        parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}

        groups = modeling.optimizer_param_groups(
            model,
            learning_rate=1e-3,
            weight_decay=0.01,
            layerwise_lr_decay=0.9,
        )

        group_by_name: dict[str, tuple[float, float]] = {}
        for group in groups:
            for parameter in group["params"]:
                group_by_name[parameter_names[id(parameter)]] = (
                    group["lr"],
                    group["weight_decay"],
                )

        self.assertEqual(group_by_name["classifier.weight"], (1e-3, 0.01))
        self.assertEqual(group_by_name["classifier.bias"], (1e-3, 0.0))
        self.assertEqual(group_by_name["fake.encoder.layer.1.linear.weight"], (1e-3, 0.01))
        self.assertAlmostEqual(group_by_name["fake.encoder.layer.0.linear.weight"][0], 9e-4)
        self.assertEqual(group_by_name["fake.encoder.layer.0.linear.weight"][1], 0.01)
        self.assertAlmostEqual(group_by_name["fake.embeddings.weight"][0], 8.1e-4)
        self.assertEqual(group_by_name["fake.embeddings.weight"][1], 0.01)
        self.assertAlmostEqual(group_by_name["fake.encoder.layer.0.norm.weight"][0], 9e-4)
        self.assertEqual(group_by_name["fake.encoder.layer.0.norm.weight"][1], 0.0)

    def test_set_embeddings_trainable_toggles_requires_grad(self) -> None:
        model = FakeClassifierModel()

        self.assertTrue(all(parameter.requires_grad for parameter in model.fake.embeddings.parameters()))
        self.assertTrue(modeling.set_embeddings_trainable(model, trainable=False))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.fake.embeddings.parameters()))
        self.assertTrue(modeling.set_embeddings_trainable(model, trainable=True))
        self.assertTrue(all(parameter.requires_grad for parameter in model.fake.embeddings.parameters()))


if __name__ == "__main__":
    unittest.main()
