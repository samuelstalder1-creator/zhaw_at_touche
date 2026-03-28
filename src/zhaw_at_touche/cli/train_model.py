from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from zhaw_at_touche.constants import DEFAULT_MODELS_DIR, DEFAULT_SETUP_NAME
from zhaw_at_touche.datasets import DEFAULT_INPUT_FORMAT, SUPPORTED_INPUT_FORMATS
from zhaw_at_touche.training_setups import DEFAULT_TRAINING_SETUPS_DIR, load_setup_defaults


def resolve_default_train_path() -> Path:
    candidates = [
        Path("data/generated/gemini/responses-train-with-neutral_gemini.jsonl"),
        Path("data/task/preprocessed/responses-train-merged.jsonl"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_default_validation_path() -> Path | None:
    candidates = [
        Path("data/generated/gemini/responses-validation-with-neutral_gemini.jsonl"),
        Path("data/task/preprocessed/responses-validation-merged.jsonl"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def base_defaults() -> dict[str, object]:
    return {
        "train_file": str(resolve_default_train_path()),
        "model_name": "FacebookAI/roberta-base",
        "model_dir": None,
        "max_length": 512,
        "epochs": 5,
        "batch_size": 16,
        "grad_accum": 4,
        "learning_rate": 2e-5,
        "device": None,
        "max_train_rows": None,
        "input_format": DEFAULT_INPUT_FORMAT,
        "reference_field": None,
        "reference_label": "GEMINI",
        "pad_to_max_length": False,
        "positive_class_weight_scale": 2.0,
        "validation_file": str(resolve_default_validation_path()) if resolve_default_validation_path() else None,
        "tensorboard_enabled": True,
        "tensorboard_dir": None,
        "wandb_enabled": True,
        "wandb_project": "zhaw-at-touche-training",
        "wandb_dir": None,
        "wandb_run_name": None,
    }


def build_parser(setup_defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    defaults = base_defaults()
    if setup_defaults:
        defaults.update(setup_defaults)

    parser = argparse.ArgumentParser(description="Train the binary ad classifier.")
    parser.add_argument("--setup-name", default=DEFAULT_SETUP_NAME)
    parser.add_argument(
        "--setups-dir",
        default=str(DEFAULT_TRAINING_SETUPS_DIR),
        help="Directory containing optional <setup-name>.json training defaults.",
    )
    parser.add_argument("--train-file", default=defaults["train_file"])
    parser.add_argument("--model-name", default=defaults["model_name"])
    parser.add_argument(
        "--model-dir",
        default=defaults["model_dir"],
        help="Directory where the trained model will be stored.",
    )
    parser.add_argument("--max-length", type=int, default=defaults["max_length"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=defaults["grad_accum"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=defaults["device"])
    parser.add_argument(
        "--validation-file",
        default=defaults["validation_file"],
        help="Optional validation JSONL file used for epoch-end monitoring.",
    )
    parser.add_argument("--input-format", choices=SUPPORTED_INPUT_FORMATS, default=defaults["input_format"])
    parser.add_argument(
        "--reference-field",
        default=defaults["reference_field"],
        help="Optional reference text field used by non-default input formats.",
    )
    parser.add_argument(
        "--reference-label",
        default=defaults["reference_label"],
        help="Label text rendered in reference-aware input formats.",
    )
    parser.add_argument(
        "--pad-to-max-length",
        action=argparse.BooleanOptionalAction,
        default=defaults["pad_to_max_length"],
        help="Pad every batch item to max_length instead of dynamic padding.",
    )
    parser.add_argument(
        "--positive-class-weight-scale",
        type=float,
        default=defaults["positive_class_weight_scale"],
        help="Multiplier used when computing the positive-class loss weight.",
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=defaults["tensorboard_enabled"],
        help="Enable local TensorBoard logging.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        default=defaults["tensorboard_dir"],
        help="Directory for TensorBoard event files. Defaults to <model-dir>/tensorboard.",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=defaults["wandb_enabled"],
        help="Enable local offline Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        default=defaults["wandb_project"],
        help="Project name used for offline Weights & Biases runs.",
    )
    parser.add_argument(
        "--wandb-dir",
        default=defaults["wandb_dir"],
        help="Directory for local offline Weights & Biases files. Defaults to <model-dir>/wandb.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=defaults["wandb_run_name"],
        help="Optional offline Weights & Biases run name.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=defaults["max_train_rows"],
        help="Optional limit for training rows. By default the full training file is used.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--setup-name", default=DEFAULT_SETUP_NAME)
    pre_parser.add_argument("--setups-dir", default=str(DEFAULT_TRAINING_SETUPS_DIR))
    pre_args, _ = pre_parser.parse_known_args(argv)

    setup_defaults = load_setup_defaults(
        setup_name=pre_args.setup_name,
        setups_dir=Path(pre_args.setups_dir),
    )
    parser = build_parser(setup_defaults)
    return parser.parse_args(argv)


def main() -> None:
    from zhaw_at_touche.modeling import TrainingConfig, resolve_device, train_model

    args = parse_args()
    model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_MODELS_DIR / args.setup_name
    config = TrainingConfig(
        model_name=args.model_name,
        train_path=Path(args.train_file),
        output_dir=model_dir,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        device=resolve_device(args.device),
        max_train_rows=args.max_train_rows,
        input_format=args.input_format,
        reference_field=args.reference_field,
        reference_label=args.reference_label,
        pad_to_max_length=args.pad_to_max_length,
        positive_class_weight_scale=args.positive_class_weight_scale,
        validation_path=Path(args.validation_file) if args.validation_file else None,
        tensorboard_enabled=args.tensorboard,
        tensorboard_dir=Path(args.tensorboard_dir) if args.tensorboard_dir else None,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_dir=Path(args.wandb_dir) if args.wandb_dir else None,
        wandb_run_name=args.wandb_run_name or args.setup_name,
    )
    summary = train_model(config)
    print(f"trained model saved to {model_dir}")
    print(f"training rows: {summary['train_rows']}")
    print(f"device: {summary['device']}")


if __name__ == "__main__":
    main()
