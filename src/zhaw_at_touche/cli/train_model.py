from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from zhaw_at_touche.constants import DEFAULT_MODELS_DIR, DEFAULT_SETUP_NAME
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
    )
    summary = train_model(config)
    print(f"trained model saved to {model_dir}")
    print(f"training rows: {summary['train_rows']}")
    print(f"device: {summary['device']}")


if __name__ == "__main__":
    main()
