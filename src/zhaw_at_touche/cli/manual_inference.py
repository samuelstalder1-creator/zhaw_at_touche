from __future__ import annotations

import argparse
from pathlib import Path

from zhaw_at_touche.constants import DEFAULT_MODELS_DIR, DEFAULT_SETUP_NAME
from zhaw_at_touche.modeling import predict_records, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run manual single-example inference with a trained model."
    )
    parser.add_argument("--setup-name", default=DEFAULT_SETUP_NAME)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--query", default=None, help="Optional query text.")
    parser.add_argument("--response", default=None, help="Optional response text.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    return parser


def run_single_prediction(
    *,
    model_dir: Path,
    query: str,
    response: str,
    batch_size: int,
    max_length: int,
    threshold: float,
    device: str,
) -> None:
    prediction = predict_records(
        model_dir=model_dir,
        records=[{"query": query, "response": response}],
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        text_key="response",
        threshold=threshold,
    )[0]
    print(f"label={prediction.label} ad_prob={prediction.ad_prob:.4f}")


def interactive_loop(
    *,
    model_dir: Path,
    batch_size: int,
    max_length: int,
    threshold: float,
    device: str,
) -> None:
    print("Interactive mode. Submit an empty response to exit.")
    while True:
        query = input("query> ").strip()
        response = input("response> ").strip()
        if not response:
            break
        run_single_prediction(
            model_dir=model_dir,
            query=query,
            response=response,
            batch_size=batch_size,
            max_length=max_length,
            threshold=threshold,
            device=device,
        )


def main() -> None:
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_MODELS_DIR / args.setup_name
    device = resolve_device(args.device)

    if args.response is not None:
        run_single_prediction(
            model_dir=model_dir,
            query=args.query or "",
            response=args.response,
            batch_size=args.batch_size,
            max_length=args.max_length,
            threshold=args.threshold,
            device=device,
        )
        return

    interactive_loop(
        model_dir=model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        threshold=args.threshold,
        device=device,
    )


if __name__ == "__main__":
    main()
