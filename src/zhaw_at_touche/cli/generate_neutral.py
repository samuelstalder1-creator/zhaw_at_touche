from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from zhaw_at_touche.constants import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_QWEN_API_BASE,
    DEFAULT_QWEN_MODEL,
    DEFAULT_TASK_DIR,
)
from zhaw_at_touche.datasets import load_label_map, merge_response_row
from zhaw_at_touche.generation_utils import (
    clean_response_text,
    default_backend_for_provider,
    generate_neutral_response_gemini,
    generate_neutral_response_openai_compatible,
    generate_neutral_response_transformers,
    load_local_generation_model,
    load_done_ids,
    model_alias,
    with_retries,
)
from zhaw_at_touche.jsonl import append_jsonl, read_jsonl
from zhaw_at_touche.modeling import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate neutral baseline responses and write enriched JSONL output."
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "validation", "test"),
        help="Dataset split to process when --responses/--labels/--out are not passed.",
    )
    parser.add_argument("--responses", default=None, help="Path to the responses JSONL file.")
    parser.add_argument("--labels", default=None, help="Path to the response labels JSONL file.")
    parser.add_argument("--out", default=None, help="Output JSONL path.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="Subdirectory name under data/generated.")
    parser.add_argument(
        "--backend",
        choices=("auto", "gemini", "transformers", "openai_compatible"),
        default="auto",
        help="Generation backend. Defaults to provider-specific auto resolution.",
    )
    parser.add_argument("--model", default=None, help="Model name. Defaults depend on --provider.")
    parser.add_argument(
        "--api-base",
        default=None,
        help="Base URL for self-hosted OpenAI-compatible generation backends.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for self-hosted OpenAI-compatible backends. Defaults to env or EMPTY.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds for self-hosted backends.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        default=None,
        help="Device used by local transformers generation backends.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=220,
        help="Maximum number of generated tokens for local transformers backends.",
    )
    parser.add_argument("--max-items", type=int, default=0, help="Process at most N items (0 = all).")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel generation workers.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional fixed sleep between generations.")
    return parser


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    responses_path = Path(args.responses) if args.responses else DEFAULT_TASK_DIR / f"responses-{args.split}.jsonl"
    labels_path = Path(args.labels) if args.labels else DEFAULT_TASK_DIR / f"responses-{args.split}-labels.jsonl"
    output_path = (
        Path(args.out)
        if args.out
        else Path("data/generated") / args.provider / f"responses-{args.split}-with-neutral_{args.provider}.jsonl"
    )
    return responses_path, labels_path, output_path


def resolve_backend(args: argparse.Namespace) -> str:
    if args.backend != "auto":
        return args.backend
    return default_backend_for_provider(args.provider)


def resolve_model(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    if args.provider == "qwen":
        return DEFAULT_QWEN_MODEL
    return DEFAULT_GEMINI_MODEL


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")

    backend = resolve_backend(args)
    model_name = resolve_model(args)
    responses_path, labels_path, output_path = resolve_paths(args)
    label_map = load_label_map(labels_path)
    response_field = model_alias(model_name)
    done_ids = load_done_ids(output_path, response_field)
    print(f"[info] Already in output ({response_field}): {len(done_ids)} ids", file=sys.stderr)

    gemini_client = None
    openai_api_base = None
    openai_api_key = None
    local_tokenizer = None
    local_model = None
    local_device = None
    if backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")

        import google.genai as genai

        gemini_client = genai.Client(api_key=api_key)
    elif backend == "openai_compatible":
        openai_api_base = (
            args.api_base
            or os.environ.get("QWEN_API_BASE")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_QWEN_API_BASE
        )
        openai_api_key = (
            args.api_key
            or os.environ.get("QWEN_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY"
        )
    elif backend == "transformers":
        local_device = resolve_device(args.device)
        print(
            f"[info] Loading local transformers model {model_name} on {local_device}",
            file=sys.stderr,
        )
        local_tokenizer, local_model = load_local_generation_model(model_name, local_device)
    else:
        raise ValueError(f"Unsupported backend '{backend}'.")

    usage_totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    effective_workers = args.workers
    if backend == "transformers" and effective_workers != 1:
        print(
            f"[info] Backend '{backend}' uses a shared local model; forcing --workers 1",
            file=sys.stderr,
        )
        effective_workers = 1
    max_in_flight = max(1, effective_workers * 4)
    total_rows_seen = 0
    rows_written = 0

    def process_row(row: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, int]]:
        row_id = row["id"]
        query = row["query"]

        def call_model():
            if backend == "gemini":
                return generate_neutral_response_gemini(
                    client=gemini_client,
                    model=model_name,
                    query=query,
                )
            if backend == "transformers":
                return generate_neutral_response_transformers(
                    tokenizer=local_tokenizer,
                    model=local_model,
                    query=query,
                    device=local_device or "cpu",
                    max_new_tokens=args.max_new_tokens,
                )
            return generate_neutral_response_openai_compatible(
                api_base=openai_api_base or "",
                api_key=openai_api_key or "EMPTY",
                model=model_name,
                query=query,
                timeout=args.request_timeout,
            )

        neutral_response, usage = with_retries(call_model)
        merged_row = merge_response_row(row, label_map.get(row_id))
        merged_row[response_field] = clean_response_text(neutral_response)

        if args.sleep > 0:
            time.sleep(args.sleep)

        return row_id, merged_row, usage

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        in_flight = set()
        pending_ids: set[str] = set()

        for row in read_jsonl(responses_path):
            total_rows_seen += 1
            if args.max_items and total_rows_seen > args.max_items:
                break

            row_id = row.get("id")
            query = row.get("query")
            if not isinstance(row_id, str):
                print("[warn] Skip row without valid id", file=sys.stderr)
                continue
            if row_id in done_ids or row_id in pending_ids:
                continue
            if not isinstance(query, str) or not query.strip():
                print(f"[warn] Skip id={row_id}: missing query", file=sys.stderr)
                continue

            future = executor.submit(process_row, row)
            in_flight.add(future)
            pending_ids.add(row_id)

            if len(in_flight) >= max_in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for finished in done:
                    done_id, out_row, usage = finished.result()
                    pending_ids.discard(done_id)
                    append_jsonl(output_path, out_row)
                    done_ids.add(done_id)
                    rows_written += 1
                    for key in usage_totals:
                        usage_totals[key] += usage.get(key, 0)
                    if rows_written % 25 == 0:
                        print(f"[info] Written {rows_written} rows...", file=sys.stderr)

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for finished in done:
                done_id, out_row, usage = finished.result()
                pending_ids.discard(done_id)
                append_jsonl(output_path, out_row)
                done_ids.add(done_id)
                rows_written += 1
                for key in usage_totals:
                    usage_totals[key] += usage.get(key, 0)
                if rows_written % 25 == 0:
                    print(f"[info] Written {rows_written} rows...", file=sys.stderr)

    print(f"[done] Wrote {rows_written} new rows to {output_path}", file=sys.stderr)
    print(
        "[usage] "
        f"input_tokens={usage_totals['input_tokens']} "
        f"cached_input_tokens={usage_totals['cached_input_tokens']} "
        f"output_tokens={usage_totals['output_tokens']} "
        f"total_tokens={usage_totals['total_tokens']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
