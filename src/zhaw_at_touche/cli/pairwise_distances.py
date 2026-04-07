from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
from typing import Sequence

from zhaw_at_touche.constants import DEFAULT_RESULTS_DIR
from zhaw_at_touche.embedding_divergence import load_embedding_model
from zhaw_at_touche.jsonl import write_jsonl
from zhaw_at_touche.modeling import resolve_device
from zhaw_at_touche.pairwise_distance import (
    FieldPair,
    merge_jsonl_records_by_id,
    parse_field_pair,
    score_record_pair,
    summarize_pairwise_scores,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute embedding-space distances between arbitrary text fields."
    )
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help=(
            "Field comparison in left:right form. "
            "Sentence-level scoring is directional from left to right."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR / "pairwise_distances"),
    )
    parser.add_argument(
        "--embedding-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--score-granularity",
        choices=("response", "sentence"),
        default="response",
    )
    parser.add_argument(
        "--sentence-agg",
        choices=("max", "mean", "top2_mean", "top3_mean"),
        default="max",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _pair_columns(pairs: Sequence[FieldPair]) -> list[str]:
    return [pair.key for pair in pairs]


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    input_paths = [Path(raw_path) for raw_path in args.input_files]
    pairs = [parse_field_pair(raw_value) for raw_value in args.pair]

    print(f"merging {len(input_paths)} input files by id")
    records = merge_jsonl_records_by_id(input_paths)
    if not records:
        raise ValueError("No input rows were loaded from the provided files.")

    device = resolve_device(args.device)
    print(f"loading embedding model {args.embedding_model_name}")
    tokenizer, model = load_embedding_model(args.embedding_model_name, device)

    compared_fields = sorted({field for pair in pairs for field in (pair.left_field, pair.right_field)})
    jsonl_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for record in records:
        pairwise_scores: dict[str, float] = {}
        sentence_candidates: dict[str, list[dict[str, Any]]] = {}

        for pair in pairs:
            score, candidates = score_record_pair(
                tokenizer=tokenizer,
                model=model,
                record=record,
                left_field=pair.left_field,
                right_field=pair.right_field,
                score_granularity=args.score_granularity,
                sentence_agg=args.sentence_agg,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            pairwise_scores[pair.key] = score
            if candidates is not None:
                sentence_candidates[pair.key] = candidates

        output_row: dict[str, Any] = {
            "id": record.get("id", ""),
            "label": record.get("label"),
            "query": record.get("query", ""),
            "pairwise_scores": pairwise_scores,
        }
        for field_name in compared_fields:
            if field_name in record:
                output_row[field_name] = record[field_name]
        if sentence_candidates:
            output_row["sentence_candidates"] = sentence_candidates
        jsonl_rows.append(output_row)

        csv_row: dict[str, Any] = {
            "id": record.get("id", ""),
            "label": record.get("label"),
            "query": record.get("query", ""),
        }
        for pair in pairs:
            csv_row[pair.key] = pairwise_scores[pair.key]
        csv_rows.append(csv_row)

    base_summary = summarize_pairwise_scores(jsonl_rows, pairs)
    pair_summaries = {
        pair.key: {
            **base_summary["pairs"][pair.key],
            "pair": pair.label,
        }
        for pair in pairs
        if pair.key in base_summary["pairs"]
    }
    summary = {
        **base_summary,
        "input_files": [str(path) for path in input_paths],
        "embedding_model_name": args.embedding_model_name,
        "score_granularity": args.score_granularity,
        "sentence_agg": args.sentence_agg,
        "device": device,
        "pairs": pair_summaries,
    }

    write_jsonl(results_dir / "pairwise_distances.jsonl", jsonl_rows)
    write_csv_rows(
        results_dir / "pairwise_distances.csv",
        ["id", "label", "query", *_pair_columns(pairs)],
        csv_rows,
    )
    (results_dir / "pairwise_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"wrote {len(jsonl_rows)} merged rows to {results_dir}")
    for pair in pairs:
        pair_summary = summary["pairs"].get(pair.key)
        if not pair_summary:
            continue
        print(
            f"{pair.label}: mean={pair_summary['mean']:.4f} "
            f"median={pair_summary['median']:.4f} max={pair_summary['max']:.4f}"
        )
