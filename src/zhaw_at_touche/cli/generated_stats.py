from __future__ import annotations

import argparse
import json
from pathlib import Path

from zhaw_at_touche.generated_stats import (
    basic_length_summaries,
    histogram_output_path,
    load_generated_rows,
    summaries_to_dict,
    token_length_analysis,
    write_histogram_svg,
)


def resolve_default_paths() -> list[str]:
    candidates = [
        Path("data/generated/gemini/responses-test-with-neutral_gemini.jsonl"),
        Path("data/generated/gemini/responses-validation-with-neutral_gemini.jsonl"),
    ]
    existing = [str(path) for path in candidates if path.exists()]
    if existing:
        return existing
    return [str(candidates[0])]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print statistics for generated neutral response files."
    )
    parser.add_argument("paths", nargs="*", default=resolve_default_paths())
    parser.add_argument(
        "--generated-field",
        "--neutral-field",
        dest="generated_field",
        default=None,
        help="Field containing the generated response. Defaults to auto-detect.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Analyze at most N rows per file.")
    parser.add_argument(
        "--tokenizer-model",
        "--model",
        dest="tokenizer_model",
        default=None,
        help="Optional Gemini tokenizer model to compute token-count summaries.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--histogram-dir",
        default="results/generated_stats/histograms",
        help="Directory where token histogram SVG files will be written.",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=30,
        help="Number of bins per token histogram.",
    )
    parser.add_argument(
        "--no-histogram",
        action="store_true",
        help="Skip token histogram generation when --tokenizer-model is set.",
    )
    parser.add_argument("--json-out", default=None, help="Optional path for the summary JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be >= 0.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.histogram_bins < 1:
        raise ValueError("--histogram-bins must be >= 1.")

    payload: dict[str, object] = {}

    for raw_path in args.paths:
        path = Path(raw_path)
        generated_field, rows = load_generated_rows(path, args.generated_field, args.limit)
        if not rows:
            raise ValueError(f"No valid rows found in {path}")

        basic_summaries = basic_length_summaries(rows)
        token_analysis = (
            token_length_analysis(rows, args.tokenizer_model, args.batch_size)
            if args.tokenizer_model
            else None
        )
        token_summaries = token_analysis.summaries if token_analysis else []
        histogram_path: Path | None = None
        if token_analysis and not args.no_histogram:
            histogram_path = histogram_output_path(path, Path(args.histogram_dir))
            write_histogram_svg(
                histogram_path,
                path,
                token_analysis.metric_values,
                token_analysis.summaries,
                len(rows),
                args.histogram_bins,
            )

        print(f"File: {path}")
        print(f"Rows analyzed: {len(rows)}")
        print(f"Generated field: {generated_field}")
        for summary in basic_summaries:
            print(
                f"  {summary.name}: total={summary.total} avg={summary.average:.2f} "
                f"median={summary.median:.2f} min={summary.min_value} ({summary.min_id}) "
                f"max={summary.max_value} ({summary.max_id})"
            )
        if token_summaries:
            print("  token metrics:")
            for summary in token_summaries:
                print(
                    f"    {summary.name}: total={summary.total} avg={summary.average:.2f} "
                    f"median={summary.median:.2f} min={summary.min_value} ({summary.min_id}) "
                    f"max={summary.max_value} ({summary.max_id})"
                )
            if histogram_path:
                print(f"  token histogram: {histogram_path}")

        payload[str(path)] = {
            "generated_field": generated_field,
            "rows_analyzed": len(rows),
            "basic_summaries": summaries_to_dict(basic_summaries),
            "token_summaries": summaries_to_dict(token_summaries),
            "token_histogram_svg": str(histogram_path) if histogram_path else None,
        }

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json summary written to {output_path}")


if __name__ == "__main__":
    main()
