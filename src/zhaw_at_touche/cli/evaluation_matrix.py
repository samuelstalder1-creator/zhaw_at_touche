from __future__ import annotations

import argparse
from pathlib import Path

from zhaw_at_touche.evaluation_utils import (
    collect_counts,
    compute_metrics,
    render_matrix,
    render_metrics,
    write_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a confusion-matrix summary from JSONL prediction files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="results",
        help="JSONL file or directory containing JSONL files (default: results).",
    )
    parser.add_argument(
        "--gold-key",
        default="gold_label",
        help="Field name for the gold label (default: gold_label).",
    )
    parser.add_argument(
        "--pred-key",
        default="response_label",
        help="Field name for the predicted label (default: response_label).",
    )
    parser.add_argument("--csv", type=Path, help="Optional path to write the confusion matrix as CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target_path = Path(args.path)
    counts, labels, total_rows, used_files, skipped_files = collect_counts(
        target_path,
        args.gold_key,
        args.pred_key,
    )
    per_label, macro, weighted = compute_metrics(counts, labels)

    print(f"rows: {total_rows}")
    print("matrix: rows=gold_label, cols=response_label")
    print(render_matrix(counts, labels))
    print("metrics:")
    print(render_metrics(per_label, macro, weighted))

    if used_files:
        print("used files:")
        for file_path, row_count in used_files:
            print(f"  {file_path}: {row_count}")

    if skipped_files:
        print("skipped files without matching columns:")
        for file_path in skipped_files:
            print(f"  {file_path}")

    if args.csv:
        write_csv(args.csv, counts, labels)
        print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
