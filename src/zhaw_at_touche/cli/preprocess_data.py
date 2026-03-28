from __future__ import annotations

import argparse
from pathlib import Path

from zhaw_at_touche.constants import DEFAULT_PREPROCESSED_DIR, DEFAULT_RESPONSE_SPLITS, DEFAULT_TASK_DIR
from zhaw_at_touche.datasets import merge_response_split
from zhaw_at_touche.jsonl import write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge response rows with their label rows into easier-to-consume JSONL files."
    )
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_PREPROCESSED_DIR))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_RESPONSE_SPLITS))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task_dir = Path(args.task_dir)
    out_dir = Path(args.out_dir)

    for split in args.splits:
        response_path = task_dir / f"responses-{split}.jsonl"
        label_path = task_dir / f"responses-{split}-labels.jsonl"
        output_path = out_dir / f"responses-{split}-merged.jsonl"

        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Pass --overwrite to replace it."
            )

        merged_rows = merge_response_split(response_path, label_path)
        write_jsonl(output_path, merged_rows)
        print(f"{split}: wrote {len(merged_rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
