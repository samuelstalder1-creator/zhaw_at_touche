from __future__ import annotations

import argparse
from pathlib import Path

from zhaw_at_touche.constants import DEFAULT_TASK_DIR
from zhaw_at_touche.overlap_utils import (
    DEFAULT_OVERLAP_FIELDS,
    collect_overlap_report,
    dataset_sizes,
    load_split,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check overlap across the train, validation, and test response datasets."
    )
    parser.add_argument(
        "--train",
        default=str(DEFAULT_TASK_DIR / "responses-train.jsonl"),
        help="Path to the train JSONL file.",
    )
    parser.add_argument(
        "--validation",
        default=str(DEFAULT_TASK_DIR / "responses-validation.jsonl"),
        help="Path to the validation JSONL file.",
    )
    parser.add_argument(
        "--test",
        default=str(DEFAULT_TASK_DIR / "responses-test.jsonl"),
        help="Path to the test JSONL file.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=list(DEFAULT_OVERLAP_FIELDS),
        help="Fields to compare for overlap.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=3,
        help="How many overlapping examples to print for each comparison.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    split_rows = {
        "train": load_split(Path(args.train)),
        "validation": load_split(Path(args.validation)),
        "test": load_split(Path(args.test)),
    }

    print("Dataset sizes")
    for split_name, row_count in dataset_sizes(split_rows).items():
        print(f"- {split_name}: {row_count:,}")
    print()

    for field_name in args.fields:
        print(f"Overlap by {field_name}")
        for comparison in collect_overlap_report(field_name, split_rows, args.sample_limit):
            print(f"- {comparison.label}: {comparison.overlap_count:,}")
            for sample in comparison.samples:
                print(f"  sample: {sample.key_text}")
                ids_text = " | ".join(
                    f"{split_name}=[{', '.join(sample.split_ids[split_name])}]"
                    for split_name in comparison.split_names
                )
                print(f"  ids: {ids_text}")
        print()


if __name__ == "__main__":
    main()
