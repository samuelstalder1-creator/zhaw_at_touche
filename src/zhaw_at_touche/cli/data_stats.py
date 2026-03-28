from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from zhaw_at_touche.constants import DEFAULT_RESPONSE_SPLITS, DEFAULT_TASK_CATEGORIES, DEFAULT_TASK_DIR
from zhaw_at_touche.datasets import word_count
from zhaw_at_touche.jsonl import read_jsonl


def average(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_category_split(task_dir: Path, category: str, split: str) -> dict[str, Any]:
    data_path = task_dir / f"{category}-{split}.jsonl"
    label_path = task_dir / f"{category}-{split}-labels.jsonl"

    data_rows = list(read_jsonl(data_path))
    label_rows = list(read_jsonl(label_path)) if label_path.exists() else []
    label_counter = Counter(
        int(row["label"])
        for row in label_rows
        if isinstance(row.get("label"), int)
    )

    summary: dict[str, Any] = {
        "category": category,
        "split": split,
        "data_rows": len(data_rows),
        "label_rows": len(label_rows),
        "label_distribution": dict(sorted(label_counter.items())),
    }

    if category == "responses":
        query_lengths = [word_count(str(row.get("query", ""))) for row in data_rows]
        response_lengths = [word_count(str(row.get("response", ""))) for row in data_rows]
        meta_topics = Counter(
            str(row["meta_topic"])
            for row in data_rows
            if isinstance(row.get("meta_topic"), str)
        )
        search_engines = Counter(
            str(row["search_engine"])
            for row in data_rows
            if isinstance(row.get("search_engine"), str)
        )
        summary.update(
            {
                "query_words_avg": average(query_lengths),
                "query_words_median": float(statistics.median(query_lengths)) if query_lengths else 0.0,
                "response_words_avg": average(response_lengths),
                "response_words_median": float(statistics.median(response_lengths)) if response_lengths else 0.0,
                "top_meta_topics": meta_topics.most_common(5),
                "top_search_engines": search_engines.most_common(5),
            }
        )
    elif category == "sentence-pairs":
        pair_lengths = [
            word_count(str(row.get("sentence1", ""))) + word_count(str(row.get("sentence2", "")))
            for row in data_rows
        ]
        summary.update(
            {
                "pair_words_avg": average(pair_lengths),
                "pair_words_median": float(statistics.median(pair_lengths)) if pair_lengths else 0.0,
            }
        )
    elif category == "tokens":
        token_lengths = [
            len(row["tokens"])
            for row in data_rows
            if isinstance(row.get("tokens"), list)
        ]
        summary.update(
            {
                "tokens_avg": average(token_lengths),
                "tokens_median": float(statistics.median(token_lengths)) if token_lengths else 0.0,
            }
        )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print summary statistics for the task data.")
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_TASK_CATEGORIES))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_RESPONSE_SPLITS))
    parser.add_argument("--json-out", default=None, help="Optional path for the summary JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task_dir = Path(args.task_dir)

    summaries = [
        summarize_category_split(task_dir, category, split)
        for category in args.categories
        for split in args.splits
    ]

    for summary in summaries:
        head = (
            f"{summary['category']}/{summary['split']}: "
            f"data_rows={summary['data_rows']} label_rows={summary['label_rows']} "
            f"labels={summary['label_distribution']}"
        )
        print(head)
        if summary["category"] == "responses":
            print(
                "  "
                f"query_words_avg={summary['query_words_avg']:.2f} "
                f"query_words_median={summary['query_words_median']:.2f} "
                f"response_words_avg={summary['response_words_avg']:.2f} "
                f"response_words_median={summary['response_words_median']:.2f}"
            )
            print(f"  top_meta_topics={summary['top_meta_topics']}")
            print(f"  top_search_engines={summary['top_search_engines']}")
        elif summary["category"] == "sentence-pairs":
            print(
                "  "
                f"pair_words_avg={summary['pair_words_avg']:.2f} "
                f"pair_words_median={summary['pair_words_median']:.2f}"
            )
        elif summary["category"] == "tokens":
            print(
                "  "
                f"tokens_avg={summary['tokens_avg']:.2f} "
                f"tokens_median={summary['tokens_median']:.2f}"
            )

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"json summary written to {output_path}")


if __name__ == "__main__":
    main()
