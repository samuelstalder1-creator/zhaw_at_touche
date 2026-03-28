from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


def iter_jsonl_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(candidate for candidate in path.rglob("*.jsonl") if candidate.is_file())
    raise FileNotFoundError(f"Path does not exist: {path}")


def counts_from_pairs(
    gold_labels: Sequence[object],
    predicted_labels: Sequence[object],
) -> tuple[Counter[tuple[object, object]], list[object], int]:
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("Gold and predicted label sequences must be the same length.")

    counts: Counter[tuple[object, object]] = Counter()
    labels: set[object] = set()
    for gold_label, predicted_label in zip(gold_labels, predicted_labels):
        counts[(gold_label, predicted_label)] += 1
        labels.add(gold_label)
        labels.add(predicted_label)
    return counts, sorted(labels), len(gold_labels)


def collect_counts(
    path: Path,
    gold_key: str,
    pred_key: str,
) -> tuple[Counter[tuple[object, object]], list[object], int, list[tuple[Path, int]], list[Path]]:
    counts: Counter[tuple[object, object]] = Counter()
    labels: set[object] = set()
    used_files: list[tuple[Path, int]] = []
    skipped_files: list[Path] = []
    total_rows = 0

    for jsonl_path in iter_jsonl_files(path):
        matched_rows = 0
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {jsonl_path} at line {line_number}"
                    ) from exc
                if gold_key not in row or pred_key not in row:
                    continue
                gold_label = row[gold_key]
                pred_label = row[pred_key]
                counts[(gold_label, pred_label)] += 1
                labels.add(gold_label)
                labels.add(pred_label)
                matched_rows += 1
                total_rows += 1
        if matched_rows:
            used_files.append((jsonl_path, matched_rows))
        else:
            skipped_files.append(jsonl_path)

    if not total_rows:
        raise ValueError(
            f"No rows with both '{gold_key}' and '{pred_key}' were found in {path}"
        )

    return counts, sorted(labels), total_rows, used_files, skipped_files


def render_matrix(counts: Counter[tuple[object, object]], labels: Sequence[object]) -> str:
    label_text = [str(label) for label in labels]
    cell_width = max(
        len("gold\\pred"),
        *(len(text) for text in label_text),
        *(len(str(counts[(gold, pred)])) for gold in labels for pred in labels),
    )

    lines = []
    header = [f"{'gold\\\\pred':>{cell_width}}", *[f"{text:>{cell_width}}" for text in label_text]]
    lines.append(" ".join(header))
    for gold in labels:
        row = [f"{str(gold):>{cell_width}}"]
        row.extend(f"{counts[(gold, pred)]:>{cell_width}}" for pred in labels)
        lines.append(" ".join(row))
    return "\n".join(lines)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(
    counts: Counter[tuple[object, object]],
    labels: Sequence[object],
) -> tuple[list[dict[str, float | int | object]], dict[str, float], dict[str, float]]:
    per_label: list[dict[str, float | int | object]] = []
    total = sum(counts.values())

    for label in labels:
        tp = counts[(label, label)]
        fp = sum(counts[(other, label)] for other in labels if other != label)
        fn = sum(counts[(label, other)] for other in labels if other != label)
        support = sum(counts[(label, other)] for other in labels)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_label.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    macro = {
        "precision": safe_divide(
            sum(float(metric["precision"]) for metric in per_label),
            len(per_label),
        ),
        "recall": safe_divide(
            sum(float(metric["recall"]) for metric in per_label),
            len(per_label),
        ),
        "f1": safe_divide(
            sum(float(metric["f1"]) for metric in per_label),
            len(per_label),
        ),
        "support": total,
    }
    weighted = {
        "precision": safe_divide(
            sum(float(metric["precision"]) * int(metric["support"]) for metric in per_label),
            total,
        ),
        "recall": safe_divide(
            sum(float(metric["recall"]) * int(metric["support"]) for metric in per_label),
            total,
        ),
        "f1": safe_divide(
            sum(float(metric["f1"]) * int(metric["support"]) for metric in per_label),
            total,
        ),
        "support": total,
    }
    return per_label, macro, weighted


def render_metrics(
    per_label: Sequence[dict[str, float | int | object]],
    macro: dict[str, float],
    weighted: dict[str, float],
) -> str:
    label_width = max(
        len("label"),
        len("macro avg"),
        len("weighted avg"),
        *(len(str(metric["label"])) for metric in per_label),
    )
    score_width = len("precision")
    support_width = max(
        len("support"),
        *(len(str(int(metric["support"]))) for metric in per_label),
        len(str(int(macro["support"]))),
        len(str(int(weighted["support"]))),
    )

    lines = [
        f"{'label':>{label_width}} {'precision':>{score_width}} {'recall':>{score_width}} {'f1':>{score_width}} {'support':>{support_width}}"
    ]
    for metric in per_label:
        lines.append(
            f"{str(metric['label']):>{label_width}} "
            f"{float(metric['precision']):>{score_width}.4f} "
            f"{float(metric['recall']):>{score_width}.4f} "
            f"{float(metric['f1']):>{score_width}.4f} "
            f"{int(metric['support']):>{support_width}}"
        )
    lines.append(
        f"{'macro avg':>{label_width}} "
        f"{macro['precision']:>{score_width}.4f} "
        f"{macro['recall']:>{score_width}.4f} "
        f"{macro['f1']:>{score_width}.4f} "
        f"{int(macro['support']):>{support_width}}"
    )
    lines.append(
        f"{'weighted avg':>{label_width}} "
        f"{weighted['precision']:>{score_width}.4f} "
        f"{weighted['recall']:>{score_width}.4f} "
        f"{weighted['f1']:>{score_width}.4f} "
        f"{int(weighted['support']):>{support_width}}"
    )
    return "\n".join(lines)


def accuracy(gold_labels: Sequence[int], predicted_labels: Sequence[int]) -> float:
    if not gold_labels:
        return 0.0
    correct = sum(int(gold == pred) for gold, pred in zip(gold_labels, predicted_labels))
    return correct / len(gold_labels)


def write_csv(
    output_path: Path,
    counts: Counter[tuple[object, object]],
    labels: Sequence[object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold_label\\response_label", *labels])
        for gold in labels:
            writer.writerow([gold, *[counts[(gold, pred)] for pred in labels]])


def metrics_dict(
    gold_labels: Sequence[int],
    predicted_labels: Sequence[int],
) -> dict[str, Any]:
    counts, labels, total_rows = counts_from_pairs(gold_labels, predicted_labels)
    per_label, macro, weighted = compute_metrics(counts, labels)
    positive_label_metrics = next(
        (
            {
                "precision": float(metric["precision"]),
                "recall": float(metric["recall"]),
                "f1": float(metric["f1"]),
                "support": int(metric["support"]),
            }
            for metric in per_label
            if metric["label"] == 1
        ),
        None,
    )
    return {
        "samples": total_rows,
        "accuracy": accuracy(gold_labels, predicted_labels),
        "labels": list(labels),
        "matrix": {
            str(gold): {str(pred): counts[(gold, pred)] for pred in labels}
            for gold in labels
        },
        "per_label": list(per_label),
        "positive_label": positive_label_metrics,
        "macro": macro,
        "weighted": weighted,
    }
