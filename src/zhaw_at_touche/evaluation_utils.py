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


def binary_metrics_from_counts(
    counts: Counter[tuple[object, object]],
    *,
    negative_label: object = 0,
    positive_label: object = 1,
) -> dict[str, float | int]:
    true_negative = int(counts[(negative_label, negative_label)])
    false_positive = int(counts[(negative_label, positive_label)])
    false_negative = int(counts[(positive_label, negative_label)])
    true_positive = int(counts[(positive_label, positive_label)])
    samples = true_negative + false_positive + false_negative + true_positive
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "accuracy": safe_divide(true_positive + true_negative, samples),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "samples": samples,
    }


def render_binary_metrics(metrics: dict[str, float | int]) -> str:
    return "\n".join(
        [
            f"Accuracy: {float(metrics['accuracy']):.4f}",
            f"Precision: {float(metrics['precision']):.4f}",
            f"Recall: {float(metrics['recall']):.4f}",
            f"F1: {float(metrics['f1']):.4f}",
        ]
    )


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
    binary_metrics = binary_metrics_from_counts(counts)
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
        "accuracy": binary_metrics["accuracy"],
        "precision": binary_metrics["precision"],
        "recall": binary_metrics["recall"],
        "f1": binary_metrics["f1"],
        "confusion_matrix": {
            "true_negative": binary_metrics["true_negative"],
            "false_positive": binary_metrics["false_positive"],
            "false_negative": binary_metrics["false_negative"],
            "true_positive": binary_metrics["true_positive"],
        },
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


def compact_metrics_summary(summary: dict[str, Any]) -> dict[str, Any]:
    omitted_metric_fields = {
        "labels",
        "matrix",
        "per_label",
        "positive_label",
        "macro",
        "weighted",
    }
    compact: dict[str, Any] = {}
    for field in ("samples", "confusion_matrix", "accuracy", "precision", "recall", "f1"):
        if field in summary:
            compact[field] = summary[field]
    for field, value in summary.items():
        if field not in compact and field not in omitted_metric_fields:
            compact[field] = value
    return compact


def validation_metrics_payload(
    *,
    loss: float,
    summary: dict[str, Any],
) -> dict[str, float | int | None]:
    matrix = summary.get("matrix")
    samples = int(summary["samples"])

    negative_row = matrix.get("0", {}) if isinstance(matrix, dict) else {}
    positive_row = matrix.get("1", {}) if isinstance(matrix, dict) else {}
    true_negative = int(negative_row.get("0", 0))
    false_positive = int(negative_row.get("1", 0))
    false_negative = int(positive_row.get("0", 0))
    true_positive = int(positive_row.get("1", 0))

    return {
        "loss": loss,
        "accuracy": float(summary["accuracy"]),
        "precision": float(summary["precision"]),
        "recall": float(summary["recall"]),
        "f1": float(summary["f1"]),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "samples": samples,
    }
