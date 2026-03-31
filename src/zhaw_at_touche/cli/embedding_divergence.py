from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any
from typing import Sequence

from zhaw_at_touche.constants import DEFAULT_MODELS_DIR, DEFAULT_RESULTS_DIR, DEFAULT_SETUP_NAME
from zhaw_at_touche.embedding_divergence import (
    calibrate_threshold,
    load_embedding_model,
    load_embedding_state,
    score_record,
    score_records,
)
from zhaw_at_touche.embedding_setups import DEFAULT_EMBEDDING_SETUPS_DIR, load_setup_defaults
from zhaw_at_touche.evaluation_utils import (
    compute_metrics,
    counts_from_pairs,
    metrics_dict,
    render_matrix,
    render_metrics,
    write_csv,
)
from zhaw_at_touche.jsonl import read_jsonl, write_jsonl
from zhaw_at_touche.modeling import resolve_device


def resolve_default_eval_paths(eval_splits: Sequence[str] | None = None) -> list[Path]:
    splits = list(eval_splits) if eval_splits else ["test"]
    generated_paths = [
        Path(f"data/generated/gemini/responses-{split}-with-neutral_gemini.jsonl")
        for split in splits
    ]
    if all(path.exists() for path in generated_paths):
        return generated_paths

    return [
        Path(f"data/task/preprocessed/responses-{split}-merged.jsonl")
        for split in splits
    ]


def resolve_default_calibration_paths() -> list[Path]:
    generated_path = Path("data/generated/gemini/responses-validation-with-neutral_gemini.jsonl")
    if generated_path.exists():
        return [generated_path]
    return [Path("data/task/preprocessed/responses-validation-merged.jsonl")]


def save_confusion_matrix_image(
    gold_labels: list[int],
    predicted_labels: list[int],
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    labels = sorted(set(gold_labels) | set(predicted_labels))
    matrix = confusion_matrix(gold_labels, predicted_labels, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[str(label) for label in labels],
    )
    display.plot(cmap=plt.cm.Blues, values_format="d", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def base_defaults() -> dict[str, object]:
    return {
        "model_dir": None,
        "results_dir": None,
        "eval_splits": ["test"],
        "input_files": None,
        "calibration_input_files": None,
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "neutral_field": "gemini25flashlite",
        "distance_metric": "cosine",
        "score_granularity": "sentence",
        "sentence_agg": "max",
        "threshold": None,
        "threshold_metric": "positive_f1",
        "batch_size": 32,
        "max_length": 256,
        "device": None,
    }


def build_parser(setup_defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    defaults = base_defaults()
    if setup_defaults:
        defaults.update(setup_defaults)
    input_file_defaults = defaults["input_files"]
    if input_file_defaults is None:
        input_file_defaults = [str(path) for path in resolve_default_eval_paths(defaults["eval_splits"])]
    calibration_defaults = defaults["calibration_input_files"]
    if calibration_defaults is None:
        calibration_defaults = [str(path) for path in resolve_default_calibration_paths()]

    parser = argparse.ArgumentParser(
        description="Evaluate embedding-space divergence against a neutral reference response."
    )
    parser.add_argument("--setup-name", default=DEFAULT_SETUP_NAME)
    parser.add_argument(
        "--setups-dir",
        default=str(DEFAULT_EMBEDDING_SETUPS_DIR),
        help="Directory containing optional <setup-name>.json embedding-divergence defaults.",
    )
    parser.add_argument(
        "--model-dir",
        default=defaults["model_dir"],
        help="Directory containing a saved embedding_state.json threshold/state bundle.",
    )
    parser.add_argument("--results-dir", default=defaults["results_dir"])
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        choices=("validation", "test"),
        default=defaults["eval_splits"],
        help="Default evaluation splits when --input-files is not passed.",
    )
    parser.add_argument("--input-files", nargs="+", default=input_file_defaults)
    parser.add_argument("--calibration-input-files", nargs="+", default=calibration_defaults)
    parser.add_argument("--embedding-model-name", default=defaults["embedding_model_name"])
    parser.add_argument("--neutral-field", default=defaults["neutral_field"])
    parser.add_argument("--distance-metric", choices=("cosine",), default=defaults["distance_metric"])
    parser.add_argument(
        "--score-granularity",
        choices=("response", "sentence"),
        default=defaults["score_granularity"],
    )
    parser.add_argument("--sentence-agg", choices=("max", "mean"), default=defaults["sentence_agg"])
    parser.add_argument("--threshold", type=float, default=defaults["threshold"])
    parser.add_argument(
        "--threshold-metric",
        choices=("positive_f1", "accuracy"),
        default=defaults["threshold_metric"],
        help="Metric used to fit the threshold when --threshold is not provided.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--max-length", type=int, default=defaults["max_length"])
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=defaults["device"])
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--setup-name", default=DEFAULT_SETUP_NAME)
    pre_parser.add_argument("--setups-dir", default=str(DEFAULT_EMBEDDING_SETUPS_DIR))
    pre_args, _ = pre_parser.parse_known_args(argv)

    setup_defaults = load_setup_defaults(
        setup_name=pre_args.setup_name,
        setups_dir=Path(pre_args.setups_dir),
    )
    parser = build_parser(setup_defaults)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR / args.setup_name
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_MODELS_DIR / args.setup_name
    device = resolve_device(args.device)
    saved_state = load_embedding_state(model_dir)

    threshold = args.threshold
    calibration_summary = None
    threshold_source = "manual"
    if threshold is None and saved_state is not None:
        raw_threshold = saved_state.get("threshold")
        if raw_threshold is not None:
            threshold = float(raw_threshold)
            calibration_summary = saved_state.get("threshold_summary")
            threshold_source = "saved_state"

    print(f"loading embedding model {args.embedding_model_name}")
    tokenizer, model = load_embedding_model(args.embedding_model_name, device)
    if threshold is None:
        calibration_records: list[dict[str, Any]] = []
        for raw_path in args.calibration_input_files:
            path = Path(raw_path)
            calibration_records.extend(list(read_jsonl(path)))
        if not calibration_records:
            raise ValueError("Calibration input files must contain at least one labeled record.")

        calibration_scores: list[float] = []
        calibration_labels: list[int] = []
        for record in calibration_records:
            score, _ = score_record(
                tokenizer=tokenizer,
                model=model,
                record=record,
                neutral_field=args.neutral_field,
                score_granularity=args.score_granularity,
                sentence_agg=args.sentence_agg,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            calibration_scores.append(score)
            calibration_labels.append(int(record["label"]))

        threshold, calibration_summary = calibrate_threshold(
            calibration_scores,
            calibration_labels,
            threshold_metric=args.threshold_metric,
        )
        threshold_source = "validation_calibration"

    print(f"using threshold={threshold:.6f} ({threshold_source})")

    all_output_rows: list[dict[str, Any]] = []
    all_gold_labels: list[int] = []
    all_predicted_labels: list[int] = []
    file_summaries: dict[str, Any] = {}

    for raw_path in args.input_files:
        path = Path(raw_path)
        records = list(read_jsonl(path))
        if not records:
            raise ValueError(f"Input file is empty: {path}")

        predictions = score_records(
            tokenizer=tokenizer,
            model=model,
            records=records,
            neutral_field=args.neutral_field,
            score_granularity=args.score_granularity,
            sentence_agg=args.sentence_agg,
            threshold=threshold,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        file_gold_labels: list[int] = []
        file_predicted_labels: list[int] = []
        file_output_rows: list[dict[str, Any]] = []

        for index, record in enumerate(records):
            gold_label = int(record["label"])
            prediction = predictions[index]
            output_row: dict[str, Any] = {
                "source_file": path.name,
                "id": record.get("id", ""),
                "query": record.get("query", ""),
                "gold_label": gold_label,
                "response_label": prediction.label,
                "response_score": prediction.score,
            }
            if "response" in record:
                output_row["response"] = record["response"]
            if args.neutral_field in record:
                output_row[args.neutral_field] = record[args.neutral_field]
            if prediction.sentence_candidates is not None:
                output_row["sentence_candidates"] = prediction.sentence_candidates

            file_gold_labels.append(gold_label)
            file_predicted_labels.append(prediction.label)
            file_output_rows.append(output_row)

        predictions_path = results_dir / f"{path.stem}-predictions.jsonl"
        write_jsonl(predictions_path, file_output_rows)
        all_output_rows.extend(file_output_rows)
        all_gold_labels.extend(file_gold_labels)
        all_predicted_labels.extend(file_predicted_labels)

        summary = metrics_dict(file_gold_labels, file_predicted_labels)
        summary["neutral_field"] = args.neutral_field
        summary["score_granularity"] = args.score_granularity
        summary["sentence_agg"] = args.sentence_agg
        file_summaries[path.name] = summary
        positive_label = summary.get("positive_label")
        file_f1 = positive_label["f1"] if isinstance(positive_label, dict) else 0.0
        print(
            f"{path.name}: accuracy={summary['accuracy']:.4f} "
            f"f1={file_f1:.4f} weighted_f1={summary['weighted']['f1']:.4f}"
        )

    overall_summary = metrics_dict(all_gold_labels, all_predicted_labels)
    summary_payload = {
        "scoring_backend": "embedding_divergence",
        "embedding_model_name": args.embedding_model_name,
        "model_dir": str(model_dir),
        "results_dir": str(results_dir),
        "device": device,
        "distance_metric": args.distance_metric,
        "neutral_field": args.neutral_field,
        "score_granularity": args.score_granularity,
        "sentence_agg": args.sentence_agg,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_metric": args.threshold_metric,
        "calibration_input_files": list(args.calibration_input_files),
        "calibration_summary": calibration_summary,
        "files": file_summaries,
        "overall": overall_summary,
    }
    (results_dir / "metrics_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    standardized_rows = [
        {
            "source_file": row["source_file"],
            "id": row["id"],
            "label": row["gold_label"],
            "pred": row["response_label"],
        }
        for row in all_output_rows
    ]
    write_csv_rows(
        results_dir / "standardized_results.csv",
        ["source_file", "id", "label", "pred"],
        standardized_rows,
    )

    misclassified_rows = [
        row for row in all_output_rows if int(row["gold_label"]) != int(row["response_label"])
    ]
    misclassified_fields = ["source_file", "id", "query", "response", "gold_label", "response_label", "response_score"]
    extra_fields = sorted(
        {
            field
            for row in misclassified_rows
            for field in row.keys()
            if field not in misclassified_fields
        }
    )
    write_csv_rows(
        results_dir / "misclassified_analysis.csv",
        misclassified_fields + extra_fields,
        misclassified_rows,
    )

    counts, labels, _ = counts_from_pairs(all_gold_labels, all_predicted_labels)
    per_label, macro, weighted = compute_metrics(counts, labels)
    matrix_text = render_matrix(counts, labels)
    metrics_text = render_metrics(per_label, macro, weighted)
    (results_dir / "response_metrics.txt").write_text(
        f"rows: {len(all_gold_labels)}\n"
        "matrix: rows=gold_label, cols=response_label\n"
        f"{matrix_text}\n"
        "metrics:\n"
        f"{metrics_text}\n",
        encoding="utf-8",
    )
    write_csv(results_dir / "confusion_matrix.csv", counts, labels)
    save_confusion_matrix_image(
        gold_labels=all_gold_labels,
        predicted_labels=all_predicted_labels,
        output_path=results_dir / "confusion_matrix.png",
    )

    print(f"overall accuracy={overall_summary['accuracy']:.4f}")
    overall_positive_label = overall_summary.get("positive_label")
    if isinstance(overall_positive_label, dict):
        print(f"overall f1={overall_positive_label['f1']:.4f}")
    print(f"overall weighted_f1={overall_summary['weighted']['f1']:.4f}")
    print(f"artifacts written to {results_dir}")


if __name__ == "__main__":
    main()
