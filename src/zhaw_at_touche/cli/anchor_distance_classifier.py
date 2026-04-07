from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any
from typing import Sequence

from zhaw_at_touche.anchor_distance_classifier import (
    load_anchor_distance_state,
    load_classifier_bundle,
    load_merged_records,
    score_records,
)
from zhaw_at_touche.constants import DEFAULT_MODELS_DIR, DEFAULT_RESULTS_DIR, DEFAULT_SETUP_NAME
from zhaw_at_touche.embedding_divergence import calibrate_threshold, load_embedding_model
from zhaw_at_touche.evaluation_utils import (
    compute_metrics,
    counts_from_pairs,
    metrics_dict,
    render_matrix,
    render_metrics,
    write_csv,
)
from zhaw_at_touche.jsonl import write_jsonl
from zhaw_at_touche.modeling import resolve_device
from zhaw_at_touche.validation_setups import DEFAULT_VALIDATION_SETUPS_DIR, load_setup_defaults


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


def cli_option_was_provided(raw_argv: Sequence[str], option_name: str) -> bool:
    return any(
        token == option_name or token.startswith(f"{option_name}=")
        for token in raw_argv
    )


def validate_paired_paths(primary_paths: Sequence[str], secondary_paths: Sequence[str], *, label: str) -> None:
    if len(primary_paths) != len(secondary_paths):
        raise ValueError(
            f"{label} must contain the same number of primary and auxiliary files, "
            f"received {len(primary_paths)} and {len(secondary_paths)}."
        )


def base_defaults() -> dict[str, object]:
    return {
        "model_dir": None,
        "results_dir": None,
        "input_files": ["data/generated/gemini/responses-test-with-neutral_gemini.jsonl"],
        "aux_input_files": ["data/generated/qwen/responses-test-with-neutral_qwen.jsonl"],
        "calibration_input_files": ["data/generated/gemini/responses-validation-with-neutral_gemini.jsonl"],
        "aux_calibration_input_files": ["data/generated/qwen/responses-validation-with-neutral_qwen.jsonl"],
        "embedding_model_name": "sentence-transformers/all-mpnet-base-v2",
        "query_field": "query",
        "response_field": "response",
        "neutral_field": "gemini25flashlite",
        "aux_neutral_field": "qwen",
        "score_granularity": "response",
        "threshold": None,
        "threshold_metric": "macro_f1",
        "batch_size": 32,
        "max_length": 512,
        "device": None,
    }


def apply_saved_state_defaults(
    args: argparse.Namespace,
    raw_argv: Sequence[str],
    saved_state: dict[str, Any] | None,
) -> argparse.Namespace:
    if saved_state is None:
        return args

    state_field_map = {
        "--embedding-model-name": "embedding_model_name",
        "--query-field": "query_field",
        "--response-field": "response_field",
        "--neutral-field": "neutral_field",
        "--aux-neutral-field": "aux_neutral_field",
        "--score-granularity": "score_granularity",
        "--threshold-metric": "threshold_metric",
        "--batch-size": "batch_size",
        "--max-length": "max_length",
    }
    for option_name, state_key in state_field_map.items():
        if cli_option_was_provided(raw_argv, option_name):
            continue
        state_value = saved_state.get(state_key)
        if state_value is not None:
            setattr(args, state_key, state_value)
    return args


def build_parser(setup_defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    defaults = base_defaults()
    if setup_defaults:
        defaults.update(setup_defaults)

    parser = argparse.ArgumentParser(
        description="Evaluate anchor-distance features between query, Gemini/Qwen neutrals, and the response."
    )
    parser.add_argument("--setup-name", default=DEFAULT_SETUP_NAME)
    parser.add_argument(
        "--setups-dir",
        default=str(DEFAULT_VALIDATION_SETUPS_DIR),
        help="Directory containing optional <setup-name>.json evaluation defaults.",
    )
    parser.add_argument(
        "--model-dir",
        default=defaults["model_dir"],
        help="Directory containing the saved classifier bundle and embedding_state.json.",
    )
    parser.add_argument("--results-dir", default=defaults["results_dir"])
    parser.add_argument("--input-files", nargs="+", default=defaults["input_files"])
    parser.add_argument("--aux-input-files", nargs="+", default=defaults["aux_input_files"])
    parser.add_argument(
        "--calibration-input-files",
        nargs="+",
        default=defaults["calibration_input_files"],
    )
    parser.add_argument(
        "--aux-calibration-input-files",
        nargs="+",
        default=defaults["aux_calibration_input_files"],
    )
    parser.add_argument("--embedding-model-name", default=defaults["embedding_model_name"])
    parser.add_argument("--query-field", default=defaults["query_field"])
    parser.add_argument("--response-field", default=defaults["response_field"])
    parser.add_argument("--neutral-field", default=defaults["neutral_field"])
    parser.add_argument("--aux-neutral-field", default=defaults["aux_neutral_field"])
    parser.add_argument(
        "--score-granularity",
        choices=("response",),
        default=defaults["score_granularity"],
    )
    parser.add_argument("--threshold", type=float, default=defaults["threshold"])
    parser.add_argument(
        "--threshold-metric",
        choices=("positive_f1", "macro_f1", "accuracy"),
        default=defaults["threshold_metric"],
    )
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--max-length", type=int, default=defaults["max_length"])
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=defaults["device"])
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--setup-name", default=DEFAULT_SETUP_NAME)
    pre_parser.add_argument("--setups-dir", default=str(DEFAULT_VALIDATION_SETUPS_DIR))
    pre_args, _ = pre_parser.parse_known_args(raw_argv)

    setup_defaults = load_setup_defaults(
        setup_name=pre_args.setup_name,
        setups_dir=Path(pre_args.setups_dir),
    )
    parser = build_parser(setup_defaults)
    args = parser.parse_args(raw_argv)
    validate_paired_paths(args.input_files, args.aux_input_files, label="--input-files")
    validate_paired_paths(
        args.calibration_input_files,
        args.aux_calibration_input_files,
        label="--calibration-input-files",
    )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(raw_argv)
    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR / args.setup_name
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_MODELS_DIR / args.setup_name
    device = resolve_device(args.device)
    saved_state = load_anchor_distance_state(model_dir)
    args = apply_saved_state_defaults(args, raw_argv, saved_state)

    classifier = load_classifier_bundle(model_dir)
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
        for primary_path, secondary_path in zip(
            args.calibration_input_files,
            args.aux_calibration_input_files,
            strict=True,
        ):
            calibration_records.extend(
                load_merged_records(Path(primary_path), Path(secondary_path))
            )
        if not calibration_records:
            raise ValueError("Calibration input files must contain at least one labeled record.")

        calibration_predictions = score_records(
            classifier=classifier,
            tokenizer=tokenizer,
            model=model,
            records=calibration_records,
            query_field=args.query_field,
            response_field=args.response_field,
            neutral_field=args.neutral_field,
            aux_neutral_field=args.aux_neutral_field,
            threshold=0.5,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        calibration_scores = [prediction.score for prediction in calibration_predictions]
        calibration_labels = [int(record["label"]) for record in calibration_records]
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

    for primary_path, secondary_path in zip(args.input_files, args.aux_input_files, strict=True):
        primary = Path(primary_path)
        secondary = Path(secondary_path)
        records = load_merged_records(primary, secondary)
        if not records:
            raise ValueError(f"Input file pair is empty: {primary} + {secondary}")

        predictions = score_records(
            classifier=classifier,
            tokenizer=tokenizer,
            model=model,
            records=records,
            query_field=args.query_field,
            response_field=args.response_field,
            neutral_field=args.neutral_field,
            aux_neutral_field=args.aux_neutral_field,
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
                "source_file": primary.name,
                "aux_source_file": secondary.name,
                "id": record.get("id", ""),
                "query": record.get(args.query_field, ""),
                "gold_label": gold_label,
                "response_label": prediction.label,
                "response_score": prediction.score,
            }
            for text_field in (args.response_field, args.neutral_field, args.aux_neutral_field):
                if text_field in record:
                    output_row[text_field] = record[text_field]
            output_row.update(prediction.feature_scores)

            file_gold_labels.append(gold_label)
            file_predicted_labels.append(prediction.label)
            file_output_rows.append(output_row)

        predictions_path = results_dir / f"{primary.stem}-predictions.jsonl"
        write_jsonl(predictions_path, file_output_rows)
        all_output_rows.extend(file_output_rows)
        all_gold_labels.extend(file_gold_labels)
        all_predicted_labels.extend(file_predicted_labels)

        summary = metrics_dict(file_gold_labels, file_predicted_labels)
        summary["query_field"] = args.query_field
        summary["response_field"] = args.response_field
        summary["neutral_field"] = args.neutral_field
        summary["aux_neutral_field"] = args.aux_neutral_field
        file_summaries[primary.name] = summary
        positive_label = summary.get("positive_label")
        file_f1 = positive_label["f1"] if isinstance(positive_label, dict) else 0.0
        print(
            f"{primary.name}: accuracy={summary['accuracy']:.4f} "
            f"f1={file_f1:.4f} weighted_f1={summary['weighted']['f1']:.4f}"
        )

    overall_summary = metrics_dict(all_gold_labels, all_predicted_labels)
    summary_payload = {
        "scoring_backend": "anchor_distance_classifier",
        "embedding_model_name": args.embedding_model_name,
        "model_dir": str(model_dir),
        "results_dir": str(results_dir),
        "device": device,
        "query_field": args.query_field,
        "response_field": args.response_field,
        "neutral_field": args.neutral_field,
        "aux_neutral_field": args.aux_neutral_field,
        "score_granularity": args.score_granularity,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_metric": args.threshold_metric,
        "calibration_input_files": list(args.calibration_input_files),
        "aux_calibration_input_files": list(args.aux_calibration_input_files),
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
    misclassified_fields = ["source_file", "aux_source_file", "id", "query", "gold_label", "response_label", "response_score"]
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
        print(f"overall positive_f1={overall_positive_label['f1']:.4f}")


if __name__ == "__main__":
    main()
