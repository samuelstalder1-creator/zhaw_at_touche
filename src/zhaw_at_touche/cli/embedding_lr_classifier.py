from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from zhaw_at_touche.constants import DEFAULT_MODELS_DIR, DEFAULT_RESULTS_DIR, DEFAULT_SETUP_NAME
from zhaw_at_touche.embedding_divergence import calibrate_threshold, load_embedding_model
from zhaw_at_touche.embedding_lr_classifier import (
    ALL_TRAINER_TYPES,
    DUAL_FILE_TRAINERS,
    load_bundle,
    load_state,
    score_records,
)
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
from zhaw_at_touche.anchor_distance_classifier import load_merged_records
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
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[str(l) for l in labels])
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
    return any(token == option_name or token.startswith(f"{option_name}=") for token in raw_argv)


def base_defaults() -> dict[str, object]:
    return {
        "model_dir": None,
        "results_dir": None,
        "input_files": ["data/generated/gemini/responses-test-with-neutral_gemini.jsonl"],
        "aux_input_files": None,
        "calibration_input_files": ["data/generated/gemini/responses-validation-with-neutral_gemini.jsonl"],
        "aux_calibration_input_files": None,
        "embedding_model_name": "sentence-transformers/all-mpnet-base-v2",
        "response_field": "response",
        "neutral_field": "gemini25flashlite",
        "aux_neutral_field": None,
        "query_field": "query",
        "threshold": None,
        "threshold_metric": "macro_f1",
        "batch_size": 32,
        "max_length": 512,
        "device": None,
    }


def build_parser(setup_defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    defaults = base_defaults()
    if setup_defaults:
        defaults.update(setup_defaults)

    parser = argparse.ArgumentParser(description="Evaluate an embedding LR classifier.")
    parser.add_argument("--setup-name", default=DEFAULT_SETUP_NAME)
    parser.add_argument("--setups-dir", default=str(DEFAULT_VALIDATION_SETUPS_DIR))
    parser.add_argument("--model-dir", default=defaults["model_dir"])
    parser.add_argument("--results-dir", default=defaults["results_dir"])
    parser.add_argument("--input-files", nargs="+", default=defaults["input_files"])
    parser.add_argument("--aux-input-files", nargs="+", default=defaults["aux_input_files"])
    parser.add_argument("--calibration-input-files", nargs="+", default=defaults["calibration_input_files"])
    parser.add_argument("--aux-calibration-input-files", nargs="+", default=defaults["aux_calibration_input_files"])
    parser.add_argument("--embedding-model-name", default=defaults["embedding_model_name"])
    parser.add_argument("--response-field", default=defaults["response_field"])
    parser.add_argument("--neutral-field", default=defaults["neutral_field"])
    parser.add_argument("--aux-neutral-field", default=defaults["aux_neutral_field"])
    parser.add_argument("--query-field", default=defaults["query_field"])
    parser.add_argument("--threshold", type=float, default=defaults["threshold"])
    parser.add_argument("--threshold-metric", choices=("positive_f1", "macro_f1", "accuracy"), default=defaults["threshold_metric"])
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
    setup_defaults = load_setup_defaults(setup_name=pre_args.setup_name, setups_dir=Path(pre_args.setups_dir))
    parser = build_parser(setup_defaults)
    return parser.parse_args(raw_argv)


def _load_records_for_eval(
    trainer_type: str,
    primary_path: Path,
    secondary_path: Path | None,
) -> list[dict[str, Any]]:
    if trainer_type in DUAL_FILE_TRAINERS:
        if not secondary_path:
            raise ValueError(f"{trainer_type} requires aux input files.")
        return load_merged_records(primary_path, secondary_path)
    return list(read_jsonl(primary_path))


def main(argv: Sequence[str] | None = None) -> None:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(raw_argv)

    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR / args.setup_name
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_MODELS_DIR / args.setup_name
    device = resolve_device(args.device)

    saved_state = load_state(model_dir)
    trainer_type = saved_state.get("trainer_type") if saved_state else None
    if not trainer_type:
        raise ValueError(f"Could not determine trainer_type from saved state in {model_dir}.")
    if trainer_type not in ALL_TRAINER_TYPES:
        raise ValueError(f"Unsupported trainer_type '{trainer_type}' in saved state.")

    # Restore fields from saved state if not overridden on CLI
    state_field_map = {
        "--embedding-model-name": "embedding_model_name",
        "--response-field": "response_field",
        "--neutral-field": "neutral_field",
        "--aux-neutral-field": "aux_neutral_field",
        "--query-field": "query_field",
        "--threshold-metric": "threshold_metric",
        "--batch-size": "batch_size",
        "--max-length": "max_length",
    }
    if saved_state:
        for option_name, state_key in state_field_map.items():
            if not cli_option_was_provided(raw_argv, option_name):
                state_value = saved_state.get(state_key)
                if state_value is not None:
                    setattr(args, state_key.replace("-", "_"), state_value)

    classifier = load_bundle(model_dir)
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
        cal_files = list(args.calibration_input_files or [])
        aux_cal_files = list(args.aux_calibration_input_files or [])
        calibration_records: list[dict[str, Any]] = []
        for i, cal_path in enumerate(cal_files):
            aux_path = Path(aux_cal_files[i]) if i < len(aux_cal_files) else None
            calibration_records.extend(
                _load_records_for_eval(trainer_type, Path(cal_path), aux_path)
            )
        if not calibration_records:
            raise ValueError("Calibration input files must contain at least one labeled record.")
        cal_predictions = score_records(
            classifier=classifier, tokenizer=tokenizer, model=model,
            records=calibration_records, trainer_type=trainer_type,
            response_field=args.response_field, neutral_field=args.neutral_field,
            aux_neutral_field=args.aux_neutral_field, query_field=args.query_field,
            threshold=0.5, device=device, batch_size=args.batch_size,
            max_length=args.max_length, progress_prefix=f"{trainer_type} calibration",
        )
        cal_scores = [p.score for p in cal_predictions]
        cal_labels = [int(r["label"]) for r in calibration_records]
        threshold, calibration_summary = calibrate_threshold(
            cal_scores, cal_labels, threshold_metric=args.threshold_metric
        )
        threshold_source = "validation_calibration"

    print(f"using threshold={threshold:.6f} ({threshold_source})")

    all_output_rows: list[dict[str, Any]] = []
    all_gold_labels: list[int] = []
    all_predicted_labels: list[int] = []
    file_summaries: dict[str, Any] = {}

    input_files = list(args.input_files or [])
    aux_input_files = list(args.aux_input_files or [])

    for i, raw_path in enumerate(input_files):
        primary = Path(raw_path)
        aux_path = Path(aux_input_files[i]) if i < len(aux_input_files) else None
        records = _load_records_for_eval(trainer_type, primary, aux_path)
        if not records:
            raise ValueError(f"Input file is empty: {primary}")

        predictions = score_records(
            classifier=classifier, tokenizer=tokenizer, model=model,
            records=records, trainer_type=trainer_type,
            response_field=args.response_field, neutral_field=args.neutral_field,
            aux_neutral_field=args.aux_neutral_field, query_field=args.query_field,
            threshold=threshold, device=device, batch_size=args.batch_size,
            max_length=args.max_length, progress_prefix=f"{trainer_type} eval {primary.stem}",
        )

        file_gold_labels: list[int] = []
        file_predicted_labels: list[int] = []
        file_output_rows: list[dict[str, Any]] = []

        for index, record in enumerate(records):
            gold_label = int(record["label"])
            prediction = predictions[index]
            output_row: dict[str, Any] = {
                "source_file": primary.name,
                "id": record.get("id", ""),
                "query": record.get(args.query_field, ""),
                "gold_label": gold_label,
                "response_label": prediction.label,
                "response_score": prediction.score,
            }
            for field in (args.response_field, args.neutral_field, args.aux_neutral_field):
                if field and field in record:
                    output_row[field] = record[field]
            file_gold_labels.append(gold_label)
            file_predicted_labels.append(prediction.label)
            file_output_rows.append(output_row)

        predictions_path = results_dir / f"{primary.stem}-predictions.jsonl"
        write_jsonl(predictions_path, file_output_rows)
        all_output_rows.extend(file_output_rows)
        all_gold_labels.extend(file_gold_labels)
        all_predicted_labels.extend(file_predicted_labels)

        summary = metrics_dict(file_gold_labels, file_predicted_labels)
        summary["response_field"] = args.response_field
        summary["neutral_field"] = args.neutral_field
        summary["aux_neutral_field"] = args.aux_neutral_field
        file_summaries[primary.name] = summary
        positive_label = summary.get("positive_label")
        file_f1 = positive_label["f1"] if isinstance(positive_label, dict) else 0.0
        print(f"{primary.name}: accuracy={summary['accuracy']:.4f} f1={file_f1:.4f}")

    overall_summary = metrics_dict(all_gold_labels, all_predicted_labels)
    summary_payload = {
        "scoring_backend": trainer_type,
        "embedding_model_name": args.embedding_model_name,
        "model_dir": str(model_dir),
        "results_dir": str(results_dir),
        "device": device,
        "response_field": args.response_field,
        "neutral_field": args.neutral_field,
        "aux_neutral_field": args.aux_neutral_field,
        "query_field": args.query_field,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_metric": args.threshold_metric,
        "input_files": input_files,
        "aux_input_files": aux_input_files,
        "calibration_input_files": list(args.calibration_input_files or []),
        "aux_calibration_input_files": list(args.aux_calibration_input_files or []),
        "calibration_summary": calibration_summary,
        "files": file_summaries,
        "overall": overall_summary,
    }
    (results_dir / "metrics_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    write_csv_rows(
        results_dir / "standardized_results.csv",
        ["source_file", "id", "label", "pred"],
        [{"source_file": r["source_file"], "id": r["id"], "label": r["gold_label"], "pred": r["response_label"]} for r in all_output_rows],
    )

    misclassified_rows = [r for r in all_output_rows if int(r["gold_label"]) != int(r["response_label"])]
    base_fields = ["source_file", "id", "query", "gold_label", "response_label", "response_score"]
    extra_fields = sorted({f for row in misclassified_rows for f in row if f not in base_fields})
    write_csv_rows(results_dir / "misclassified_analysis.csv", base_fields + extra_fields, misclassified_rows)

    counts, labels, _ = counts_from_pairs(all_gold_labels, all_predicted_labels)
    per_label, macro, weighted = compute_metrics(counts, labels)
    (results_dir / "response_metrics.txt").write_text(
        f"rows: {len(all_gold_labels)}\n"
        "matrix: rows=gold_label, cols=response_label\n"
        f"{render_matrix(counts, labels)}\n"
        "metrics:\n"
        f"{render_metrics(per_label, macro, weighted)}\n",
        encoding="utf-8",
    )
    write_csv(results_dir / "confusion_matrix.csv", counts, labels)
    save_confusion_matrix_image(all_gold_labels, all_predicted_labels, results_dir / "confusion_matrix.png")

    print(f"overall accuracy={overall_summary['accuracy']:.4f}")
    overall_positive = overall_summary.get("positive_label")
    if isinstance(overall_positive, dict):
        print(f"overall f1={overall_positive['f1']:.4f}")
    print(f"artifacts written to {results_dir}")


if __name__ == "__main__":
    main()
