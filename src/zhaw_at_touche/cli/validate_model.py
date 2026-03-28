from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any
from typing import Sequence

from zhaw_at_touche.constants import DEFAULT_MODELS_DIR, DEFAULT_RESULTS_DIR, DEFAULT_SETUP_NAME
from zhaw_at_touche.datasets import detect_generated_text_field
from zhaw_at_touche.evaluation_utils import (
    compute_metrics,
    counts_from_pairs,
    metrics_dict,
    render_matrix,
    render_metrics,
    write_csv,
)
from zhaw_at_touche.jsonl import read_jsonl, write_jsonl
from zhaw_at_touche.validation_setups import DEFAULT_VALIDATION_SETUPS_DIR, load_setup_defaults


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


def maybe_detect_generated_field(records: list[dict[str, Any]], explicit_field: str | None) -> str | None:
    if explicit_field is not None:
        return detect_generated_text_field(records[0], explicit_field)

    try:
        return detect_generated_text_field(records[0], None)
    except ValueError:
        return None


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def base_defaults() -> dict[str, object]:
    return {
        "model_name": None,
        "model_dir": None,
        "results_dir": None,
        "eval_splits": ["test"],
        "input_files": None,
        "text_field": "response",
        "generated_field": None,
        "batch_size": 16,
        "max_length": 512,
        "threshold": 0.5,
        "device": None,
    }


def build_parser(setup_defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    defaults = base_defaults()
    if setup_defaults:
        defaults.update(setup_defaults)
    input_file_defaults = defaults["input_files"]
    if input_file_defaults is None:
        input_file_defaults = [str(path) for path in resolve_default_eval_paths(defaults["eval_splits"])]

    parser = argparse.ArgumentParser(description="Validate a trained model and write evaluation artifacts.")
    parser.add_argument("--setup-name", default=DEFAULT_SETUP_NAME)
    parser.add_argument(
        "--setups-dir",
        default=str(DEFAULT_VALIDATION_SETUPS_DIR),
        help="Directory containing optional <setup-name>.json evaluation defaults.",
    )
    parser.add_argument(
        "--model-name",
        default=defaults["model_name"],
        help="Remote or local Hugging Face model reference for evaluation.",
    )
    parser.add_argument("--model-dir", default=defaults["model_dir"])
    parser.add_argument("--results-dir", default=defaults["results_dir"])
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        choices=("validation", "test"),
        default=defaults["eval_splits"],
        help="Default evaluation splits when --input-files is not passed. Defaults to test only.",
    )
    parser.add_argument("--input-files", nargs="+", default=input_file_defaults)
    parser.add_argument("--text-field", default=defaults["text_field"], help="Text field used for the main prediction.")
    parser.add_argument(
        "--generated-field",
        default=defaults["generated_field"],
        help="Optional generated text field to score for false-positive monitoring.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--max-length", type=int, default=defaults["max_length"])
    parser.add_argument("--threshold", type=float, default=defaults["threshold"])
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=defaults["device"])
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--setup-name", default=DEFAULT_SETUP_NAME)
    pre_parser.add_argument("--setups-dir", default=str(DEFAULT_VALIDATION_SETUPS_DIR))
    pre_args, _ = pre_parser.parse_known_args(argv)

    setup_defaults = load_setup_defaults(
        setup_name=pre_args.setup_name,
        setups_dir=Path(pre_args.setups_dir),
    )
    setup_provided_input_files = "input_files" in setup_defaults
    parser = build_parser(setup_defaults)
    args = parser.parse_args(argv)

    provided_model_name = any(
        token == "--model-name" or token.startswith("--model-name=")
        for token in raw_argv
    )
    provided_model_dir = any(
        token == "--model-dir" or token.startswith("--model-dir=")
        for token in raw_argv
    )
    if provided_model_name and provided_model_dir:
        parser.error("Pass only one of --model-name or --model-dir.")
    if provided_model_name:
        args.model_dir = None
    if provided_model_dir:
        args.model_name = None
    provided_input_files = any(
        token == "--input-files" or token.startswith("--input-files=")
        for token in raw_argv
    )
    if not provided_input_files and not setup_provided_input_files:
        args.input_files = [str(path) for path in resolve_default_eval_paths(args.eval_splits)]
    return args


def resolve_model_source(args: argparse.Namespace) -> str | Path:
    if args.model_name:
        return str(args.model_name)
    if args.model_dir:
        return Path(args.model_dir)
    return DEFAULT_MODELS_DIR / args.setup_name


def main() -> None:
    from zhaw_at_touche.modeling import load_model_reference, predict_with_bundle, resolve_device

    args = parse_args()
    model_source = resolve_model_source(args)
    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR / args.setup_name
    results_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"loading model from {model_source}")
    tokenizer, model = load_model_reference(model_source, device)

    all_output_rows: list[dict[str, Any]] = []
    all_gold_labels: list[int] = []
    all_predicted_labels: list[int] = []
    file_summaries: dict[str, Any] = {}

    for raw_path in args.input_files:
        path = Path(raw_path)
        records = list(read_jsonl(path))
        if not records:
            raise ValueError(f"Input file is empty: {path}")

        generated_field = maybe_detect_generated_field(records, args.generated_field)
        response_predictions = predict_with_bundle(
            tokenizer=tokenizer,
            model=model,
            records=records,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            text_key=args.text_field,
            threshold=args.threshold,
        )
        generated_predictions = None
        if generated_field:
            generated_predictions = predict_with_bundle(
                tokenizer=tokenizer,
                model=model,
                records=records,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                text_key=generated_field,
                threshold=args.threshold,
            )

        file_gold_labels: list[int] = []
        file_predicted_labels: list[int] = []
        file_output_rows: list[dict[str, Any]] = []

        for index, record in enumerate(records):
            gold_label = int(record["label"])
            response_prediction = response_predictions[index]
            output_row = {
                "source_file": path.name,
                "id": record.get("id", ""),
                "query": record.get("query", ""),
                "gold_label": gold_label,
                "response_label": response_prediction.label,
                "response_ad_prob": response_prediction.ad_prob,
            }
            if "response" in record:
                output_row["response"] = record["response"]
            if generated_field and generated_predictions is not None:
                generated_prediction = generated_predictions[index]
                output_row[f"{generated_field}_label"] = generated_prediction.label
                output_row[f"{generated_field}_ad_prob"] = generated_prediction.ad_prob

            file_gold_labels.append(gold_label)
            file_predicted_labels.append(response_prediction.label)
            file_output_rows.append(output_row)

        predictions_path = results_dir / f"{path.stem}-predictions.jsonl"
        write_jsonl(predictions_path, file_output_rows)
        all_output_rows.extend(file_output_rows)
        all_gold_labels.extend(file_gold_labels)
        all_predicted_labels.extend(file_predicted_labels)

        summary = metrics_dict(file_gold_labels, file_predicted_labels)
        summary["generated_field"] = generated_field
        if generated_field and generated_predictions is not None:
            generated_positive_count = sum(prediction.label for prediction in generated_predictions)
            summary["generated_positive_rate"] = generated_positive_count / len(generated_predictions)
        file_summaries[path.name] = summary
        positive_label = summary.get("positive_label")
        file_f1 = positive_label["f1"] if isinstance(positive_label, dict) else 0.0
        print(
            f"{path.name}: accuracy={summary['accuracy']:.4f} "
            f"f1={file_f1:.4f} weighted_f1={summary['weighted']['f1']:.4f}"
        )

    overall_summary = metrics_dict(all_gold_labels, all_predicted_labels)
    summary_payload = {
        "model_source": str(model_source),
        "results_dir": str(results_dir),
        "device": device,
        "text_field": args.text_field,
        "threshold": args.threshold,
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
    misclassified_fields = ["source_file", "id", "query", "response", "gold_label", "response_label", "response_ad_prob"]
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
