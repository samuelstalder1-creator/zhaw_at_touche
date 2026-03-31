from __future__ import annotations

import json
import math
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from .datasets import DEFAULT_INPUT_FORMAT, build_model_input
from .evaluation_utils import metrics_dict, validation_metrics_payload
from .jsonl import append_jsonl, read_jsonl


def resolve_device(requested_device: str | None) -> str:
    if requested_device is not None:
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested device 'cuda' is not available.")
        if requested_device == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise ValueError("Requested device 'mps' is not available.")
        return requested_device

    if torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"


def autocast_context(device: str):
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def model_inputs(batch: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor) and key != "labels"
    }


def load_jsonl_rows(file_path: Path) -> list[dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    records = list(read_jsonl(file_path))
    if not records:
        raise ValueError(f"Dataset file is empty: {file_path}")
    return records


def load_tokenizer_from_pretrained(model_source: str | Path):
    try:
        return AutoTokenizer.from_pretrained(model_source)
    except (ImportError, ValueError) as exc:
        error_message = str(exc).lower()
        if "sentencepiece" not in error_message and "tiktoken" not in error_message:
            raise
        raise RuntimeError(
            f"Tokenizer for '{model_source}' could not be loaded because an optional tokenizer "
            "backend is missing. Reinstall the project dependencies so `sentencepiece` is "
            "available, for example with `uv sync` or `.venv/bin/pip install -e .`, then retry."
        ) from exc


def limit_records(records: Sequence[dict[str, Any]], max_rows: int | None) -> list[dict[str, Any]]:
    if max_rows is None or max_rows == 0:
        return list(records)
    if max_rows < 0:
        raise ValueError("--max-train-rows must be >= 0.")
    return list(records[:max_rows])


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    train_path: Path
    output_dir: Path
    max_length: int
    epochs: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    optimizer_eps: float
    lr_scheduler: str
    warmup_ratio: float
    max_grad_norm: float | None
    gradient_checkpointing: bool
    device: str
    max_train_rows: int | None = None
    input_format: str = DEFAULT_INPUT_FORMAT
    reference_field: str | None = None
    reference_label: str = "GEMINI"
    pad_to_max_length: bool = False
    positive_class_weight_scale: float = 2.0
    validation_path: Path | None = None
    wandb_enabled: bool = True
    wandb_project: str | None = "zhaw-at-touche-training"
    wandb_dir: Path | None = None
    wandb_run_name: str | None = None


class ResponseClassificationDataset(Dataset):
    def __init__(self, file_path: Path, max_rows: int | None = None):
        self.records = limit_records(load_jsonl_rows(file_path), max_rows)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def record_input_text(
    record: dict[str, Any],
    *,
    text_key: str,
    input_format: str,
    reference_field: str | None,
    reference_label: str,
) -> str:
    query = record.get("query")
    response = record.get(text_key)
    if not isinstance(query, str):
        query = ""
    if not isinstance(response, str) or not response.strip():
        raise ValueError(
            f"Record {record.get('id', '<unknown>')} is missing a valid '{text_key}' field."
        )

    reference_response: str | None = None
    if input_format != DEFAULT_INPUT_FORMAT:
        if not reference_field:
            raise ValueError(f"Input format '{input_format}' requires a reference field.")
        reference_value = record.get(reference_field)
        if not isinstance(reference_value, str) or not reference_value.strip():
            raise ValueError(
                f"Record {record.get('id', '<unknown>')} is missing a valid '{reference_field}' field."
            )
        reference_response = reference_value

    return build_model_input(
        query,
        response,
        input_format=input_format,
        reference_response=reference_response,
        reference_label=reference_label,
    )


class InstructionCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        text_key: str = "response",
        input_format: str = DEFAULT_INPUT_FORMAT,
        reference_field: str | None = None,
        reference_label: str = "GEMINI",
        pad_to_max_length: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        self.input_format = input_format
        self.reference_field = reference_field
        self.reference_label = reference_label
        self.pad_to_max_length = pad_to_max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        for record in batch:
            texts.append(
                record_input_text(
                    record,
                    text_key=self.text_key,
                    input_format=self.input_format,
                    reference_field=self.reference_field,
                    reference_label=self.reference_label,
                )
            )

        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length" if self.pad_to_max_length else True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([int(record["label"]) for record in batch], dtype=torch.long)
        return {**tokenized, "labels": labels, "raw_data": batch}


def build_class_weights(
    records: Sequence[dict[str, Any]],
    device: str,
    positive_class_weight_scale: float = 2.0,
) -> torch.Tensor:
    counts = Counter(int(record["label"]) for record in records)
    negative_count = counts.get(0, 0)
    positive_count = counts.get(1, 0)

    if negative_count == 0 or positive_count == 0:
        return torch.ones(2, device=device)

    positive_weight = (negative_count / positive_count) * positive_class_weight_scale
    return torch.tensor([1.0, positive_weight], device=device)


def evaluate_records(
    *,
    tokenizer,
    model,
    records: Sequence[dict[str, Any]],
    device: str,
    batch_size: int,
    max_length: int,
    text_key: str,
    input_format: str,
    reference_field: str | None,
    reference_label: str,
    pad_to_max_length: bool,
    loss_function,
) -> dict[str, float | int | dict[str, Any] | None]:
    if not records:
        raise ValueError("Validation records must not be empty.")

    collator = InstructionCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        text_key=text_key,
        input_format=input_format,
        reference_field=reference_field,
        reference_label=reference_label,
        pad_to_max_length=pad_to_max_length,
    )
    loader = DataLoader(
        list(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    total_loss = 0.0
    batch_count = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = model_inputs(batch, device)
            labels = batch["labels"].to(device)
            with autocast_context(device):
                logits = model(**inputs).logits
                loss = loss_function(logits, labels)
            predictions = torch.argmax(logits, dim=-1)

            total_loss += float(loss.item())
            batch_count += 1
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    summary = metrics_dict(all_labels, all_predictions)
    return validation_metrics_payload(
        loss=total_loss / max(batch_count, 1),
        summary=summary,
    )


def maybe_init_wandb(config: TrainingConfig):
    if not config.wandb_enabled:
        return None
    if not config.wandb_project:
        raise ValueError("--wandb-project is required when W&B logging is enabled.")

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install the project dependencies or pass --no-wandb."
        ) from exc

    wandb_dir = config.wandb_dir or (config.output_dir / "wandb")
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        dir=str(wandb_dir),
        mode="online",
    )
    run.config.update(
        {
            "model_name": config.model_name,
            "train_path": str(config.train_path),
            "validation_path": str(config.validation_path) if config.validation_path else None,
            "output_dir": str(config.output_dir),
            "max_length": config.max_length,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "grad_accum": config.grad_accum,
            "learning_rate": config.learning_rate,
            "optimizer_eps": config.optimizer_eps,
            "lr_scheduler": config.lr_scheduler,
            "warmup_ratio": config.warmup_ratio,
            "max_grad_norm": config.max_grad_norm,
            "gradient_checkpointing": config.gradient_checkpointing,
            "device": config.device,
            "max_train_rows": config.max_train_rows,
            "input_format": config.input_format,
            "reference_field": config.reference_field,
            "reference_label": config.reference_label,
            "pad_to_max_length": config.pad_to_max_length,
            "positive_class_weight_scale": config.positive_class_weight_scale,
        }
    )
    return run


def maybe_init_scheduler(
    *,
    optimizer,
    scheduler_name: str,
    warmup_ratio: float,
    total_steps: int,
):
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine_with_warmup":
        warmup_steps = int(total_steps * warmup_ratio)
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    raise ValueError(f"Unsupported scheduler '{scheduler_name}'.")


def train_model(config: TrainingConfig) -> dict[str, Any]:
    if config.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1.")
    if config.warmup_ratio < 0 or config.warmup_ratio >= 1:
        raise ValueError("--warmup-ratio must be >= 0 and < 1.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = config.output_dir / "training_metrics.jsonl"
    if metrics_log_path.exists():
        metrics_log_path.unlink()

    tokenizer = load_tokenizer_from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
    ).to(config.device)
    if config.gradient_checkpointing:
        gradient_checkpointing_enable = getattr(model, "gradient_checkpointing_enable", None)
        if gradient_checkpointing_enable is None:
            raise ValueError(f"Model '{config.model_name}' does not support gradient checkpointing.")
        gradient_checkpointing_enable()

    train_dataset = ResponseClassificationDataset(config.train_path, max_rows=config.max_train_rows)
    collator = InstructionCollator(
        tokenizer=tokenizer,
        max_length=config.max_length,
        input_format=config.input_format,
        reference_field=config.reference_field,
        reference_label=config.reference_label,
        pad_to_max_length=config.pad_to_max_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    class_weights = build_class_weights(
        train_dataset.records,
        config.device,
        positive_class_weight_scale=config.positive_class_weight_scale,
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        eps=config.optimizer_eps,
    )
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum)
    total_optimizer_steps = max(optimizer_steps_per_epoch * config.epochs, 1)
    scheduler = maybe_init_scheduler(
        optimizer=optimizer,
        scheduler_name=config.lr_scheduler,
        warmup_ratio=config.warmup_ratio,
        total_steps=total_optimizer_steps,
    )
    validation_records = (
        load_jsonl_rows(config.validation_path)
        if config.validation_path is not None
        else None
    )
    epoch_losses: list[float] = []
    validation_history: list[dict[str, float | int | None]] = []
    best_validation_metrics: dict[str, float | int | None] | None = None
    global_step = 0
    wandb_run = None
    try:
        wandb_run = maybe_init_wandb(config)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "dataset/train_rows": len(train_dataset),
                    "dataset/validation_rows": len(validation_records) if validation_records is not None else 0,
                    "train/class_weight_negative": class_weights[0].item(),
                    "train/class_weight_positive": class_weights[1].item(),
                    "train/optimizer_steps_total": total_optimizer_steps,
                },
                step=0,
            )

        for epoch in range(config.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

            for step, batch in enumerate(progress, start=1):
                inputs = model_inputs(batch, config.device)
                labels = batch["labels"].to(config.device)

                with autocast_context(config.device):
                    logits = model(**inputs).logits
                    loss = loss_function(logits, labels) / config.grad_accum

                loss.backward()
                batch_loss = float(loss.item() * config.grad_accum)
                running_loss += batch_loss
                global_step += 1

                step_metrics = {
                    "phase": "train",
                    "event": "step",
                    "epoch": epoch,
                    "step": global_step,
                    "batch_loss": batch_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                append_jsonl(metrics_log_path, step_metrics)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": batch_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                            "train/step": global_step,
                        }
                    )

                if step % config.grad_accum == 0 or step == len(train_loader):
                    if config.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                progress.set_postfix(loss=f"{batch_loss:.4f}")

            epoch_loss = running_loss / max(len(train_loader), 1)
            epoch_losses.append(epoch_loss)
            epoch_metrics = {
                "phase": "train",
                "event": "epoch",
                "epoch": epoch,
                "epoch_loss": epoch_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_jsonl(metrics_log_path, epoch_metrics)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/epoch_loss": epoch_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                )

            if validation_records is not None:
                validation_metrics = evaluate_records(
                    tokenizer=tokenizer,
                    model=model,
                    records=validation_records,
                    device=config.device,
                    batch_size=config.batch_size,
                    max_length=config.max_length,
                    text_key="response",
                    input_format=config.input_format,
                    reference_field=config.reference_field,
                    reference_label=config.reference_label,
                    pad_to_max_length=config.pad_to_max_length,
                    loss_function=loss_function,
                )
                validation_entry = {
                    "phase": "validation",
                    "event": "epoch",
                    "epoch": epoch,
                    **validation_metrics,
                }
                validation_history.append(validation_entry)
                append_jsonl(metrics_log_path, validation_entry)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "val/loss": validation_metrics["loss"],
                            "val/acc": validation_metrics["accuracy"],
                            "val/positive_f1": validation_metrics["positive_f1"],
                            "val/positive_precision": validation_metrics["positive_precision"],
                            "val/positive_recall": validation_metrics["positive_recall"],
                            "val/f1": validation_metrics["positive_f1"],
                            "val/macro_precision": validation_metrics["macro_precision"],
                            "val/macro_recall": validation_metrics["macro_recall"],
                            "val/macro_f1": validation_metrics["macro_f1"],
                            "val/weighted_precision": validation_metrics["weighted_precision"],
                            "val/weighted_recall": validation_metrics["weighted_recall"],
                            "val/weighted_f1": validation_metrics["weighted_f1"],
                            "val/tn": validation_metrics["true_negative"],
                            "val/fp": validation_metrics["false_positive"],
                            "val/fn": validation_metrics["false_negative"],
                            "val/tp": validation_metrics["true_positive"],
                            "val/predicted_positive_rate": validation_metrics["predicted_positive_rate"],
                            "val/gold_positive_rate": validation_metrics["gold_positive_rate"],
                            "val/samples": validation_metrics["samples"],
                            "epoch": epoch,
                        }
                    )

                current_score = validation_metrics["positive_f1"]
                if current_score is None:
                    current_score = validation_metrics["weighted_f1"]
                best_score = (
                    best_validation_metrics["positive_f1"]
                    if best_validation_metrics is not None
                    else None
                )
                if best_score is None and best_validation_metrics is not None:
                    best_score = best_validation_metrics["weighted_f1"]
                if current_score is not None and (best_score is None or current_score > best_score):
                    best_validation_metrics = {
                        "epoch": epoch,
                        **validation_metrics,
                    }
                    if wandb_run is not None:
                        wandb_run.summary.update(
                            {
                                "best_val/epoch": epoch,
                                "best_val/loss": validation_metrics["loss"],
                                "best_val/acc": validation_metrics["accuracy"],
                                "best_val/positive_f1": validation_metrics["positive_f1"],
                                "best_val/positive_precision": validation_metrics["positive_precision"],
                                "best_val/positive_recall": validation_metrics["positive_recall"],
                                "best_val/weighted_f1": validation_metrics["weighted_f1"],
                            }
                        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    summary = {
        "model_name": config.model_name,
        "train_path": str(config.train_path),
        "output_dir": str(config.output_dir),
        "device": config.device,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "learning_rate": config.learning_rate,
        "optimizer_eps": config.optimizer_eps,
        "lr_scheduler": config.lr_scheduler,
        "warmup_ratio": config.warmup_ratio,
        "max_grad_norm": config.max_grad_norm,
        "gradient_checkpointing": config.gradient_checkpointing,
        "max_length": config.max_length,
        "input_format": config.input_format,
        "reference_field": config.reference_field,
        "reference_label": config.reference_label,
        "pad_to_max_length": config.pad_to_max_length,
        "positive_class_weight_scale": config.positive_class_weight_scale,
        "validation_path": str(config.validation_path) if config.validation_path else None,
        "wandb_enabled": config.wandb_enabled,
        "wandb_project": config.wandb_project,
        "wandb_dir": str(config.wandb_dir or (config.output_dir / "wandb")) if config.wandb_enabled else None,
        "wandb_run_name": config.wandb_run_name,
        "metrics_log_path": str(metrics_log_path),
        "train_rows": len(train_dataset),
        "epoch_losses": epoch_losses,
        "validation_history": validation_history,
        "best_validation_metrics": best_validation_metrics,
    }
    summary_path = config.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@dataclass(frozen=True)
class Prediction:
    label: int
    ad_prob: float


def load_model_bundle(model_dir: Path, device: str):
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = load_tokenizer_from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model


def load_model_reference(model_source: str | Path, device: str):
    if isinstance(model_source, Path):
        return load_model_bundle(model_source, device)

    tokenizer = load_tokenizer_from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source).to(device)
    model.eval()
    return tokenizer, model


def predict_records(
    *,
    model_dir: Path,
    records: Sequence[dict[str, Any]],
    device: str,
    batch_size: int,
    max_length: int,
    text_key: str,
    threshold: float,
    input_format: str = DEFAULT_INPUT_FORMAT,
    reference_field: str | None = None,
    reference_label: str = "GEMINI",
    pad_to_max_length: bool = False,
) -> list[Prediction]:
    tokenizer, model = load_model_bundle(model_dir, device)
    return predict_with_bundle(
        tokenizer=tokenizer,
        model=model,
        records=records,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        text_key=text_key,
        threshold=threshold,
        input_format=input_format,
        reference_field=reference_field,
        reference_label=reference_label,
        pad_to_max_length=pad_to_max_length,
    )


def predict_with_bundle(
    *,
    tokenizer,
    model,
    records: Sequence[dict[str, Any]],
    device: str,
    batch_size: int,
    max_length: int,
    text_key: str,
    threshold: float,
    input_format: str = DEFAULT_INPUT_FORMAT,
    reference_field: str | None = None,
    reference_label: str = "GEMINI",
    pad_to_max_length: bool = False,
) -> list[Prediction]:
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    predictions: list[Prediction] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts = [
                record_input_text(
                    record,
                    text_key=text_key,
                    input_format=input_format,
                    reference_field=reference_field,
                    reference_label=reference_label,
                )
                for record in batch
            ]

            tokenized = tokenizer(
                texts,
                truncation=True,
                padding="max_length" if pad_to_max_length else True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in tokenized.items()}
            with autocast_context(device):
                logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            for probability in probabilities:
                label = 1 if probability >= threshold else 0
                predictions.append(Prediction(label=label, ad_prob=float(probability)))

    return predictions
