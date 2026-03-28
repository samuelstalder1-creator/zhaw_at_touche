from __future__ import annotations

import json
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .datasets import build_model_input
from .jsonl import read_jsonl


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


def load_jsonl_rows(file_path: Path) -> list[dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    records = list(read_jsonl(file_path))
    if not records:
        raise ValueError(f"Dataset file is empty: {file_path}")
    return records


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
    device: str
    max_train_rows: int | None = None


class ResponseClassificationDataset(Dataset):
    def __init__(self, file_path: Path, max_rows: int | None = None):
        self.records = limit_records(load_jsonl_rows(file_path), max_rows)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class InstructionCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int, text_key: str = "response"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        for record in batch:
            query = record.get("query")
            response = record.get(self.text_key)
            if not isinstance(query, str):
                query = ""
            if not isinstance(response, str) or not response.strip():
                raise ValueError(
                    f"Record {record.get('id', '<unknown>')} is missing a valid '{self.text_key}' field."
                )
            texts.append(build_model_input(query, response))

        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([int(record["label"]) for record in batch], dtype=torch.long)
        return {**tokenized, "labels": labels, "raw_data": batch}


def build_class_weights(records: Sequence[dict[str, Any]], device: str) -> torch.Tensor:
    counts = Counter(int(record["label"]) for record in records)
    negative_count = counts.get(0, 0)
    positive_count = counts.get(1, 0)

    if negative_count == 0 or positive_count == 0:
        return torch.ones(2, device=device)

    positive_weight = (negative_count / positive_count) * 2.0
    return torch.tensor([1.0, positive_weight], device=device)


def train_model(config: TrainingConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
    ).to(config.device)

    train_dataset = ResponseClassificationDataset(config.train_path, max_rows=config.max_train_rows)
    collator = InstructionCollator(tokenizer=tokenizer, max_length=config.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    class_weights = build_class_weights(train_dataset.records, config.device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    epoch_losses: list[float] = []
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for step, batch in enumerate(progress, start=1):
            inputs = {
                key: value.to(config.device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask"}
            }
            labels = batch["labels"].to(config.device)

            with autocast_context(config.device):
                logits = model(**inputs).logits
                loss = loss_function(logits, labels) / config.grad_accum

            loss.backward()
            running_loss += float(loss.item() * config.grad_accum)

            if step % config.grad_accum == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=f"{loss.item() * config.grad_accum:.4f}")

        epoch_losses.append(running_loss / max(len(train_loader), 1))

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
        "max_length": config.max_length,
        "train_rows": len(train_dataset),
        "epoch_losses": epoch_losses,
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

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model


def load_model_reference(model_source: str | Path, device: str):
    if isinstance(model_source, Path):
        return load_model_bundle(model_source, device)

    tokenizer = AutoTokenizer.from_pretrained(model_source)
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
) -> list[Prediction]:
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    predictions: list[Prediction] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts: list[str] = []
            for record in batch:
                query = record.get("query")
                response = record.get(text_key)
                if not isinstance(query, str):
                    query = ""
                if not isinstance(response, str) or not response.strip():
                    raise ValueError(
                        f"Record {record.get('id', '<unknown>')} is missing a valid '{text_key}' field."
                    )
                texts.append(build_model_input(query, response))

            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=True,
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
