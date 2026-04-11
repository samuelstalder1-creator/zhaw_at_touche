from __future__ import annotations

import json
import random
import re
import sys
import time
from contextlib import nullcontext
from urllib import error as urllib_error
from urllib import request as urllib_request
from pathlib import Path
from typing import Any

from .jsonl import read_jsonl

SYSTEM_PROMPT = """Goal:
Write a helpful, factual answer to the user's query that matches the style of existing neutral responses.

Rules:
- Do not mention brand names, companies, vendors, product models, or specific services.
- Do not promote or recommend a specific item.
- Avoid marketing language, persuasion, links, or calls to action.
- Generic product or technical terms are allowed.

Style Requirements:
- Write in flowing prose using natural sentences and short paragraphs.
- Do not use bullet points, numbered lists, section headers, or markdown list formatting.
- Keep tone factual, balanced, and conversational.
- Return exactly one continuous paragraph.
- Do not output any newline characters.

Length:
- Target roughly 130-200 words unless the query is trivial.
"""

MODEL_ALIASES = {
    "gemini-1.5-flash": "gemini15flash",
    "gemini-2.5-flash-lite": "gemini25flashlite",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen",
    "gemma4:2b": "gemma4_2b",
    "gemma4:12b": "gemma4_12b",
    "gemma4:27b": "gemma4_27b",
    "gemma4:31b": "gemma4_31b",
    "gemma4:e4b": "gemma4_e4b",
}

BULLET_CHARS = "•◦▪●‣∙"
UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
LIST_PREFIX_RE = re.compile(
    rf"^\s*(?:[{re.escape(BULLET_CHARS)}]|[-*+]|(?:\d+|[a-zA-Z])[.)])\s+"
)


def model_alias(model_name: str) -> str:
    alias = MODEL_ALIASES.get(model_name)
    if alias:
        return alias
    return re.sub(r"[^a-zA-Z0-9]+", "", model_name).lower()


def default_backend_for_provider(provider: str) -> str:
    if provider == "gemini":
        return "gemini"
    if provider == "qwen":
        return "transformers"
    raise ValueError(
        f"Unsupported provider '{provider}'. Supported providers: gemini, qwen."
    )


def load_done_ids(out_path: Path, response_field: str) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done

    for row in read_jsonl(out_path):
        row_id = row.get("id")
        response_value = row.get(response_field)
        if isinstance(row_id, str) and isinstance(response_value, str) and response_value.strip():
            done.add(row_id)
    return done


def _decode_basic_unicode_escapes(text: str) -> str:
    return UNICODE_ESCAPE_RE.sub(lambda match: chr(int(match.group(1), 16)), text)


def clean_response_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = _decode_basic_unicode_escapes(cleaned)
    cleaned = cleaned.replace("\\n", "\n")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\t", " ")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    paragraphs: list[str] = []
    current_lines: list[str] = []
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line:
            if current_lines:
                paragraphs.append(" ".join(current_lines).strip())
                current_lines = []
            continue
        line = LIST_PREFIX_RE.sub("", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            current_lines.append(line)

    if current_lines:
        paragraphs.append(" ".join(current_lines).strip())

    one_flow_text = " ".join(paragraph for paragraph in paragraphs if paragraph).strip()
    one_flow_text = re.sub(r"\s+", " ", one_flow_text)
    return one_flow_text


def _deep_get_int(obj: Any, *path: str) -> int:
    current = obj
    for key in path:
        if current is None:
            return 0
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return int(current) if isinstance(current, (int, float)) else 0


def get_usage_counts(response: Any) -> dict[str, int]:
    input_tokens = _deep_get_int(response, "usage", "input_tokens")
    if input_tokens == 0:
        input_tokens = _deep_get_int(response, "usage_metadata", "prompt_token_count")
    if input_tokens == 0:
        input_tokens = _deep_get_int(response, "usage_metadata", "input_token_count")

    cached_input_tokens = _deep_get_int(
        response,
        "usage",
        "input_tokens_details",
        "cached_tokens",
    )
    if cached_input_tokens == 0:
        cached_input_tokens = _deep_get_int(
            response,
            "usage_metadata",
            "cached_content_token_count",
        )

    output_tokens = _deep_get_int(response, "usage", "output_tokens")
    if output_tokens == 0:
        output_tokens = _deep_get_int(response, "usage_metadata", "candidates_token_count")
    if output_tokens == 0:
        output_tokens = _deep_get_int(response, "usage_metadata", "output_token_count")

    total_tokens = _deep_get_int(response, "usage", "total_tokens")
    if total_tokens == 0:
        total_tokens = _deep_get_int(response, "usage_metadata", "total_token_count")

    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def generate_neutral_response_gemini(client: Any, model: str, query: str) -> tuple[str, dict[str, int]]:
    response = client.models.generate_content(
        model=model,
        contents=query.strip(),
        config={"system_instruction": SYSTEM_PROMPT},
    )
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        parts: list[str] = []
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            candidate_parts = getattr(content, "parts", None) or []
            for part in candidate_parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    parts.append(part_text.strip())
        text = " ".join(parts).strip()

    if not text:
        raise RuntimeError("Empty model output.")
    return text, get_usage_counts(response)


def build_chat_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query.strip()},
    ]


def get_transformers_usage_counts(*, input_tokens: int, output_tokens: int) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": 0,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def load_local_generation_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()
    return tokenizer, model


def generate_neutral_response_transformers(
    *,
    tokenizer,
    model,
    query: str,
    device: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, int]]:
    import torch

    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support chat templates for local generation.")

    prompt_text = tokenizer.apply_chat_template(
        build_chat_messages(query),
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenized = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in tokenized.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=model.dtype)
        if device == "cuda" and isinstance(model.dtype, torch.dtype)
        else nullcontext()
    )
    with torch.inference_mode():
        with autocast_context:
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    generated_ids = generated[:, input_length:]
    output_length = int(generated_ids.shape[-1])
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if not text:
        raise RuntimeError("Empty model output.")
    return text, get_transformers_usage_counts(
        input_tokens=input_length,
        output_tokens=output_length,
    )


def _openai_compatible_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return ""

    parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
            continue
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())

    return " ".join(parts).strip()


def get_openai_compatible_usage_counts(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}

    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": 0,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def generate_neutral_response_openai_compatible(
    *,
    api_base: str,
    api_key: str,
    model: str,
    query: str,
    timeout: float,
) -> tuple[str, dict[str, int]]:
    if not api_base:
        raise RuntimeError("OpenAI-compatible API base URL is required.")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query.strip()},
        ],
        "temperature": 0.2,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url=f"{api_base.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI-compatible request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"OpenAI-compatible request failed: {exc.reason}") from exc

    parsed = json.loads(raw)
    text = _openai_compatible_text(parsed)
    if not text:
        raise RuntimeError("Empty model output.")
    return text, get_openai_compatible_usage_counts(parsed)


def with_retries(function, *, max_retries: int = 6, base_sleep: float = 1.0):
    for attempt in range(max_retries + 1):
        try:
            return function()
        except Exception as exc:
            if attempt >= max_retries:
                raise
            sleep_seconds = base_sleep * (2**attempt) + random.random() * 0.25
            print(
                f"[warn] API call failed ({type(exc).__name__}: {exc}). "
                f"Retry in {sleep_seconds:.2f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)
