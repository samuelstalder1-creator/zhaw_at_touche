from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TASK_DIR = Path("data/task")
DEFAULT_PREPROCESSED_DIR = DEFAULT_TASK_DIR / "preprocessed"
DEFAULT_GENERATED_DIR = Path("data/generated")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_RESULTS_DIR = Path("results")

DEFAULT_RESPONSE_SPLITS = ("train", "validation", "test")
DEFAULT_TASK_CATEGORIES = ("responses", "sentence-pairs", "tokens")

DEFAULT_SETUP_NAME = "setupX"
DEFAULT_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_QWEN_API_BASE = "http://127.0.0.1:8000/v1"

BASE_GENERATED_FIELDS = {
    "ad_num",
    "advertiser",
    "gold_label",
    "id",
    "item",
    "label",
    "meta_topic",
    "query",
    "response",
    "response_ad_prob",
    "response_label",
    "search_engine",
    "sentence_spans",
    "source_file",
    "spans",
}
