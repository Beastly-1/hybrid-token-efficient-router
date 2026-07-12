import os
import json
from pathlib import Path

# ==========================================================
# Local Model (OpenVINO GenAI)
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Directory containing an OpenVINO-exported, NPU-compatible instruct model.
LOCAL_MODEL_PATH = os.getenv(
    "LOCAL_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "qwen2.5-1.5b-instruct-int4-ov"),
)
LOCAL_PREFERRED_DEVICE = os.getenv("LOCAL_PREFERRED_DEVICE", "NPU")
LOCAL_FALLBACK_DEVICE = os.getenv("LOCAL_FALLBACK_DEVICE", "CPU")
LOCAL_MODEL_CACHE_DIR = os.getenv(
    "LOCAL_MODEL_CACHE_DIR", str(PROJECT_ROOT / ".openvino_cache")
)
LOCAL_MAX_NEW_TOKENS = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "256"))
LOCAL_TIMEOUT_SECONDS = int(os.getenv("LOCAL_TIMEOUT_SECONDS", "120"))
LOCAL_NPU_GENERATE_HINT = os.getenv("LOCAL_NPU_GENERATE_HINT", "FAST_COMPILE")
LOCAL_MODEL_ENABLED = os.getenv("LOCAL_MODEL_ENABLED", "true").lower() == "true"
LOCAL_ACCEPT_DIFFICULTIES = {
    value.strip()
    for value in os.getenv("LOCAL_ACCEPT_DIFFICULTIES", "easy").split(",")
    if value.strip()
}

# Input limits keep the pre-router bounded on a laptop.
MAX_INPUT_CHARACTERS = int(os.getenv("MAX_INPUT_CHARACTERS", "12000"))
MAX_PROMPT_CHARACTERS = int(os.getenv("MAX_PROMPT_CHARACTERS", "14000"))

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = os.getenv(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
)
FIREWORKS_MODEL = os.getenv(
    "FIREWORKS_MODEL", "accounts/fireworks/models/minimax-m3"
)
FIREWORKS_TIMEOUT_SECONDS = int(os.getenv("FIREWORKS_TIMEOUT_SECONDS", "120"))
ALLOWED_MODELS = [
    model.strip()
    for model in os.getenv("ALLOWED_MODELS", "").split(",")
    if model.strip()
]
FIREWORKS_MODEL_CATALOG = json.loads(os.getenv("FIREWORKS_MODEL_CATALOG", "[]"))
MODEL_SELECTION_MODE = os.getenv("MODEL_SELECTION_MODE", "cost_first").strip().lower()

# ==========================================================
# Routing Thresholds
# ==========================================================

ROUTE_THRESHOLDS = {

    "math": 0.92,

    "logic": 0.94,

    "code_generation": 0.90,

    "code_debug": 0.90,

    "summarization": 0.85,

    "ner": 0.80,

    "sentiment": 0.75,

    "factual": 0.82,

    "general": 0.85,

    "default": 0.85,
}

# ==========================================================
# Confidence Evaluator
# ==========================================================

LOGPROB_WEIGHT = 0.40
ANSWER_WEIGHT = 0.60

# ==========================================================
# Self Consistency
# ==========================================================

CONSISTENCY_RUNS = 3

CONSISTENCY_THRESHOLD = 0.90

# ==========================================================
# Logging
# ==========================================================

DEBUG = True

LOG_DIR = "logs"

# ==========================================================
# Input / Output
# ==========================================================

INPUT_FILE = "input/tasks.json"

OUTPUT_FILE = "output/results.json"
