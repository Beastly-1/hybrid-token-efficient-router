import os

# ==========================================================
# Local Model (Ollama)
# ==========================================================

LOCAL_MODEL = "qwen2.5:1.5b"

LOCAL_ENDPOINT = "http://localhost:11434/api/generate"

# ==========================================================
# Fireworks API
# (Read from environment variables)
# ==========================================================

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")

FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "")

ALLOWED_MODELS = [
    model.strip()
    for model in os.getenv("ALLOWED_MODELS", "").split(",")
    if model.strip()
]

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