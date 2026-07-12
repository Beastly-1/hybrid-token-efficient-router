import json
import re

from config import LOCAL_MODEL_ENABLED
from models.local_model import LocalModel
from state import RoutingState


class LocalTaskClassifier:
    """Uses the local AI model to refine task classification."""

    KNOWN_TASK_TYPES = {
        "factual",
        "summarization",
        "sentiment",
        "ner",
        "code_generation",
        "code_debug",
        "logic",
        "math",
        "general",
    }
    KNOWN_TOOL_CANDIDATES = {
        "calculator",
        "datetime",
        "json_validator",
        "regex_verifier",
        "python_executor",
        "verification",
        None,
    }
    KNOWN_DIFFICULTIES = {"easy", "medium", "hard"}

    def __init__(self, local_model=None):
        self.local_model = local_model or LocalModel()

    def classify(self, state: RoutingState):
        if not LOCAL_MODEL_ENABLED:
            return state

        # Only refine tasks where the zero-token analyzer is uncertain.
        if state.task_type != "factual" or state.tool_candidate is not None:
            return state

        prompt = self._build_prompt(state.query)
        classification_state = RoutingState(query=state.query)
        classification_state.system_prompt = prompt
        classification_state.max_new_tokens = 64

        try:
            classification_state = self.local_model.generate(
                classification_state
            )
        except Exception:
            return state

        classification = self._parse_classification(
            classification_state.local_answer
        )
        if not classification:
            return state

        if (
            state.task_type == "factual"
            and classification["task_type"] in self.KNOWN_TASK_TYPES
        ):
            state.task_type = classification["task_type"]

        if state.tool_candidate is None and classification["tool_candidate"] in self.KNOWN_TOOL_CANDIDATES:
            state.tool_candidate = classification["tool_candidate"]

        if classification["difficulty"] in self.KNOWN_DIFFICULTIES:
            state.difficulty = classification["difficulty"]

        return state

    def _build_prompt(self, query: str) -> str:
        return (
            "You are a task classification assistant. Classify the user query into a task type, "
            "candidate tool, and difficulty. Output only a JSON object with keys: task_type, "
            "tool_candidate, difficulty. Use one of the allowed values exactly. Do not add any "
            "explanation.\n\n"
            "Allowed task_type values: factual, summarization, sentiment, ner, code_generation, "
            "code_debug, logic, math, general.\n"
            "Allowed tool_candidate values: calculator, datetime, json_validator, regex_verifier, "
            "python_executor, verification, null.\n"
            "Allowed difficulty values: easy, medium, hard.\n\n"
            f"Query: \"{query}\""
        )

    def _parse_classification(self, text: str) -> dict[str, str | None] | None:
        if not isinstance(text, str) or not text.strip():
            return None

        text = text.strip()
        try:
            body = self._extract_json_payload(text)
            parsed = json.loads(body)
            return {
                "task_type": parsed.get("task_type"),
                "tool_candidate": parsed.get("tool_candidate"),
                "difficulty": parsed.get("difficulty"),
            }
        except Exception:
            return self._parse_key_value(text)

    def _extract_json_payload(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON payload found")
        return text[start : end + 1]

    def _parse_key_value(self, text: str) -> dict[str, str | None] | None:
        result = {"task_type": None, "tool_candidate": None, "difficulty": None}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(task_type|tool_candidate|difficulty)\s*[:=]\s*(.+)$", line, re.I)
            if not match:
                continue
            key = match.group(1).lower()
            value = match.group(2).strip().strip('"').strip("'")
            if value.lower() == "null":
                value = None
            result[key] = value

        if result["task_type"] is None and result["difficulty"] is None and result["tool_candidate"] is None:
            return None
        return result
