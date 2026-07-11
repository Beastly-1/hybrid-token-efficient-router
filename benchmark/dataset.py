"""Benchmark dataset schema, loader and validator."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


VALID_CATEGORIES = {
    "factual", "math", "sentiment", "summarization",
    "ner", "code_debug", "logic", "code_generation",
}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}

VALID_SCORING_METHODS = {
    "exact", "normalized_exact", "acceptable_answers",
    "numeric", "sentiment", "summarization", "ner",
    "code_tests", "logic",
}

VALID_SENTIMENT_LABELS = {"positive", "negative", "neutral"}


@dataclass
class BenchmarkCase:
    task_id: str
    category: str
    difficulty: str
    prompt: str
    scoring_method: str
    reference: Any = None
    acceptable_answers: Optional[list] = None
    tolerance: Optional[float] = None
    constraints: Optional[dict] = None
    tests: Optional[list] = None
    metadata: Optional[dict] = None


class DatasetValidationError(Exception):
    """Raised when a benchmark dataset fails validation."""


def _is_finite_number(value) -> bool:
    """Check if value is a finite number (not bool)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    if math.isnan(value) or math.isinf(value):
        return False
    return True


def _numeric_from_value(value) -> Optional[float]:
    """Parse a numeric value from a number or numeric string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, str):
        try:
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except (ValueError, TypeError):
            return None
    return None


def validate_case(data: dict, line_number: int, source_path: str) -> BenchmarkCase:
    """Validate a single benchmark case dict and return a BenchmarkCase."""
    loc = f"{source_path}:{line_number}"
    errors: list[str] = []

    if not isinstance(data, dict):
        raise DatasetValidationError(f"{loc}: record must be a JSON object")

    # Required string fields
    task_id = data.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        errors.append(f"{loc}: task_id must be a non-empty string")
        task_id = ""

    category = data.get("category")
    if category not in VALID_CATEGORIES:
        errors.append(f"{loc}: category '{category}' not recognised; expected one of {sorted(VALID_CATEGORIES)}")

    difficulty = data.get("difficulty")
    if difficulty not in VALID_DIFFICULTIES:
        errors.append(f"{loc}: difficulty '{difficulty}' not recognised; expected one of {sorted(VALID_DIFFICULTIES)}")

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append(f"{loc}: prompt must be a non-empty string")

    scoring_method = data.get("scoring_method")
    if scoring_method not in VALID_SCORING_METHODS:
        errors.append(f"{loc}: scoring_method '{scoring_method}' not recognised; expected one of {sorted(VALID_SCORING_METHODS)}")

    # Optional typed fields
    metadata = data.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        errors.append(f"{loc}: metadata must be an object when present")

    constraints = data.get("constraints")
    if constraints is not None and not isinstance(constraints, dict):
        errors.append(f"{loc}: constraints must be an object when present")

    tests = data.get("tests")
    if tests is not None and not isinstance(tests, list):
        errors.append(f"{loc}: tests must be a list when present")

    tolerance = data.get("tolerance")
    if tolerance is not None:
        if isinstance(tolerance, bool):
            errors.append(f"{loc}: tolerance must not be a boolean")
        elif not _is_finite_number(tolerance):
            errors.append(f"{loc}: tolerance must be a finite number")
        elif tolerance < 0:
            errors.append(f"{loc}: tolerance must be non-negative")

    # Scoring-method-specific validation
    if scoring_method in VALID_SCORING_METHODS and not errors:
        errors.extend(_validate_scoring(data, scoring_method, loc))

    if errors:
        raise DatasetValidationError("\n".join(errors))

    return BenchmarkCase(
        task_id=data["task_id"],
        category=data["category"],
        difficulty=data["difficulty"],
        prompt=data["prompt"],
        scoring_method=data["scoring_method"],
        reference=data.get("reference"),
        acceptable_answers=data.get("acceptable_answers"),
        tolerance=float(tolerance) if tolerance is not None and _is_finite_number(tolerance) else None,
        constraints=constraints if isinstance(constraints, dict) else None,
        tests=tests if isinstance(tests, list) else None,
        metadata=metadata if isinstance(metadata, dict) else None,
    )


def _validate_scoring(data: dict, method: str, loc: str) -> list[str]:
    """Validate scoring-method-specific field requirements."""
    errors: list[str] = []

    if method in ("exact", "normalized_exact"):
        if "reference" not in data or data["reference"] is None:
            errors.append(f"{loc}: scoring_method '{method}' requires reference")

    elif method == "numeric":
        ref = data.get("reference")
        if ref is None:
            errors.append(f"{loc}: scoring_method 'numeric' requires reference")
        elif _numeric_from_value(ref) is None:
            errors.append(f"{loc}: scoring_method 'numeric' requires a numeric reference")

    elif method == "acceptable_answers":
        aa = data.get("acceptable_answers")
        if not isinstance(aa, list) or len(aa) == 0:
            errors.append(f"{loc}: scoring_method 'acceptable_answers' requires a non-empty acceptable_answers list")
        elif any(a is None for a in aa):
            errors.append(f"{loc}: acceptable_answers must not contain null values")

    elif method == "sentiment":
        ref = data.get("reference")
        aa = data.get("acceptable_answers")
        if ref is None and (not isinstance(aa, list) or len(aa) == 0):
            errors.append(f"{loc}: scoring_method 'sentiment' requires reference or acceptable_answers")
        # Validate labels
        labels_to_check = []
        if isinstance(ref, str):
            labels_to_check.append(ref)
        if isinstance(aa, list):
            labels_to_check.extend(a for a in aa if isinstance(a, str))
        for label in labels_to_check:
            if label.lower() not in VALID_SENTIMENT_LABELS:
                errors.append(f"{loc}: sentiment label '{label}' not recognised; expected one of {sorted(VALID_SENTIMENT_LABELS)}")

    elif method == "summarization":
        ref = data.get("reference")
        constraints = data.get("constraints")
        if ref is None and (not isinstance(constraints, dict) or len(constraints) == 0):
            errors.append(f"{loc}: scoring_method 'summarization' requires reference and/or constraints")
        if isinstance(constraints, dict):
            for key in ("max_words", "max_sentences", "max_characters"):
                if key in constraints:
                    v = constraints[key]
                    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                        errors.append(f"{loc}: constraint '{key}' must be a positive integer")

    elif method == "ner":
        ref = data.get("reference")
        if not isinstance(ref, list):
            errors.append(f"{loc}: scoring_method 'ner' requires reference as a list of entity objects")
        elif len(ref) == 0:
            errors.append(f"{loc}: scoring_method 'ner' requires a non-empty reference list")
        else:
            for i, entity in enumerate(ref):
                if not isinstance(entity, dict):
                    errors.append(f"{loc}: reference[{i}] must be an object")
                elif "text" not in entity or "type" not in entity:
                    errors.append(f"{loc}: reference[{i}] must contain 'text' and 'type'")

    elif method == "code_tests":
        tests_list = data.get("tests")
        if not isinstance(tests_list, list) or len(tests_list) == 0:
            errors.append(f"{loc}: scoring_method 'code_tests' requires a non-empty tests list")
        elif isinstance(tests_list, list):
            for i, t in enumerate(tests_list):
                if not isinstance(t, dict):
                    errors.append(f"{loc}: tests[{i}] must be an object")
                elif "call" not in t or "expected" not in t:
                    errors.append(f"{loc}: tests[{i}] must contain 'call' and 'expected'")

    elif method == "logic":
        ref = data.get("reference")
        aa = data.get("acceptable_answers")
        if ref is None and (not isinstance(aa, list) or len(aa) == 0):
            errors.append(f"{loc}: scoring_method 'logic' requires reference or acceptable_answers")

    return errors


def load_dataset(path: Path) -> list[BenchmarkCase]:
    """Load and validate a JSONL benchmark dataset. Raises on any error."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    source = str(path)

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise DatasetValidationError(
                    f"{source}:{line_number}: malformed JSON: {e}"
                )

            if not isinstance(data, dict):
                raise DatasetValidationError(
                    f"{source}:{line_number}: record must be a JSON object, got {type(data).__name__}"
                )

            case = validate_case(data, line_number, source)

            if case.task_id in seen_ids:
                raise DatasetValidationError(
                    f"{source}:{line_number}: duplicate task_id '{case.task_id}'"
                )
            seen_ids.add(case.task_id)
            cases.append(case)

    return cases


def validate_dataset(path: Path) -> list[BenchmarkCase]:
    """Alias for load_dataset; validates and returns cases or raises."""
    return load_dataset(path)


def dataset_summary(cases: list[BenchmarkCase]) -> dict:
    """Produce a summary of a loaded dataset."""
    by_category: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    by_scoring: dict[str, int] = {}

    for c in cases:
        by_category[c.category] = by_category.get(c.category, 0) + 1
        by_difficulty[c.difficulty] = by_difficulty.get(c.difficulty, 0) + 1
        by_scoring[c.scoring_method] = by_scoring.get(c.scoring_method, 0) + 1

    # Check for duplicates (should be zero for valid datasets)
    ids = [c.task_id for c in cases]
    duplicate_count = len(ids) - len(set(ids))

    return {
        "total_cases": len(cases),
        "by_category": dict(sorted(by_category.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "by_scoring_method": dict(sorted(by_scoring.items())),
        "duplicate_count": duplicate_count,
    }
