"""Deterministic answer scorers for benchmark evaluation.

Supported scoring methods (this module):
- exact
- normalized_exact
- acceptable_answers
- numeric
- sentiment
- logic
- ner
- summarization
- code_tests
"""

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from benchmark.dataset import BenchmarkCase
from benchmark.code_execution import execute_code_tests
from benchmark.code_validation import validate_code_answer


# ===================================================================
# Score Result
# ===================================================================


@dataclass
class ScoreResult:
    """Structured result from scoring a candidate answer against a benchmark case.

    Status values:
    - "scored": scoring completed, score and correct are populated
    - "invalid_answer": candidate answer could not be interpreted
    - "unsupported": scoring method is not yet implemented
    """

    scoring_method: str
    score: Optional[float]
    correct: Optional[bool]
    status: str
    details: dict = field(default_factory=dict)


# ===================================================================
# Text Normalization
# ===================================================================


def normalize_text(value: Any) -> Optional[str]:
    """Conservative text normalization for deterministic comparison.

    Steps:
    1. Convert supported scalar values to string
    2. Apply Unicode NFKC normalization
    3. Strip leading/trailing whitespace
    4. Collapse repeated internal whitespace
    5. Apply casefold()

    Returns None if value cannot be converted to text.
    Does not remove punctuation, words, or perform stemming.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        text = str(value)
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        text = str(value)
    elif isinstance(value, str):
        text = value
    elif isinstance(value, (list, dict)):
        return None
    else:
        text = str(value)

    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.casefold()
    return text


# ===================================================================
# Dispatcher
# ===================================================================


def score_answer(case: BenchmarkCase, answer: Any) -> ScoreResult:
    """Score a candidate answer against a benchmark case.

    Dispatches to the appropriate scorer based on case.scoring_method.
    """
    method = case.scoring_method

    if method == "exact":
        return _score_exact(case, answer)
    elif method == "normalized_exact":
        return _score_normalized_exact(case, answer)
    elif method == "acceptable_answers":
        return _score_acceptable_answers(case, answer)
    elif method == "numeric":
        return _score_numeric(case, answer)
    elif method == "sentiment":
        return _score_sentiment(case, answer)
    elif method == "logic":
        return _score_logic(case, answer)
    elif method == "ner":
        return _score_ner(case, answer)
    elif method == "summarization":
        return _score_summarization(case, answer)
    elif method == "code_tests":
        return _score_code_tests(case, answer)
    else:
        return ScoreResult(
            scoring_method=method,
            score=None,
            correct=None,
            status="unsupported",
            details={"reason": f"unknown scoring method '{method}'"},
        )


# ===================================================================
# Exact Scorer
# ===================================================================


def _score_exact(case: BenchmarkCase, answer: Any) -> ScoreResult:
    """Exact match: compare candidate with reference after stripping whitespace.

    Comparison behaviour:
    - Leading/trailing whitespace is stripped from both values
    - Case is preserved (case-sensitive comparison)
    - Internal punctuation and whitespace are preserved
    """
    if answer is None:
        return ScoreResult("exact", None, None, "invalid_answer",
                           {"reason": "answer is None"})

    if not isinstance(answer, str):
        answer = str(answer)

    ref = case.reference
    if ref is None:
        return ScoreResult("exact", None, None, "invalid_answer",
                           {"reason": "case has no reference"})
    if not isinstance(ref, str):
        ref = str(ref)

    candidate = answer.strip()
    expected = ref.strip()
    matched = candidate == expected
    score = 1.0 if matched else 0.0

    return ScoreResult("exact", score, matched, "scored",
                       {"candidate": candidate, "expected": expected})


# ===================================================================
# Normalized Exact Scorer
# ===================================================================


def _score_normalized_exact(case: BenchmarkCase, answer: Any) -> ScoreResult:
    """Normalized exact match using shared text normalization."""
    if answer is None:
        return ScoreResult("normalized_exact", None, None, "invalid_answer",
                           {"reason": "answer is None"})

    ref = case.reference
    if ref is None:
        return ScoreResult("normalized_exact", None, None, "invalid_answer",
                           {"reason": "case has no reference"})

    norm_answer = normalize_text(answer)
    norm_ref = normalize_text(ref)

    if norm_answer is None:
        return ScoreResult("normalized_exact", None, None, "invalid_answer",
                           {"reason": "answer could not be normalized"})
    if norm_ref is None:
        return ScoreResult("normalized_exact", None, None, "invalid_answer",
                           {"reason": "reference could not be normalized"})

    matched = norm_answer == norm_ref
    score = 1.0 if matched else 0.0

    return ScoreResult("normalized_exact", score, matched, "scored",
                       {"normalized_candidate": norm_answer,
                        "normalized_reference": norm_ref})


# ===================================================================
# Acceptable Answers Scorer
# ===================================================================


def _score_acceptable_answers(case: BenchmarkCase, answer: Any) -> ScoreResult:
    """Match candidate against any acceptable answer using normalization."""
    if answer is None:
        return ScoreResult("acceptable_answers", None, None, "invalid_answer",
                           {"reason": "answer is None"})

    aa = case.acceptable_answers
    if not isinstance(aa, list) or len(aa) == 0:
        return ScoreResult("acceptable_answers", None, None, "invalid_answer",
                           {"reason": "no acceptable_answers defined"})

    norm_answer = normalize_text(answer)
    if norm_answer is None:
        return ScoreResult("acceptable_answers", None, None, "invalid_answer",
                           {"reason": "answer could not be normalized"})

    matched = False
    for accepted in aa:
        if accepted is None:
            continue
        norm_accepted = normalize_text(accepted)
        if norm_accepted is not None and norm_answer == norm_accepted:
            matched = True
            break

    score = 1.0 if matched else 0.0
    return ScoreResult("acceptable_answers", score, matched, "scored",
                       {"normalized_candidate": norm_answer,
                        "acceptable_count": len(aa)})


# ===================================================================
# Numeric Scorer
# ===================================================================


_NUMERIC_PATTERN = re.compile(
    r"[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?%?"
    r"|[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?%?"
)


def _parse_candidate_numeric(answer: Any) -> Optional[Decimal]:
    """Parse a single numeric value from a candidate answer.

    Supports: integers, decimals, negatives, scientific notation,
    comma-separated numbers, trailing %, surrounding text (if exactly one
    numeric value is present).

    Returns None if zero or multiple numeric values are found.
    """
    if answer is None:
        return None
    if isinstance(answer, bool):
        return None
    if isinstance(answer, (int, float)):
        if isinstance(answer, float) and (math.isnan(answer) or math.isinf(answer)):
            return None
        try:
            return Decimal(str(answer))
        except InvalidOperation:
            return None

    text = str(answer).strip()
    matches = _NUMERIC_PATTERN.findall(text)
    if len(matches) != 1:
        return None

    raw = matches[0]
    # Remove trailing percent (treat as display notation, not division by 100)
    if raw.endswith("%"):
        raw = raw[:-1]
    # Remove commas
    raw = raw.replace(",", "")

    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _parse_reference_numeric(ref: Any) -> Optional[Decimal]:
    """Parse the reference value as a Decimal."""
    if ref is None:
        return None
    if isinstance(ref, bool):
        return None
    if isinstance(ref, (int, float)):
        if isinstance(ref, float) and (math.isnan(ref) or math.isinf(ref)):
            return None
        try:
            return Decimal(str(ref))
        except InvalidOperation:
            return None
    if isinstance(ref, str):
        cleaned = ref.strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None
    return None


def _score_numeric(case: BenchmarkCase, answer: Any) -> ScoreResult:
    """Numeric comparison with optional absolute tolerance."""
    if answer is None:
        return ScoreResult("numeric", None, None, "invalid_answer",
                           {"reason": "answer is None"})

    # Validate tolerance
    tolerance = case.tolerance
    if tolerance is not None:
        if isinstance(tolerance, bool):
            return ScoreResult("numeric", None, None, "invalid_answer",
                               {"reason": "boolean tolerance rejected"})
        if not isinstance(tolerance, (int, float)):
            return ScoreResult("numeric", None, None, "invalid_answer",
                               {"reason": "invalid tolerance type"})
        if math.isnan(tolerance) or math.isinf(tolerance):
            return ScoreResult("numeric", None, None, "invalid_answer",
                               {"reason": "NaN or infinite tolerance rejected"})
        if tolerance < 0:
            return ScoreResult("numeric", None, None, "invalid_answer",
                               {"reason": "negative tolerance rejected"})

    tol = Decimal(str(tolerance)) if tolerance is not None else Decimal("0")

    ref_decimal = _parse_reference_numeric(case.reference)
    if ref_decimal is None:
        return ScoreResult("numeric", None, None, "invalid_answer",
                           {"reason": "reference is not numeric"})

    candidate_decimal = _parse_candidate_numeric(answer)
    if candidate_decimal is None:
        return ScoreResult("numeric", 0.0, False, "scored",
                           {"reason": "could not parse numeric value from answer",
                            "reference": str(ref_decimal),
                            "tolerance": str(tol)})

    diff = abs(candidate_decimal - ref_decimal)
    matched = diff <= tol
    score = 1.0 if matched else 0.0

    return ScoreResult("numeric", score, matched, "scored",
                       {"parsed_candidate": str(candidate_decimal),
                        "reference": str(ref_decimal),
                        "tolerance": str(tol),
                        "absolute_difference": str(diff)})


# ===================================================================
# Sentiment Scorer
# ===================================================================


_SENTIMENT_LABELS = {"positive", "negative", "neutral"}

# Word-boundary pattern for sentiment labels
_SENTIMENT_PATTERN = re.compile(
    r"\b(positive|negative|neutral)\b", re.IGNORECASE
)


def _score_sentiment(case: BenchmarkCase, answer: Any) -> ScoreResult:
    """Extract sentiment label from candidate and compare with expected.

    Rules:
    - Exactly one unique supported label must appear (word-boundary match)
    - Multiple conflicting labels -> invalid
    - No label found -> score 0.0
    """
    if answer is None:
        return ScoreResult("sentiment", None, None, "invalid_answer",
                           {"reason": "answer is None"})

    if not isinstance(answer, str):
        answer = str(answer)

    # Find all sentiment labels in the answer
    found = _SENTIMENT_PATTERN.findall(answer)
    unique_labels = set(label.lower() for label in found)

    if len(unique_labels) > 1:
        return ScoreResult("sentiment", 0.0, False, "scored",
                           {"reason": "multiple conflicting labels found",
                            "labels_found": sorted(unique_labels)})

    if len(unique_labels) == 0:
        return ScoreResult("sentiment", 0.0, False, "scored",
                           {"reason": "no sentiment label found"})

    extracted = unique_labels.pop()

    # Build expected labels set
    expected_labels: set[str] = set()
    if case.reference and isinstance(case.reference, str):
        expected_labels.add(case.reference.lower())
    if isinstance(case.acceptable_answers, list):
        for a in case.acceptable_answers:
            if isinstance(a, str):
                expected_labels.add(a.lower())

    if not expected_labels:
        return ScoreResult("sentiment", None, None, "invalid_answer",
                           {"reason": "no expected labels in case"})

    matched = extracted in expected_labels
    score = 1.0 if matched else 0.0

    return ScoreResult("sentiment", score, matched, "scored",
                       {"extracted_label": extracted,
                        "expected_labels": sorted(expected_labels)})


# ===================================================================
# Logic Scorer
# ===================================================================


def _score_logic(case: BenchmarkCase, answer: Any) -> ScoreResult:
    """Logic scorer: normalized comparison against reference or acceptable answers."""
    if answer is None:
        return ScoreResult("logic", None, None, "invalid_answer",
                           {"reason": "answer is None"})

    norm_answer = normalize_text(answer)
    if norm_answer is None:
        return ScoreResult("logic", None, None, "invalid_answer",
                           {"reason": "answer could not be normalized"})

    # Check acceptable_answers first (more permissive)
    if isinstance(case.acceptable_answers, list) and len(case.acceptable_answers) > 0:
        for accepted in case.acceptable_answers:
            if accepted is None:
                continue
            norm_accepted = normalize_text(accepted)
            if norm_accepted is not None and norm_answer == norm_accepted:
                return ScoreResult("logic", 1.0, True, "scored",
                                   {"normalized_candidate": norm_answer,
                                    "matched_answer": str(accepted)})

    # Fall back to reference
    if case.reference is not None:
        norm_ref = normalize_text(case.reference)
        if norm_ref is not None and norm_answer == norm_ref:
            return ScoreResult("logic", 1.0, True, "scored",
                               {"normalized_candidate": norm_answer,
                                "matched_reference": str(case.reference)})

    return ScoreResult("logic", 0.0, False, "scored",
                       {"normalized_candidate": norm_answer,
                        "reason": "no match found"})


# ===================================================================
# NER Scorer
# ===================================================================


_TYPE_ALIASES: dict[str, str] = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "ORGANIZATION": "ORGANIZATION",
    "ORGANISATION": "ORGANIZATION",
    "ORG": "ORGANIZATION",
    "LOCATION": "LOCATION",
    "LOC": "LOCATION",
    "PLACE": "LOCATION",
    "DATE": "DATE",
}


def _normalize_entity_type(raw: str) -> str:
    """Normalize entity type: strip, uppercase, apply aliases."""
    upper = raw.strip().upper()
    return _TYPE_ALIASES.get(upper, upper)


def _normalize_entity_text(raw: str) -> str:
    """Normalize entity text using the same rules as normalize_text."""
    text = unicodedata.normalize("NFKC", raw)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.casefold()
    return text


def _validate_entity(item: Any) -> Optional[tuple[str, str]]:
    """Validate and normalize a single entity dict.

    Returns (normalized_text, normalized_type) or None if invalid.
    """
    if not isinstance(item, dict):
        return None
    text = item.get("text")
    etype = item.get("type")
    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(etype, str) or not etype.strip():
        return None
    return (_normalize_entity_text(text), _normalize_entity_type(etype))


def _parse_ner_candidate(answer: Any) -> Optional[list[dict]]:
    """Parse candidate answer into a list of entity dicts.

    Supported formats:
    1. Python list of dicts
    2. Dict with 'entities' key
    3. JSON string (array or object with 'entities')
    4. JSON inside ```json markdown fence
    5. Pipe-separated lines: text | type
    6. Colon-separated lines: text: type (unambiguous single colon)

    Returns None if the answer cannot be parsed into a structured entity list.
    """
    if answer is None:
        return None

    # Already a list
    if isinstance(answer, list):
        return answer

    # Dict with entities key
    if isinstance(answer, dict):
        entities = answer.get("entities")
        if isinstance(entities, list):
            return entities
        return None

    if not isinstance(answer, str):
        return None

    text = answer.strip()
    if not text:
        return None

    # Try to extract JSON from markdown fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            entities = parsed.get("entities")
            if isinstance(entities, list):
                return entities
            return None
        return None
    except (json.JSONDecodeError, ValueError):
        pass

    # Try line-based formats (pipe or colon separated)
    lines = text.splitlines()
    non_empty_lines = [l.strip() for l in lines if l.strip()]
    if not non_empty_lines:
        return None

    # Check if this looks like structured lines (all lines must match a pattern)
    entities: list[dict] = []

    # Try pipe-separated first
    pipe_ok = True
    for line in non_empty_lines:
        if "|" in line:
            parts = line.split("|", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                entities.append({"text": parts[0].strip(), "type": parts[1].strip()})
            else:
                pipe_ok = False
                break
        else:
            pipe_ok = False
            break

    if pipe_ok and entities:
        return entities

    # Try colon-separated (only if every line has exactly one colon
    # and both sides are non-empty, to avoid ambiguity)
    entities = []
    colon_ok = True
    for line in non_empty_lines:
        # Must have exactly one colon to be unambiguous
        if line.count(":") == 1:
            parts = line.split(":", 1)
            if parts[0].strip() and parts[1].strip():
                entities.append({"text": parts[0].strip(), "type": parts[1].strip()})
            else:
                colon_ok = False
                break
        else:
            colon_ok = False
            break

    if colon_ok and entities:
        return entities

    return None


def _score_ner(case: BenchmarkCase, answer: Any) -> ScoreResult:
    """NER scorer: precision, recall and F1 over normalized entity pairs.

    Matching uses exact (normalized_text, normalized_type) pair comparison.
    Duplicates in predictions or reference are deduplicated.
    """
    if answer is None:
        return ScoreResult("ner", None, None, "invalid_answer",
                           {"reason": "answer is None"})

    # Parse candidate
    raw_entities = _parse_ner_candidate(answer)
    if raw_entities is None:
        return ScoreResult("ner", None, None, "invalid_answer",
                           {"reason": "could not parse entities from answer"})

    # Validate all predicted entities (reject if any are malformed)
    predicted_pairs: list[tuple[str, str]] = []
    for i, item in enumerate(raw_entities):
        pair = _validate_entity(item)
        if pair is None:
            return ScoreResult("ner", None, None, "invalid_answer",
                               {"reason": f"malformed entity at index {i}"})
        predicted_pairs.append(pair)

    # Deduplicate predictions
    predicted_set = set(predicted_pairs)

    # Parse and validate reference
    ref = case.reference
    if not isinstance(ref, list):
        return ScoreResult("ner", None, None, "invalid_answer",
                           {"reason": "case reference is not a list"})

    reference_pairs: list[tuple[str, str]] = []
    for item in ref:
        pair = _validate_entity(item)
        if pair is None:
            return ScoreResult("ner", None, None, "invalid_answer",
                               {"reason": "malformed entity in reference"})
        reference_pairs.append(pair)

    # Deduplicate reference
    reference_set = set(reference_pairs)

    # Calculate metrics
    tp = len(predicted_set & reference_set)
    fp = len(predicted_set - reference_set)
    fn = len(reference_set - predicted_set)

    # Handle edge cases
    if len(predicted_set) == 0 and len(reference_set) == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    elif len(predicted_set) == 0 or len(reference_set) == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

    correct = f1 == 1.0
    return ScoreResult("ner", f1, correct, "scored", {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "predicted_count": len(predicted_set),
        "reference_count": len(reference_set),
    })


# ===================================================================
# Summarization Scorer
# ===================================================================

# Safety limit: reject inputs exceeding this token count to avoid
# expensive LCS computation on very large strings.
_MAX_SUMMARY_TOKENS = 2000

# Default minimum content score threshold for correctness.
_DEFAULT_MIN_CONTENT_SCORE = 0.60

# Recognised constraint keys.
_RECOGNISED_CONSTRAINTS = {"max_words", "max_sentences", "max_characters"}

# Word-token pattern: sequences of alphanumeric/underscore characters.
_WORD_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

# Sentence boundary: one or more terminal punctuation marks.
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"[.!?]+")


def _tokenize_for_content(text: str) -> list[str]:
    """Extract word tokens for content comparison.

    Steps:
    1. Unicode NFKC normalization
    2. casefold()
    3. Extract \\w+ tokens (alphanumeric + underscore sequences)

    Surrounding punctuation is ignored. Word order is preserved.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()
    return _WORD_TOKEN_PATTERN.findall(normalized)


def _lcs_length(seq_a: list[str], seq_b: list[str]) -> int:
    """Compute length of the longest common subsequence.

    Uses O(min(m,n)) space with a single-row DP approach.
    """
    if not seq_a or not seq_b:
        return 0
    # Ensure seq_b is the shorter sequence for space efficiency
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    m, n = len(seq_a), len(seq_b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _rouge_l_f1(candidate_tokens: list[str], reference_tokens: list[str]) -> tuple[float, float, float]:
    """Calculate ROUGE-L precision, recall and F1.

    Returns (precision, recall, f1).
    """
    cand_len = len(candidate_tokens)
    ref_len = len(reference_tokens)

    if cand_len == 0 and ref_len == 0:
        return 1.0, 1.0, 1.0
    if cand_len == 0 or ref_len == 0:
        return 0.0, 0.0, 0.0

    lcs = _lcs_length(candidate_tokens, reference_tokens)
    precision = lcs / cand_len
    recall = lcs / ref_len
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _count_sentences(text: str) -> int:
    """Count sentences using a simple deterministic heuristic.

    Rules:
    - Terminal '.', '!' or '?' sequences are sentence boundaries.
    - When no terminal punctuation is present, non-empty newline-separated
      segments are counted.
    - A non-empty text with no terminal punctuation and no newlines counts
      as one sentence.
    - Empty segments are ignored.

    Limitations: abbreviations (e.g. 'Dr.') may be miscounted. This is
    acceptable for a deterministic scorer without NLP dependencies.
    """
    stripped = text.strip()
    if not stripped:
        return 0

    # Check if terminal punctuation exists
    if _SENTENCE_BOUNDARY_PATTERN.search(stripped):
        # Split on sentence boundaries and count non-empty segments before each
        parts = _SENTENCE_BOUNDARY_PATTERN.split(stripped)
        count = sum(1 for p in parts if p.strip())
        return max(count, 1)

    # No terminal punctuation: use newline-separated segments
    lines = stripped.splitlines()
    non_empty = [l for l in lines if l.strip()]
    return max(len(non_empty), 1)


def _validate_constraint_value(value: Any) -> Optional[int]:
    """Validate a constraint value. Returns int or None if invalid."""
    if isinstance(value, bool):
        return None
    if not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value


def _extract_summary_text(answer: Any) -> Optional[str]:
    """Extract summary text from a candidate answer.

    Supported formats:
    1. Non-empty string
    2. Dict with 'summary' key (preferred) or 'answer' key

    Returns None for invalid inputs.
    """
    if answer is None:
        return None
    if isinstance(answer, bool):
        return None
    if isinstance(answer, (int, float)):
        return None
    if isinstance(answer, list):
        return None

    if isinstance(answer, dict):
        # Prefer 'summary' over 'answer'
        text = answer.get("summary")
        if isinstance(text, str) and text.strip():
            return text
        text = answer.get("answer")
        if isinstance(text, str) and text.strip():
            return text
        return None

    if isinstance(answer, str):
        if answer.strip():
            return answer
        return None

    return None


def _score_summarization(case: BenchmarkCase, answer: Any) -> ScoreResult:
    """Summarization scorer: ROUGE-L F1 content overlap + constraint compliance.

    Final score formula:
    - reference + constraints: rouge_l_f1 * constraint_score
    - reference only: rouge_l_f1
    - constraints only: constraint_score
    - neither: invalid_answer
    """
    # Extract candidate text
    candidate_text = _extract_summary_text(answer)
    if candidate_text is None:
        return ScoreResult("summarization", None, None, "invalid_answer",
                           {"reason": "could not extract summary text from answer"})

    # Determine what we have to score against
    has_reference = case.reference is not None and isinstance(case.reference, str) and case.reference.strip()
    constraints_dict = case.constraints if isinstance(case.constraints, dict) else {}
    recognised = {k: v for k, v in constraints_dict.items() if k in _RECOGNISED_CONSTRAINTS}
    unknown_keys = [k for k in constraints_dict if k not in _RECOGNISED_CONSTRAINTS]
    has_constraints = len(recognised) > 0

    if not has_reference and not has_constraints:
        return ScoreResult("summarization", None, None, "invalid_answer",
                           {"reason": "case has neither reference nor recognised constraints"})

    # Validate constraint values
    for key, val in recognised.items():
        valid_val = _validate_constraint_value(val)
        if valid_val is None:
            return ScoreResult("summarization", None, None, "invalid_answer",
                               {"reason": f"constraint '{key}' has invalid value: {val!r}"})

    # Validate minimum_content_score from metadata
    min_content_score = _DEFAULT_MIN_CONTENT_SCORE
    if isinstance(case.metadata, dict) and "minimum_content_score" in case.metadata:
        mcs = case.metadata["minimum_content_score"]
        if isinstance(mcs, bool):
            return ScoreResult("summarization", None, None, "invalid_answer",
                               {"reason": "metadata minimum_content_score must not be boolean"})
        if not isinstance(mcs, (int, float)):
            return ScoreResult("summarization", None, None, "invalid_answer",
                               {"reason": "metadata minimum_content_score must be numeric"})
        if isinstance(mcs, float) and (math.isnan(mcs) or math.isinf(mcs)):
            return ScoreResult("summarization", None, None, "invalid_answer",
                               {"reason": "metadata minimum_content_score must be finite"})
        if not (0.0 <= mcs <= 1.0):
            return ScoreResult("summarization", None, None, "invalid_answer",
                               {"reason": "metadata minimum_content_score must be between 0.0 and 1.0"})
        min_content_score = float(mcs)

    # Tokenize candidate
    candidate_tokens = _tokenize_for_content(candidate_text)

    # Safety limit check on candidate
    if len(candidate_tokens) > _MAX_SUMMARY_TOKENS:
        return ScoreResult("summarization", None, None, "invalid_answer",
                           {"reason": f"candidate exceeds safety limit of {_MAX_SUMMARY_TOKENS} tokens"})

    # Content scoring
    rouge_precision = None
    rouge_recall = None
    rouge_f1 = None

    if has_reference:
        reference_tokens = _tokenize_for_content(case.reference)
        # Safety limit check on reference
        if len(reference_tokens) > _MAX_SUMMARY_TOKENS:
            return ScoreResult("summarization", None, None, "invalid_answer",
                               {"reason": f"reference exceeds safety limit of {_MAX_SUMMARY_TOKENS} tokens"})
        rouge_precision, rouge_recall, rouge_f1 = _rouge_l_f1(candidate_tokens, reference_tokens)

    # Constraint scoring
    word_count = len(candidate_tokens)
    stripped_text = candidate_text.strip()
    character_count = len(stripped_text)
    sentence_count = _count_sentences(stripped_text)

    constraint_details: dict[str, dict] = {}
    passed_count = 0
    total_recognised = len(recognised)

    if "max_words" in recognised:
        limit = recognised["max_words"]
        passed = word_count <= limit
        constraint_details["max_words"] = {"limit": limit, "actual": word_count, "passed": passed}
        if passed:
            passed_count += 1

    if "max_characters" in recognised:
        limit = recognised["max_characters"]
        passed = character_count <= limit
        constraint_details["max_characters"] = {"limit": limit, "actual": character_count, "passed": passed}
        if passed:
            passed_count += 1

    if "max_sentences" in recognised:
        limit = recognised["max_sentences"]
        passed = sentence_count <= limit
        constraint_details["max_sentences"] = {"limit": limit, "actual": sentence_count, "passed": passed}
        if passed:
            passed_count += 1

    constraint_score = passed_count / total_recognised if total_recognised > 0 else 1.0
    all_constraints_passed = (passed_count == total_recognised)

    # Final score
    if has_reference and has_constraints:
        final_score = rouge_f1 * constraint_score
    elif has_reference:
        final_score = rouge_f1
    else:
        final_score = constraint_score

    # Clamp to [0, 1] for safety
    final_score = max(0.0, min(1.0, final_score))

    # Correctness
    if has_reference:
        correct = (rouge_f1 >= min_content_score) and all_constraints_passed
    else:
        correct = all_constraints_passed

    # Build details
    details: dict[str, Any] = {}
    if has_reference:
        details["content_metric"] = "rouge_l_f1"
        details["rouge_l_precision"] = rouge_precision
        details["rouge_l_recall"] = rouge_recall
        details["rouge_l_f1"] = rouge_f1
    details["constraint_score"] = constraint_score
    details["all_constraints_passed"] = all_constraints_passed
    details["minimum_content_score"] = min_content_score
    details["word_count"] = word_count
    details["sentence_count"] = sentence_count
    details["character_count"] = character_count
    if constraint_details:
        details["constraints"] = constraint_details
    details["unknown_constraints"] = unknown_keys

    return ScoreResult("summarization", final_score, correct, "scored", details)


# ===================================================================
# Code Tests Scorer
# ===================================================================


def _score_code_tests(case: BenchmarkCase, answer: Any) -> ScoreResult:
    """Code tests scorer: execute candidate code and compare outputs.

    Flow:
    1. Static validation (reuses code_validation.py)
    2. Subprocess execution with timeout
    3. Per-test result comparison
    """
    # Static validation first
    validation = validate_code_answer(case, answer)
    if not validation.valid:
        return ScoreResult("code_tests", None, None, "invalid_answer", {
            "reason": validation.details.get("reason", validation.status),
            "validation_status": validation.status,
        })

    # Execute in subprocess
    result = execute_code_tests(case, answer)

    if result.status == "invalid_test":
        return ScoreResult("code_tests", None, None, "invalid_answer", {
            "reason": "invalid test definition",
            "execution_details": result.details,
        })

    if result.status == "infrastructure_error":
        return ScoreResult("code_tests", None, None, "invalid_answer", {
            "reason": "execution infrastructure error",
            "execution_details": result.details,
        })

    if result.status == "timeout":
        return ScoreResult("code_tests", 0.0, False, "scored", {
            "execution_status": "timeout",
            "passed_tests": 0,
            "failed_tests": result.total,
            "total_tests": result.total,
            "timed_out": True,
        })

    # completed or runtime_error
    total = result.total
    passed = result.passed
    score = passed / total if total > 0 else 0.0
    correct = passed == total

    return ScoreResult("code_tests", score, correct, "scored", {
        "execution_status": result.status,
        "passed_tests": passed,
        "failed_tests": result.failed,
        "total_tests": total,
        "timed_out": False,
        "test_results": result.details.get("test_results", []),
    })
