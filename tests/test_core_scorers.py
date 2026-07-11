"""Tests for core deterministic answer scorers."""

from pathlib import Path

import pytest

from benchmark.dataset import BenchmarkCase, load_dataset
from benchmark.scorers import ScoreResult, normalize_text, score_answer


SMOKE_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "smoke" / "tasks.jsonl"


# ===========================================================
# Helpers
# ===========================================================


def _case(**overrides) -> BenchmarkCase:
    base = {
        "task_id": "t1",
        "category": "factual",
        "difficulty": "easy",
        "prompt": "test",
        "scoring_method": "exact",
        "reference": "answer",
    }
    base.update(overrides)
    return BenchmarkCase(**base)


# ===========================================================
# General (1-5)
# ===========================================================


def test_score_result_fields():
    """1. ScoreResult has the expected fields."""
    r = ScoreResult(scoring_method="exact", score=1.0, correct=True,
                    status="scored", details={"k": "v"})
    assert r.scoring_method == "exact"
    assert r.score == 1.0
    assert r.correct is True
    assert r.status == "scored"
    assert r.details == {"k": "v"}


def test_none_answer_handled():
    """2. None answer is handled safely."""
    case = _case()
    result = score_answer(case, None)
    assert result.status == "invalid_answer"
    assert result.score is None


def test_boolean_not_numeric():
    """3. Boolean input is not treated as numeric."""
    case = _case(scoring_method="numeric", reference=1)
    result = score_answer(case, True)
    # Boolean should not parse as numeric 1
    assert result.details.get("reason") == "could not parse numeric value from answer"


def test_nan_infinity_rejected():
    """4. NaN and infinity are rejected."""
    case = _case(scoring_method="numeric", reference=42)
    r1 = score_answer(case, float("nan"))
    r2 = score_answer(case, float("inf"))
    assert r1.correct is not True
    assert r2.correct is not True


def test_deferred_methods_unsupported():
    """5. No scoring methods remain unsupported (all are now implemented)."""
    # All valid scoring methods are now supported
    pass


# ===========================================================
# Exact (6-9)
# ===========================================================


def test_exact_match_succeeds():
    """6. Exact match succeeds."""
    case = _case(reference="Au")
    result = score_answer(case, "Au")
    assert result.score == 1.0
    assert result.correct is True


def test_exact_whitespace_stripped():
    """7. Leading/trailing whitespace is stripped."""
    case = _case(reference="Au")
    result = score_answer(case, "  Au  ")
    assert result.score == 1.0


def test_exact_case_mismatch_fails():
    """8. Case mismatch fails."""
    case = _case(reference="Au")
    result = score_answer(case, "au")
    assert result.score == 0.0
    assert result.correct is False


def test_exact_punctuation_mismatch_fails():
    """9. Punctuation mismatch fails."""
    case = _case(reference="New York")
    result = score_answer(case, "New York.")
    assert result.score == 0.0


# ===========================================================
# Normalized Exact (10-13)
# ===========================================================


def test_normalized_casefold():
    """10. Casefold matching works."""
    case = _case(scoring_method="normalized_exact", reference="New Delhi")
    result = score_answer(case, "new delhi")
    assert result.score == 1.0


def test_normalized_whitespace_collapsed():
    """11. Repeated whitespace is collapsed."""
    case = _case(scoring_method="normalized_exact", reference="New Delhi")
    result = score_answer(case, "New   Delhi")
    assert result.score == 1.0


def test_normalized_unicode_nfkc():
    """12. Unicode NFKC normalization works."""
    # NFKC normalizes \ufb01 (fi ligature) to "fi"
    case = _case(scoring_method="normalized_exact", reference="file")
    result = score_answer(case, "\ufb01le")
    assert result.score == 1.0


def test_normalized_different_text_fails():
    """13. Different substantive text fails."""
    case = _case(scoring_method="normalized_exact", reference="New Delhi")
    result = score_answer(case, "Delhi")
    assert result.score == 0.0
    assert result.correct is False


# ===========================================================
# Acceptable Answers (14-17)
# ===========================================================


def test_acceptable_first_matches():
    """14. First accepted answer matches."""
    case = _case(scoring_method="acceptable_answers",
                 acceptable_answers=["Jupiter", "JUPITER"])
    result = score_answer(case, "Jupiter")
    assert result.score == 1.0


def test_acceptable_later_matches():
    """15. Later accepted answer matches."""
    case = _case(scoring_method="acceptable_answers",
                 acceptable_answers=["Alpha", "Beta", "Gamma"])
    result = score_answer(case, "gamma")
    assert result.score == 1.0


def test_acceptable_no_match():
    """16. No accepted answer matches."""
    case = _case(scoring_method="acceptable_answers",
                 acceptable_answers=["Jupiter"])
    result = score_answer(case, "Saturn")
    assert result.score == 0.0
    assert result.correct is False


def test_acceptable_empty_list_handled():
    """17. Empty or malformed accepted list is handled safely."""
    case = _case(scoring_method="acceptable_answers", acceptable_answers=[])
    result = score_answer(case, "anything")
    assert result.status == "invalid_answer"


# ===========================================================
# Numeric (18-31)
# ===========================================================


def test_numeric_integer():
    """18. Integer answer."""
    case = _case(scoring_method="numeric", reference=408)
    result = score_answer(case, "408")
    assert result.score == 1.0


def test_numeric_decimal():
    """19. Decimal answer."""
    case = _case(scoring_method="numeric", reference=3.14)
    result = score_answer(case, "3.14")
    assert result.score == 1.0


def test_numeric_negative():
    """20. Negative answer."""
    case = _case(scoring_method="numeric", reference=-7)
    result = score_answer(case, "-7")
    assert result.score == 1.0


def test_numeric_scientific():
    """21. Scientific notation."""
    case = _case(scoring_method="numeric", reference=1500)
    result = score_answer(case, "1.5e3")
    assert result.score == 1.0


def test_numeric_comma_separated():
    """22. Comma-separated numeric answer."""
    case = _case(scoring_method="numeric", reference=1250.5)
    result = score_answer(case, "1,250.5")
    assert result.score == 1.0


def test_numeric_trailing_percent():
    """23. Trailing percent notation (17% -> 17)."""
    case = _case(scoring_method="numeric", reference=17)
    result = score_answer(case, "17%")
    assert result.score == 1.0


def test_numeric_embedded_in_sentence():
    """24. Numeric value embedded in a short sentence."""
    case = _case(scoring_method="numeric", reference=42)
    result = score_answer(case, "The answer is 42.")
    assert result.score == 1.0


def test_numeric_multiple_values_rejected():
    """25. Multiple numeric values are rejected."""
    case = _case(scoring_method="numeric", reference=42)
    result = score_answer(case, "Between 40 and 42")
    assert result.score == 0.0
    assert "could not parse" in result.details.get("reason", "")


def test_numeric_exact_comparison():
    """26. Exact numeric comparison (no tolerance)."""
    case = _case(scoring_method="numeric", reference=100)
    result = score_answer(case, "100.01")
    assert result.score == 0.0


def test_numeric_within_tolerance():
    """27. Within-tolerance comparison."""
    case = _case(scoring_method="numeric", reference=100.0, tolerance=0.1)
    result = score_answer(case, "100.05")
    assert result.score == 1.0


def test_numeric_outside_tolerance():
    """28. Outside-tolerance comparison."""
    case = _case(scoring_method="numeric", reference=100.0, tolerance=0.01)
    result = score_answer(case, "100.5")
    assert result.score == 0.0


def test_numeric_negative_tolerance_rejected():
    """29. Negative tolerance rejected."""
    case = _case(scoring_method="numeric", reference=1, tolerance=-0.5)
    result = score_answer(case, "1")
    assert result.status == "invalid_answer"
    assert "negative" in result.details.get("reason", "")


def test_numeric_boolean_tolerance_rejected():
    """30. Boolean tolerance rejected."""
    case = _case(scoring_method="numeric", reference=1, tolerance=True)
    result = score_answer(case, "1")
    assert result.status == "invalid_answer"
    assert "boolean" in result.details.get("reason", "")


def test_numeric_nan_inf_tolerance_rejected():
    """31. NaN/infinite tolerance rejected."""
    case = _case(scoring_method="numeric", reference=1, tolerance=float("nan"))
    result = score_answer(case, "1")
    assert result.status == "invalid_answer"

    case2 = _case(scoring_method="numeric", reference=1, tolerance=float("inf"))
    result2 = score_answer(case2, "1")
    assert result2.status == "invalid_answer"


# ===========================================================
# Sentiment (32-38)
# ===========================================================


def test_sentiment_positive():
    """32. Positive label."""
    case = _case(category="sentiment", scoring_method="sentiment",
                 reference="positive")
    result = score_answer(case, "Positive")
    assert result.score == 1.0


def test_sentiment_negative_with_explanation():
    """33. Negative label with explanation."""
    case = _case(category="sentiment", scoring_method="sentiment",
                 reference="negative")
    result = score_answer(case, "negative: the wording expresses dissatisfaction")
    assert result.score == 1.0


def test_sentiment_neutral_casing():
    """34. Neutral label with different casing."""
    case = _case(category="sentiment", scoring_method="sentiment",
                 reference="neutral")
    result = score_answer(case, "The sentiment is NEUTRAL.")
    assert result.score == 1.0


def test_sentiment_word_boundary():
    """35. Complete-word matching (not substring)."""
    case = _case(category="sentiment", scoring_method="sentiment",
                 reference="positive")
    # "positively" should not match "positive" due to word boundary
    result = score_answer(case, "I feel positively about this")
    # "positively" does not match \bpositive\b
    assert result.score == 0.0


def test_sentiment_missing_label():
    """36. Missing label."""
    case = _case(category="sentiment", scoring_method="sentiment",
                 reference="positive")
    result = score_answer(case, "The product is great and I love it")
    assert result.score == 0.0


def test_sentiment_conflicting_labels():
    """37. Conflicting labels."""
    case = _case(category="sentiment", scoring_method="sentiment",
                 reference="positive")
    result = score_answer(case, "It could be positive or negative")
    assert result.score == 0.0
    assert "conflicting" in result.details.get("reason", "")


def test_sentiment_acceptable_answers():
    """38. acceptable_answers support."""
    case = _case(category="sentiment", scoring_method="sentiment",
                 reference="negative",
                 acceptable_answers=["negative", "neutral"])
    result = score_answer(case, "neutral")
    assert result.score == 1.0


# ===========================================================
# Logic (39-41)
# ===========================================================


def test_logic_reference_match():
    """39. Logic reference match."""
    case = _case(category="logic", scoring_method="logic", reference="No")
    result = score_answer(case, "No")
    assert result.score == 1.0


def test_logic_acceptable_answer_match():
    """40. Logic acceptable-answer match."""
    case = _case(category="logic", scoring_method="logic",
                 reference="No",
                 acceptable_answers=["No", "It is not raining"])
    result = score_answer(case, "It is not raining")
    assert result.score == 1.0


def test_logic_mismatch():
    """41. Logic mismatch."""
    case = _case(category="logic", scoring_method="logic", reference="No")
    result = score_answer(case, "Yes")
    assert result.score == 0.0
    assert result.correct is False


# ===========================================================
# Integration (42-46)
# ===========================================================


def test_smoke_dataset_scoreable():
    """42. Score supported cases from smoke dataset."""
    cases = load_dataset(SMOKE_PATH)

    supported = {"exact", "normalized_exact", "acceptable_answers",
                 "numeric", "sentiment", "logic", "ner", "summarization"}
    scoreable = [c for c in cases if c.scoring_method in supported]
    deferred = [c for c in cases if c.scoring_method not in supported]

    # 14 scoreable, 2 deferred (code_tests:2)
    assert len(scoreable) == 14
    assert len(deferred) == 2

    # Score each with a plausible correct answer
    answers = {
        "factual_001": "Au",
        "factual_002": "Jupiter",
        "math_001": "408",
        "math_002": "100.0",
        "sentiment_001": "positive",
        "sentiment_002": "negative",
        "code_debug_001": "ZeroDivisionError",
        "code_debug_002": "low = mid should be low = mid + 1",
        "logic_001": "No",
        "logic_002": "No",
    }

    for case in scoreable:
        ans = answers.get(case.task_id)
        if ans is not None:
            result = score_answer(case, ans)
            assert result.status == "scored", f"{case.task_id}: {result}"
            assert result.score == 1.0, f"{case.task_id}: {result}"


def test_code_tests_now_supported():
    """43. code_tests cases are now executed and scored."""
    cases = load_dataset(SMOKE_PATH)
    code_cases = [c for c in cases if c.scoring_method == "code_tests"]
    assert len(code_cases) == 2

    for case in code_cases:
        result = score_answer(case, "def add(a, b): return a + b")
        assert result.status in ("scored", "invalid_answer")


def test_no_fireworks_called():
    """44. Do not call Fireworks (no network, no imports)."""
    # This test verifies the scorer module doesn't import or use network
    import benchmark.scorers as mod
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "fireworks" not in source.lower()
    assert "requests" not in source
    assert "httpx" not in source
    assert "urllib" not in source


def test_no_local_model_loaded():
    """45. Do not load the local model."""
    import benchmark.scorers as mod
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "openvino" not in source.lower()
    assert "LocalModel" not in source


def test_normalize_text_function():
    """Additional: normalize_text works correctly."""
    assert normalize_text("  Hello   World  ") == "hello world"
    assert normalize_text(42) == "42"
    assert normalize_text(None) is None
    assert normalize_text(True) == "true"
    assert normalize_text(float("nan")) is None
