"""Tests for the summarization scorer."""

from pathlib import Path

import pytest

from benchmark.dataset import BenchmarkCase, load_dataset
from benchmark.scorers import ScoreResult, score_answer, _MAX_SUMMARY_TOKENS


SMOKE_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "smoke" / "tasks.jsonl"


# ===========================================================
# Helpers
# ===========================================================


def _sum_case(reference=None, constraints=None, metadata=None, **overrides) -> BenchmarkCase:
    base = {
        "task_id": "sum_t",
        "category": "summarization",
        "difficulty": "medium",
        "prompt": "Summarize this.",
        "scoring_method": "summarization",
        "reference": reference,
        "constraints": constraints,
        "metadata": metadata,
    }
    base.update(overrides)
    return BenchmarkCase(**base)


# ===========================================================
# General (1-8)
# ===========================================================


def test_perfect_reference_match():
    """1. Perfect reference match."""
    ref = "The committee agreed to increase funding."
    case = _sum_case(reference=ref)
    result = score_answer(case, ref)
    assert result.status == "scored"
    assert result.score == 1.0
    assert result.correct is True


def test_candidate_plain_string():
    """2. Candidate as plain string."""
    case = _sum_case(reference="Hello world.")
    result = score_answer(case, "Hello world.")
    assert result.status == "scored"
    assert result.score == 1.0


def test_candidate_summary_dict():
    """3. Candidate as {'summary': '...'}."""
    case = _sum_case(reference="Hello world.")
    result = score_answer(case, {"summary": "Hello world."})
    assert result.status == "scored"
    assert result.score == 1.0


def test_candidate_answer_dict():
    """4. Candidate as {'answer': '...'}."""
    case = _sum_case(reference="Hello world.")
    result = score_answer(case, {"answer": "Hello world."})
    assert result.status == "scored"
    assert result.score == 1.0


def test_summary_preferred_over_answer():
    """5. 'summary' is preferred when both keys exist."""
    case = _sum_case(reference="correct text")
    result = score_answer(case, {"summary": "correct text", "answer": "wrong text"})
    assert result.score == 1.0


def test_none_candidate_invalid():
    """6. None candidate returns invalid_answer."""
    case = _sum_case(reference="text")
    result = score_answer(case, None)
    assert result.status == "invalid_answer"


def test_empty_string_invalid():
    """7. Empty string returns invalid_answer."""
    case = _sum_case(reference="text")
    result = score_answer(case, "")
    assert result.status == "invalid_answer"


def test_unsupported_types_invalid():
    """8. Unsupported candidate types return invalid_answer."""
    case = _sum_case(reference="text")
    for bad in [True, False, 42, 3.14, [], {"other_key": "val"}]:
        result = score_answer(case, bad)
        assert result.status == "invalid_answer", f"Failed for {bad!r}"


# ===========================================================
# Content metric (9-17)
# ===========================================================


def test_identical_tokens_f1_one():
    """9. Identical token sequences produce ROUGE-L F1 1.0."""
    case = _sum_case(reference="The quick brown fox jumps.")
    result = score_answer(case, "The quick brown fox jumps.")
    assert result.details["rouge_l_f1"] == 1.0


def test_case_normalization():
    """10. Case differences normalize correctly."""
    case = _sum_case(reference="System Failures Increased")
    result = score_answer(case, "system failures increased")
    assert result.details["rouge_l_f1"] == 1.0


def test_unicode_nfkc():
    """11. Unicode NFKC normalization works."""
    # fi ligature normalizes to "fi"
    case = _sum_case(reference="file system")
    result = score_answer(case, "\ufb01le system")
    assert result.details["rouge_l_f1"] == 1.0


def test_punctuation_ignored_for_tokens():
    """12. Surrounding punctuation does not prevent token matches."""
    case = _sum_case(reference="System failures increased.")
    result = score_answer(case, "system failures increased")
    assert result.details["rouge_l_f1"] == 1.0


def test_different_word_order():
    """13. Different word order affects LCS score."""
    case = _sum_case(reference="the cat sat on the mat")
    result = score_answer(case, "on the mat sat the cat")
    # LCS won't be full length due to reordering
    assert result.details["rouge_l_f1"] < 1.0
    assert result.details["rouge_l_f1"] > 0.0


def test_partial_overlap():
    """14. Partial overlap produces correct precision, recall and F1."""
    # Reference: "a b c d" (4 tokens)
    # Candidate: "a b e f" (4 tokens)
    # LCS: "a b" (2 tokens)
    # precision = 2/4 = 0.5, recall = 2/4 = 0.5, f1 = 0.5
    case = _sum_case(reference="a b c d")
    result = score_answer(case, "a b e f")
    assert result.details["rouge_l_precision"] == pytest.approx(0.5)
    assert result.details["rouge_l_recall"] == pytest.approx(0.5)
    assert result.details["rouge_l_f1"] == pytest.approx(0.5)


def test_no_overlap():
    """15. No overlap produces 0.0."""
    case = _sum_case(reference="alpha beta gamma")
    result = score_answer(case, "delta epsilon zeta")
    assert result.details["rouge_l_f1"] == 0.0
    assert result.score == 0.0


def test_empty_token_edge_cases():
    """16. Empty-token edge cases are handled."""
    # Reference with tokens, candidate with only punctuation (no word tokens)
    case = _sum_case(reference="hello world")
    result = score_answer(case, "...")
    assert result.details["rouge_l_f1"] == 0.0


def test_repeated_tokens():
    """17. Repeated tokens are handled correctly."""
    # Reference: "the the the" (3 tokens)
    # Candidate: "the the" (2 tokens)
    # LCS: "the the" (2)
    # precision = 2/2 = 1.0, recall = 2/3
    case = _sum_case(reference="the the the")
    result = score_answer(case, "the the")
    assert result.details["rouge_l_precision"] == pytest.approx(1.0)
    assert result.details["rouge_l_recall"] == pytest.approx(2 / 3)


# ===========================================================
# Constraints (18-32)
# ===========================================================


def test_max_words_passes():
    """18. max_words passes."""
    case = _sum_case(constraints={"max_words": 10})
    result = score_answer(case, "Short summary here.")
    assert result.details["constraints"]["max_words"]["passed"] is True


def test_max_words_fails():
    """19. max_words fails."""
    case = _sum_case(constraints={"max_words": 2})
    result = score_answer(case, "This is a longer summary that exceeds the limit.")
    assert result.details["constraints"]["max_words"]["passed"] is False


def test_max_characters_passes():
    """20. max_characters passes."""
    case = _sum_case(constraints={"max_characters": 100})
    result = score_answer(case, "Short.")
    assert result.details["constraints"]["max_characters"]["passed"] is True


def test_max_characters_fails():
    """21. max_characters fails."""
    case = _sum_case(constraints={"max_characters": 5})
    result = score_answer(case, "This is too long.")
    assert result.details["constraints"]["max_characters"]["passed"] is False


def test_max_sentences_passes():
    """22. max_sentences passes."""
    case = _sum_case(constraints={"max_sentences": 2})
    result = score_answer(case, "First sentence. Second sentence.")
    assert result.details["constraints"]["max_sentences"]["passed"] is True
    assert result.details["sentence_count"] == 2


def test_max_sentences_fails():
    """23. max_sentences fails."""
    case = _sum_case(constraints={"max_sentences": 1})
    result = score_answer(case, "First. Second. Third.")
    assert result.details["constraints"]["max_sentences"]["passed"] is False


def test_no_punctuation_one_sentence():
    """24. Non-empty text without punctuation counts as one sentence."""
    case = _sum_case(constraints={"max_sentences": 1})
    result = score_answer(case, "This has no terminal punctuation")
    assert result.details["sentence_count"] == 1
    assert result.details["constraints"]["max_sentences"]["passed"] is True


def test_multiple_terminal_punctuation():
    """25. Multiple terminal punctuation marks do not create empty sentences."""
    case = _sum_case(constraints={"max_sentences": 3})
    result = score_answer(case, "Really?! Yes! Done.")
    assert result.details["sentence_count"] == 3


def test_newline_separated_sentences():
    """26. Newline-separated summary sentence counting."""
    case = _sum_case(constraints={"max_sentences": 3})
    result = score_answer(case, "Line one\nLine two\nLine three")
    assert result.details["sentence_count"] == 3


def test_multiple_constraints_score():
    """27. Multiple constraints calculate constraint_score correctly."""
    case = _sum_case(constraints={"max_words": 5, "max_sentences": 1})
    # 7 words, 1 sentence -> max_words fails, max_sentences passes
    result = score_answer(case, "This is a summary with many words.")
    assert result.details["constraint_score"] == pytest.approx(0.5)


def test_unknown_constraints_reported():
    """28. Unknown constraints are reported but do not affect score."""
    case = _sum_case(constraints={"max_words": 100, "future_constraint": 5})
    result = score_answer(case, "Short summary.")
    assert "future_constraint" in result.details["unknown_constraints"]
    assert result.details["constraint_score"] == 1.0


def test_boolean_constraint_rejected():
    """29. Boolean constraint value is rejected."""
    case = _sum_case(constraints={"max_words": True})
    result = score_answer(case, "text")
    assert result.status == "invalid_answer"


def test_zero_constraint_rejected():
    """30. Zero constraint value is rejected."""
    case = _sum_case(constraints={"max_words": 0})
    result = score_answer(case, "text")
    assert result.status == "invalid_answer"


def test_negative_constraint_rejected():
    """31. Negative constraint value is rejected."""
    case = _sum_case(constraints={"max_sentences": -1})
    result = score_answer(case, "text")
    assert result.status == "invalid_answer"


def test_non_integer_constraint_rejected():
    """32. Non-integer constraint value is rejected."""
    case = _sum_case(constraints={"max_words": 5.5})
    result = score_answer(case, "text")
    assert result.status == "invalid_answer"


# ===========================================================
# Final scoring (33-37)
# ===========================================================


def test_reference_only_equals_rouge():
    """33. Reference-only score equals ROUGE-L F1."""
    case = _sum_case(reference="alpha beta gamma delta")
    result = score_answer(case, "alpha beta gamma delta")
    assert result.score == result.details["rouge_l_f1"]
    assert result.score == 1.0


def test_constraints_only_equals_constraint_score():
    """34. Constraints-only score equals constraint_score."""
    case = _sum_case(constraints={"max_words": 10, "max_sentences": 2})
    result = score_answer(case, "Short text.")
    assert result.score == result.details["constraint_score"]


def test_reference_plus_constraints_multiplied():
    """35. Reference plus constraints multiplies both scores."""
    case = _sum_case(reference="a b c d", constraints={"max_words": 2})
    # Candidate "a b" has 2 tokens matching ref "a b c d" (4 tokens)
    # LCS = 2, precision = 2/2 = 1.0, recall = 2/4 = 0.5, f1 = 2/3
    # max_words: 2 <= 2 -> passes, constraint_score = 1.0
    # final = f1 * 1.0 = 2/3
    result = score_answer(case, "a b")
    assert result.score == pytest.approx(2 / 3)


def test_failed_constraint_lowers_score():
    """36. Failed constraint lowers final score."""
    case = _sum_case(reference="a b c d", constraints={"max_words": 1})
    # Candidate "a b c d" matches perfectly (f1=1.0) but has 4 words > 1
    # constraint_score = 0.0, final = 1.0 * 0.0 = 0.0
    result = score_answer(case, "a b c d")
    assert result.score == 0.0


def test_final_score_bounded():
    """37. Final score remains between 0 and 1."""
    case = _sum_case(reference="x y z", constraints={"max_words": 100})
    for answer in ["x y z", "a b c", "x", ""]:
        if not answer:
            continue
        result = score_answer(case, answer)
        if result.status == "scored":
            assert 0.0 <= result.score <= 1.0


# ===========================================================
# Correctness (38-43)
# ===========================================================


def test_perfect_summary_correct():
    """38. Perfect summary with constraints passes."""
    case = _sum_case(reference="the quick fox", constraints={"max_words": 10})
    result = score_answer(case, "the quick fox")
    assert result.correct is True


def test_content_below_threshold_fails():
    """39. Content below threshold fails."""
    case = _sum_case(reference="alpha beta gamma delta epsilon")
    # Candidate shares no tokens
    result = score_answer(case, "zeta eta theta iota kappa")
    assert result.correct is False


def test_constraint_violation_fails():
    """40. Constraint violation fails even when content overlap is high."""
    case = _sum_case(reference="a b c", constraints={"max_words": 1})
    result = score_answer(case, "a b c")
    assert result.correct is False


def test_custom_minimum_content_score():
    """41. Custom metadata minimum_content_score is respected."""
    # With threshold 0.9, partial overlap won't pass
    case = _sum_case(reference="a b c d e f g h i j",
                     metadata={"minimum_content_score": 0.9})
    # Candidate shares 5/10 tokens
    result = score_answer(case, "a b c d e x y z w v")
    assert result.details["minimum_content_score"] == 0.9
    assert result.correct is False


def test_invalid_metadata_threshold():
    """42. Invalid metadata threshold is rejected."""
    case = _sum_case(reference="text", metadata={"minimum_content_score": True})
    result = score_answer(case, "text")
    assert result.status == "invalid_answer"

    case2 = _sum_case(reference="text", metadata={"minimum_content_score": 1.5})
    result2 = score_answer(case2, "text")
    assert result2.status == "invalid_answer"

    case3 = _sum_case(reference="text", metadata={"minimum_content_score": float("nan")})
    result3 = score_answer(case3, "text")
    assert result3.status == "invalid_answer"


def test_default_threshold():
    """43. Default content threshold is 0.60."""
    case = _sum_case(reference="a b c d e f g h i j")
    result = score_answer(case, "a b c d e f g h i j")
    assert result.details["minimum_content_score"] == 0.60


# ===========================================================
# Safety (44-45)
# ===========================================================


def test_candidate_above_token_limit():
    """44. Candidate above token safety limit is rejected."""
    case = _sum_case(reference="short")
    long_answer = " ".join(["word"] * (_MAX_SUMMARY_TOKENS + 1))
    result = score_answer(case, long_answer)
    assert result.status == "invalid_answer"
    assert "safety limit" in result.details["reason"]


def test_reference_above_token_limit():
    """45. Reference above token safety limit is rejected."""
    long_ref = " ".join(["word"] * (_MAX_SUMMARY_TOKENS + 1))
    case = _sum_case(reference=long_ref)
    result = score_answer(case, "short answer")
    assert result.status == "invalid_answer"
    assert "safety limit" in result.details["reason"]


# ===========================================================
# Integration (46-50)
# ===========================================================


def test_smoke_summarization_cases():
    """46. The two summarization smoke cases can be scored."""
    cases = load_dataset(SMOKE_PATH)
    sum_cases = [c for c in cases if c.scoring_method == "summarization"]
    assert len(sum_cases) == 2

    # Score summarization_001 with a plausible answer
    r1 = score_answer(sum_cases[0],
                      "The committee agreed to increase infrastructure funding by 12% and cut administrative costs.")
    assert r1.status == "scored"
    assert r1.score > 0.0
    assert r1.details["rouge_l_f1"] > 0.0

    # Score summarization_002 (constraints only, no reference)
    r2 = score_answer(sum_cases[1],
                      "Daily moderate exercise reduces cardiovascular risk by 30% in adults over 50.")
    assert r2.status == "scored"
    assert r2.score > 0.0


def test_code_tests_now_supported():
    """47. code_tests is now supported."""
    case = BenchmarkCase(task_id="t", category="code_generation", difficulty="easy",
                         prompt="q", scoring_method="code_tests",
                         tests=[{"call": "f()", "expected": 1}])
    result = score_answer(case, "def f(): return 1")
    assert result.status == "scored"
    assert result.score == 1.0


def test_ner_and_core_scorers_unchanged():
    """48. NER and all six core scorers remain unchanged."""
    # Exact
    case = BenchmarkCase(task_id="t", category="factual", difficulty="easy",
                         prompt="q", scoring_method="exact", reference="Au")
    assert score_answer(case, "Au").score == 1.0

    # NER
    case2 = BenchmarkCase(task_id="t", category="ner", difficulty="easy",
                          prompt="q", scoring_method="ner",
                          reference=[{"text": "X", "type": "PERSON"}])
    assert score_answer(case2, [{"text": "X", "type": "PERSON"}]).score == 1.0

    # Numeric
    case3 = BenchmarkCase(task_id="t", category="math", difficulty="easy",
                          prompt="q", scoring_method="numeric", reference=42)
    assert score_answer(case3, "42").score == 1.0


def test_no_fireworks_called():
    """49. No Fireworks call is made."""
    import benchmark.scorers as mod
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "fireworks" not in source.lower()
    assert "requests" not in source
    assert "httpx" not in source


def test_no_local_model_loaded():
    """50. The local model is not loaded."""
    import benchmark.scorers as mod
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "openvino" not in source.lower()
    assert "LocalModel" not in source
