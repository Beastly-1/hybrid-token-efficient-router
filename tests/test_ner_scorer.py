"""Tests for the NER scorer."""

import json
from pathlib import Path

import pytest

from benchmark.dataset import BenchmarkCase, load_dataset
from benchmark.scorers import ScoreResult, score_answer


SMOKE_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "smoke" / "tasks.jsonl"


# ===========================================================
# Helpers
# ===========================================================


def _ner_case(reference=None, **overrides) -> BenchmarkCase:
    base = {
        "task_id": "ner_t",
        "category": "ner",
        "difficulty": "easy",
        "prompt": "Extract entities.",
        "scoring_method": "ner",
        "reference": reference if reference is not None else [
            {"text": "Maya", "type": "PERSON"},
            {"text": "Acme Labs", "type": "ORGANIZATION"},
            {"text": "Bengaluru", "type": "LOCATION"},
            {"text": "12 June 2026", "type": "DATE"},
        ],
    }
    base.update(overrides)
    return BenchmarkCase(**base)


# ===========================================================
# 1. Perfect entity match
# ===========================================================


def test_perfect_match():
    case = _ner_case()
    answer = [
        {"text": "Maya", "type": "PERSON"},
        {"text": "Acme Labs", "type": "ORGANIZATION"},
        {"text": "Bengaluru", "type": "LOCATION"},
        {"text": "12 June 2026", "type": "DATE"},
    ]
    result = score_answer(case, answer)
    assert result.status == "scored"
    assert result.score == 1.0
    assert result.correct is True
    assert result.details["precision"] == 1.0
    assert result.details["recall"] == 1.0


# ===========================================================
# 2. Candidate as list of dictionaries
# ===========================================================


def test_candidate_list_of_dicts():
    case = _ner_case(reference=[{"text": "X", "type": "PERSON"}])
    result = score_answer(case, [{"text": "X", "type": "PERSON"}])
    assert result.score == 1.0


# ===========================================================
# 3. Candidate as {"entities": [...]}
# ===========================================================


def test_candidate_entities_dict():
    case = _ner_case(reference=[{"text": "X", "type": "PERSON"}])
    result = score_answer(case, {"entities": [{"text": "X", "type": "PERSON"}]})
    assert result.score == 1.0


# ===========================================================
# 4. Candidate as JSON array string
# ===========================================================


def test_candidate_json_array_string():
    case = _ner_case(reference=[{"text": "X", "type": "PERSON"}])
    answer = json.dumps([{"text": "X", "type": "PERSON"}])
    result = score_answer(case, answer)
    assert result.score == 1.0


# ===========================================================
# 5. Candidate as JSON object string
# ===========================================================


def test_candidate_json_object_string():
    case = _ner_case(reference=[{"text": "X", "type": "PERSON"}])
    answer = json.dumps({"entities": [{"text": "X", "type": "PERSON"}]})
    result = score_answer(case, answer)
    assert result.score == 1.0


# ===========================================================
# 6. JSON inside a ```json Markdown fence
# ===========================================================


def test_candidate_markdown_fence():
    case = _ner_case(reference=[{"text": "Maya", "type": "PERSON"}])
    answer = '```json\n[{"text": "Maya", "type": "PERSON"}]\n```'
    result = score_answer(case, answer)
    assert result.score == 1.0


# ===========================================================
# 7. Pipe-separated line format
# ===========================================================


def test_candidate_pipe_separated():
    case = _ner_case(reference=[
        {"text": "Maya", "type": "PERSON"},
        {"text": "Acme Labs", "type": "ORGANIZATION"},
    ])
    answer = "Maya | PERSON\nAcme Labs | ORGANIZATION"
    result = score_answer(case, answer)
    assert result.score == 1.0


# ===========================================================
# 8. Colon-separated line format
# ===========================================================


def test_candidate_colon_separated():
    case = _ner_case(reference=[
        {"text": "Maya", "type": "PERSON"},
        {"text": "Bengaluru", "type": "LOCATION"},
    ])
    answer = "Maya: PERSON\nBengaluru: LOCATION"
    result = score_answer(case, answer)
    assert result.score == 1.0


# ===========================================================
# 9. Entity text case normalization
# ===========================================================


def test_text_case_normalization():
    case = _ner_case(reference=[{"text": "Maya", "type": "PERSON"}])
    result = score_answer(case, [{"text": "maya", "type": "PERSON"}])
    assert result.score == 1.0


# ===========================================================
# 10. Repeated-whitespace normalization
# ===========================================================


def test_text_whitespace_normalization():
    case = _ner_case(reference=[{"text": "Acme Labs", "type": "ORGANIZATION"}])
    result = score_answer(case, [{"text": "Acme   Labs", "type": "ORGANIZATION"}])
    assert result.score == 1.0


# ===========================================================
# 11. Unicode NFKC normalization
# ===========================================================


def test_text_unicode_nfkc():
    # fi ligature -> "fi"
    case = _ner_case(reference=[{"text": "file", "type": "ORGANIZATION"}])
    result = score_answer(case, [{"text": "\ufb01le", "type": "ORGANIZATION"}])
    assert result.score == 1.0


# ===========================================================
# 12. Entity-type case normalization
# ===========================================================


def test_type_case_normalization():
    case = _ner_case(reference=[{"text": "Maya", "type": "PERSON"}])
    result = score_answer(case, [{"text": "Maya", "type": "person"}])
    assert result.score == 1.0


# ===========================================================
# 13. PERSON/PER alias
# ===========================================================


def test_person_per_alias():
    case = _ner_case(reference=[{"text": "Maya", "type": "PERSON"}])
    result = score_answer(case, [{"text": "Maya", "type": "PER"}])
    assert result.score == 1.0


# ===========================================================
# 14. ORGANIZATION/ORGANISATION/ORG aliases
# ===========================================================


def test_organization_aliases():
    case = _ner_case(reference=[{"text": "Acme", "type": "ORGANIZATION"}])

    r1 = score_answer(case, [{"text": "Acme", "type": "ORGANISATION"}])
    assert r1.score == 1.0

    r2 = score_answer(case, [{"text": "Acme", "type": "ORG"}])
    assert r2.score == 1.0


# ===========================================================
# 15. LOCATION/LOC/PLACE aliases
# ===========================================================


def test_location_aliases():
    case = _ner_case(reference=[{"text": "Berlin", "type": "LOCATION"}])

    r1 = score_answer(case, [{"text": "Berlin", "type": "LOC"}])
    assert r1.score == 1.0

    r2 = score_answer(case, [{"text": "Berlin", "type": "PLACE"}])
    assert r2.score == 1.0


# ===========================================================
# 16. Correct text but wrong type does not match
# ===========================================================


def test_wrong_type_no_match():
    case = _ner_case(reference=[{"text": "Maya", "type": "PERSON"}])
    result = score_answer(case, [{"text": "Maya", "type": "ORGANIZATION"}])
    assert result.score == 0.0
    assert result.correct is False


# ===========================================================
# 17. Partial set gives correct precision, recall and F1
# ===========================================================


def test_partial_match_metrics():
    case = _ner_case(reference=[
        {"text": "A", "type": "PERSON"},
        {"text": "B", "type": "LOCATION"},
        {"text": "C", "type": "DATE"},
    ])
    # Predict 2 correct out of 3 reference, plus 1 extra
    answer = [
        {"text": "A", "type": "PERSON"},
        {"text": "B", "type": "LOCATION"},
        {"text": "D", "type": "ORGANIZATION"},
    ]
    result = score_answer(case, answer)
    assert result.status == "scored"
    # tp=2, fp=1, fn=1
    assert result.details["true_positives"] == 2
    assert result.details["false_positives"] == 1
    assert result.details["false_negatives"] == 1
    # precision = 2/3, recall = 2/3, f1 = 2/3
    assert result.details["precision"] == pytest.approx(2 / 3)
    assert result.details["recall"] == pytest.approx(2 / 3)
    assert result.details["f1"] == pytest.approx(2 / 3)
    assert result.score == pytest.approx(2 / 3)


# ===========================================================
# 18. Extra predicted entity lowers precision
# ===========================================================


def test_extra_prediction_lowers_precision():
    case = _ner_case(reference=[{"text": "A", "type": "PERSON"}])
    answer = [
        {"text": "A", "type": "PERSON"},
        {"text": "B", "type": "LOCATION"},
    ]
    result = score_answer(case, answer)
    # tp=1, fp=1, fn=0 -> precision=0.5, recall=1.0, f1=2/3
    assert result.details["precision"] == pytest.approx(0.5)
    assert result.details["recall"] == 1.0
    assert result.score == pytest.approx(2 / 3)


# ===========================================================
# 19. Missing entity lowers recall
# ===========================================================


def test_missing_entity_lowers_recall():
    case = _ner_case(reference=[
        {"text": "A", "type": "PERSON"},
        {"text": "B", "type": "LOCATION"},
    ])
    answer = [{"text": "A", "type": "PERSON"}]
    result = score_answer(case, answer)
    # tp=1, fp=0, fn=1 -> precision=1.0, recall=0.5, f1=2/3
    assert result.details["precision"] == 1.0
    assert result.details["recall"] == pytest.approx(0.5)
    assert result.score == pytest.approx(2 / 3)


# ===========================================================
# 20. Duplicate predictions do not change score
# ===========================================================


def test_duplicate_predictions_no_change():
    case = _ner_case(reference=[{"text": "A", "type": "PERSON"}])
    answer = [
        {"text": "A", "type": "PERSON"},
        {"text": "A", "type": "PERSON"},
        {"text": "A", "type": "PERSON"},
    ]
    result = score_answer(case, answer)
    assert result.score == 1.0
    assert result.details["predicted_count"] == 1


# ===========================================================
# 21. Duplicate references do not change score
# ===========================================================


def test_duplicate_references_no_change():
    case = _ner_case(reference=[
        {"text": "A", "type": "PERSON"},
        {"text": "A", "type": "PERSON"},
    ])
    answer = [{"text": "A", "type": "PERSON"}]
    result = score_answer(case, answer)
    assert result.score == 1.0
    assert result.details["reference_count"] == 1


# ===========================================================
# 22. Empty prediction with non-empty reference scores 0
# ===========================================================


def test_empty_prediction_scores_zero():
    case = _ner_case(reference=[{"text": "A", "type": "PERSON"}])
    result = score_answer(case, [])
    assert result.status == "scored"
    assert result.score == 0.0
    assert result.correct is False


# ===========================================================
# 23. Both lists empty score 1
# ===========================================================


def test_both_empty_scores_one():
    case = _ner_case(reference=[])
    result = score_answer(case, [])
    assert result.score == 1.0
    assert result.correct is True


# ===========================================================
# 24. Malformed JSON returns invalid_answer
# ===========================================================


def test_malformed_json_invalid():
    case = _ner_case()
    result = score_answer(case, "{not valid json[")
    assert result.status == "invalid_answer"


# ===========================================================
# 25. Ordinary prose returns invalid_answer
# ===========================================================


def test_prose_invalid():
    case = _ner_case()
    result = score_answer(case, "Maya is a person and Acme Labs is an organization.")
    assert result.status == "invalid_answer"


# ===========================================================
# 26. Missing text field returns invalid_answer
# ===========================================================


def test_missing_text_field():
    case = _ner_case()
    result = score_answer(case, [{"type": "PERSON"}])
    assert result.status == "invalid_answer"


# ===========================================================
# 27. Missing type field returns invalid_answer
# ===========================================================


def test_missing_type_field():
    case = _ner_case()
    result = score_answer(case, [{"text": "Maya"}])
    assert result.status == "invalid_answer"


# ===========================================================
# 28. Empty text returns invalid_answer
# ===========================================================


def test_empty_text_invalid():
    case = _ner_case()
    result = score_answer(case, [{"text": "", "type": "PERSON"}])
    assert result.status == "invalid_answer"


# ===========================================================
# 29. Empty type returns invalid_answer
# ===========================================================


def test_empty_type_invalid():
    case = _ner_case()
    result = score_answer(case, [{"text": "Maya", "type": ""}])
    assert result.status == "invalid_answer"


# ===========================================================
# 30. Null entity item returns invalid_answer
# ===========================================================


def test_null_entity_invalid():
    case = _ner_case()
    result = score_answer(case, [None, {"text": "Maya", "type": "PERSON"}])
    assert result.status == "invalid_answer"


# ===========================================================
# 31. Mixed valid and malformed entities returns invalid_answer
# ===========================================================


def test_mixed_valid_malformed_invalid():
    case = _ner_case()
    result = score_answer(case, [
        {"text": "Maya", "type": "PERSON"},
        {"text": "Acme", "missing_type": "ORG"},
    ])
    assert result.status == "invalid_answer"


# ===========================================================
# 32. Unknown non-empty entity type is handled deterministically
# ===========================================================


def test_unknown_type_handled():
    case = _ner_case(reference=[{"text": "X", "type": "CUSTOM_TYPE"}])
    result = score_answer(case, [{"text": "X", "type": "custom_type"}])
    assert result.score == 1.0  # uppercase normalization matches


# ===========================================================
# 33. correct is True only for F1 1.0
# ===========================================================


def test_correct_only_for_perfect_f1():
    case = _ner_case(reference=[
        {"text": "A", "type": "PERSON"},
        {"text": "B", "type": "LOCATION"},
    ])
    # Partial match
    result = score_answer(case, [{"text": "A", "type": "PERSON"}])
    assert result.score < 1.0
    assert result.correct is False

    # Perfect match
    result2 = score_answer(case, [
        {"text": "A", "type": "PERSON"},
        {"text": "B", "type": "LOCATION"},
    ])
    assert result2.score == 1.0
    assert result2.correct is True


# ===========================================================
# 34. score remains within 0.0–1.0
# ===========================================================


def test_score_bounded():
    case = _ner_case(reference=[
        {"text": "A", "type": "PERSON"},
        {"text": "B", "type": "LOCATION"},
    ])
    # Various predictions
    for answer in [
        [],
        [{"text": "A", "type": "PERSON"}],
        [{"text": "A", "type": "PERSON"}, {"text": "B", "type": "LOCATION"}],
        [{"text": "X", "type": "Y"}, {"text": "Z", "type": "W"}],
    ]:
        result = score_answer(case, answer)
        assert result.status == "scored"
        assert 0.0 <= result.score <= 1.0


# ===========================================================
# 35. Existing non-NER scorers remain unchanged
# ===========================================================


def test_existing_scorers_unchanged():
    # Exact
    case = BenchmarkCase(task_id="t", category="factual", difficulty="easy",
                         prompt="q", scoring_method="exact", reference="Au")
    assert score_answer(case, "Au").score == 1.0
    assert score_answer(case, "au").score == 0.0

    # Numeric
    case2 = BenchmarkCase(task_id="t", category="math", difficulty="easy",
                          prompt="q", scoring_method="numeric", reference=42)
    assert score_answer(case2, "42").score == 1.0


# ===========================================================
# 36. The two NER smoke cases can be scored
# ===========================================================


def test_smoke_ner_cases():
    cases = load_dataset(SMOKE_PATH)
    ner_cases = [c for c in cases if c.scoring_method == "ner"]
    assert len(ner_cases) == 2

    # Score ner_001 with correct answer
    result1 = score_answer(ner_cases[0], [
        {"text": "Maya", "type": "PERSON"},
        {"text": "Acme Labs", "type": "ORGANIZATION"},
        {"text": "Bengaluru", "type": "LOCATION"},
        {"text": "12 June 2026", "type": "DATE"},
    ])
    assert result1.status == "scored"
    assert result1.score == 1.0

    # Score ner_002 with correct answer
    result2 = score_answer(ner_cases[1], [
        {"text": "March 2024", "type": "DATE"},
        {"text": "Tesla", "type": "ORGANIZATION"},
        {"text": "Samsung", "type": "ORGANIZATION"},
        {"text": "Austin", "type": "LOCATION"},
    ])
    assert result2.status == "scored"
    assert result2.score == 1.0


# ===========================================================
# 37. Summarization is now supported
# ===========================================================


def test_summarization_supported():
    case = BenchmarkCase(task_id="t", category="summarization", difficulty="easy",
                         prompt="q", scoring_method="summarization",
                         reference="summary")
    result = score_answer(case, "some summary")
    assert result.status == "scored"
    assert result.score is not None


# ===========================================================
# 38. code_tests remains unsupported
# ===========================================================


def test_code_tests_now_supported():
    case = BenchmarkCase(task_id="t", category="code_generation", difficulty="easy",
                         prompt="q", scoring_method="code_tests",
                         tests=[{"call": "f()", "expected": 1}])
    result = score_answer(case, "def f(): return 1")
    assert result.status == "scored"
    assert result.score == 1.0
