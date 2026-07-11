"""Tests for benchmark dataset schema, loader and validator."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.dataset import (
    VALID_CATEGORIES,
    VALID_DIFFICULTIES,
    VALID_SCORING_METHODS,
    BenchmarkCase,
    DatasetValidationError,
    dataset_summary,
    load_dataset,
    validate_case,
    validate_dataset,
)

SMOKE_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "smoke" / "tasks.jsonl"


# ===========================================================
# Helpers
# ===========================================================


def _write_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            if isinstance(r, str):
                f.write(r + "\n")
            else:
                f.write(json.dumps(r) + "\n")


def _minimal_case(**overrides) -> dict:
    base = {
        "task_id": "test_001",
        "category": "factual",
        "difficulty": "easy",
        "prompt": "What is 1+1?",
        "scoring_method": "exact",
        "reference": "2",
    }
    base.update(overrides)
    return base


# ===========================================================
# 1. Valid minimum benchmark case
# ===========================================================


def test_valid_minimum_case():
    case = validate_case(_minimal_case(), 1, "test.jsonl")
    assert isinstance(case, BenchmarkCase)
    assert case.task_id == "test_001"
    assert case.category == "factual"


# ===========================================================
# 2. All eight categories accepted
# ===========================================================


@pytest.mark.parametrize("category", sorted(VALID_CATEGORIES))
def test_all_categories_accepted(category):
    data = _minimal_case(category=category)
    # Adjust scoring method to satisfy category-specific requirements
    if category == "ner":
        data["scoring_method"] = "ner"
        data["reference"] = [{"text": "X", "type": "PERSON"}]
    elif category == "code_generation":
        data["scoring_method"] = "code_tests"
        data["tests"] = [{"call": "f()", "expected": 1}]
    elif category == "sentiment":
        data["scoring_method"] = "sentiment"
        data["reference"] = "positive"
    case = validate_case(data, 1, "test.jsonl")
    assert case.category == category


# ===========================================================
# 3. easy, medium and hard accepted
# ===========================================================


@pytest.mark.parametrize("difficulty", sorted(VALID_DIFFICULTIES))
def test_all_difficulties_accepted(difficulty):
    case = validate_case(_minimal_case(difficulty=difficulty), 1, "test.jsonl")
    assert case.difficulty == difficulty


# ===========================================================
# 4. Unknown category rejected
# ===========================================================


def test_unknown_category_rejected():
    with pytest.raises(DatasetValidationError, match="category"):
        validate_case(_minimal_case(category="unknown_cat"), 1, "test.jsonl")


# ===========================================================
# 5. Unknown difficulty rejected
# ===========================================================


def test_unknown_difficulty_rejected():
    with pytest.raises(DatasetValidationError, match="difficulty"):
        validate_case(_minimal_case(difficulty="extreme"), 1, "test.jsonl")


# ===========================================================
# 6. Unknown scoring method rejected
# ===========================================================


def test_unknown_scoring_method_rejected():
    with pytest.raises(DatasetValidationError, match="scoring_method"):
        validate_case(_minimal_case(scoring_method="magic"), 1, "test.jsonl")


# ===========================================================
# 7. Empty task_id rejected
# ===========================================================


def test_empty_task_id_rejected():
    with pytest.raises(DatasetValidationError, match="task_id"):
        validate_case(_minimal_case(task_id=""), 1, "test.jsonl")


def test_whitespace_task_id_rejected():
    with pytest.raises(DatasetValidationError, match="task_id"):
        validate_case(_minimal_case(task_id="   "), 1, "test.jsonl")


# ===========================================================
# 8. Empty prompt rejected
# ===========================================================


def test_empty_prompt_rejected():
    with pytest.raises(DatasetValidationError, match="prompt"):
        validate_case(_minimal_case(prompt=""), 1, "test.jsonl")


# ===========================================================
# 9. Duplicate task IDs rejected
# ===========================================================


def test_duplicate_task_ids_rejected(tmp_path):
    f = tmp_path / "dup.jsonl"
    _write_jsonl(f, [_minimal_case(task_id="dup"), _minimal_case(task_id="dup")])
    with pytest.raises(DatasetValidationError, match="duplicate task_id"):
        load_dataset(f)


# ===========================================================
# 10. Malformed JSON reports the correct line number
# ===========================================================


def test_malformed_json_line_number(tmp_path):
    f = tmp_path / "bad.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_minimal_case()) + "\n")
        fh.write("not valid json\n")
        fh.write(json.dumps(_minimal_case(task_id="x")) + "\n")

    with pytest.raises(DatasetValidationError, match=":2:"):
        load_dataset(f)


# ===========================================================
# 11. Non-object JSONL record rejected
# ===========================================================


def test_non_object_record_rejected(tmp_path):
    f = tmp_path / "arr.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        fh.write("[1, 2, 3]\n")

    with pytest.raises(DatasetValidationError, match="must be a JSON object"):
        load_dataset(f)


def test_string_record_rejected(tmp_path):
    f = tmp_path / "str.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        fh.write('"just a string"\n')

    with pytest.raises(DatasetValidationError, match="must be a JSON object"):
        load_dataset(f)


# ===========================================================
# 12. Missing file produces a clear error
# ===========================================================


def test_missing_file_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_dataset(tmp_path / "nonexistent.jsonl")


# ===========================================================
# 13. Blank lines are skipped
# ===========================================================


def test_blank_lines_skipped(tmp_path):
    f = tmp_path / "blanks.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write(json.dumps(_minimal_case(task_id="a")) + "\n")
        fh.write("   \n")
        fh.write(json.dumps(_minimal_case(task_id="b")) + "\n")
        fh.write("\n")

    cases = load_dataset(f)
    assert len(cases) == 2


# ===========================================================
# 14. Input order is preserved
# ===========================================================


def test_input_order_preserved(tmp_path):
    f = tmp_path / "order.jsonl"
    ids = ["z_first", "a_second", "m_third"]
    _write_jsonl(f, [_minimal_case(task_id=tid) for tid in ids])

    cases = load_dataset(f)
    assert [c.task_id for c in cases] == ids


# ===========================================================
# 15. Numeric reference validation
# ===========================================================


def test_numeric_reference_valid():
    case = validate_case(
        _minimal_case(scoring_method="numeric", reference=42), 1, "t"
    )
    assert case.reference == 42


def test_numeric_string_reference_valid():
    case = validate_case(
        _minimal_case(scoring_method="numeric", reference="3.14"), 1, "t"
    )
    assert case.reference == "3.14"


def test_numeric_reference_missing():
    with pytest.raises(DatasetValidationError, match="numeric.*requires reference"):
        validate_case(
            _minimal_case(scoring_method="numeric", reference=None), 1, "t"
        )


def test_numeric_reference_non_numeric():
    with pytest.raises(DatasetValidationError, match="numeric reference"):
        validate_case(
            _minimal_case(scoring_method="numeric", reference="abc"), 1, "t"
        )


# ===========================================================
# 16. Negative tolerance rejected
# ===========================================================


def test_negative_tolerance_rejected():
    with pytest.raises(DatasetValidationError, match="tolerance.*non-negative"):
        validate_case(
            _minimal_case(scoring_method="numeric", reference=1, tolerance=-0.5), 1, "t"
        )


# ===========================================================
# 17. Boolean tolerance rejected
# ===========================================================


def test_boolean_tolerance_rejected():
    with pytest.raises(DatasetValidationError, match="tolerance.*boolean"):
        validate_case(
            _minimal_case(scoring_method="numeric", reference=1, tolerance=True), 1, "t"
        )


# ===========================================================
# 18. NaN and infinity rejected
# ===========================================================


def test_nan_tolerance_rejected():
    with pytest.raises(DatasetValidationError, match="tolerance.*finite"):
        validate_case(
            _minimal_case(scoring_method="numeric", reference=1, tolerance=float("nan")), 1, "t"
        )


def test_inf_tolerance_rejected():
    with pytest.raises(DatasetValidationError, match="tolerance.*finite"):
        validate_case(
            _minimal_case(scoring_method="numeric", reference=1, tolerance=float("inf")), 1, "t"
        )


# ===========================================================
# 19. Empty acceptable_answers rejected
# ===========================================================


def test_empty_acceptable_answers_rejected():
    with pytest.raises(DatasetValidationError, match="non-empty acceptable_answers"):
        validate_case(
            _minimal_case(scoring_method="acceptable_answers", acceptable_answers=[]),
            1, "t",
        )


def test_null_in_acceptable_answers_rejected():
    with pytest.raises(DatasetValidationError, match="must not contain null"):
        validate_case(
            _minimal_case(scoring_method="acceptable_answers", acceptable_answers=["ok", None]),
            1, "t",
        )


# ===========================================================
# 20. Invalid sentiment label rejected
# ===========================================================


def test_invalid_sentiment_label_rejected():
    with pytest.raises(DatasetValidationError, match="sentiment label"):
        validate_case(
            _minimal_case(
                category="sentiment",
                scoring_method="sentiment",
                reference="happy",
            ),
            1, "t",
        )


def test_valid_sentiment_labels_accepted():
    for label in ("positive", "negative", "neutral"):
        case = validate_case(
            _minimal_case(category="sentiment", scoring_method="sentiment", reference=label),
            1, "t",
        )
        assert case.reference == label


# ===========================================================
# 21. Summarization constraints validated
# ===========================================================


def test_summarization_requires_reference_or_constraints():
    with pytest.raises(DatasetValidationError, match="summarization.*requires"):
        validate_case(
            _minimal_case(
                category="summarization",
                scoring_method="summarization",
                reference=None,
                constraints=None,
            ),
            1, "t",
        )


def test_summarization_invalid_constraint():
    with pytest.raises(DatasetValidationError, match="max_words.*positive integer"):
        validate_case(
            _minimal_case(
                category="summarization",
                scoring_method="summarization",
                constraints={"max_words": -5},
            ),
            1, "t",
        )


def test_summarization_boolean_constraint_rejected():
    with pytest.raises(DatasetValidationError, match="max_sentences.*positive integer"):
        validate_case(
            _minimal_case(
                category="summarization",
                scoring_method="summarization",
                constraints={"max_sentences": True},
            ),
            1, "t",
        )


def test_summarization_valid_constraints():
    case = validate_case(
        _minimal_case(
            category="summarization",
            scoring_method="summarization",
            constraints={"max_words": 50, "max_sentences": 2},
        ),
        1, "t",
    )
    assert case.constraints == {"max_words": 50, "max_sentences": 2}


# ===========================================================
# 22. Invalid NER reference rejected
# ===========================================================


def test_ner_reference_not_list():
    with pytest.raises(DatasetValidationError, match="ner.*list"):
        validate_case(
            _minimal_case(category="ner", scoring_method="ner", reference="not a list"),
            1, "t",
        )


def test_ner_reference_empty_list():
    with pytest.raises(DatasetValidationError, match="ner.*non-empty"):
        validate_case(
            _minimal_case(category="ner", scoring_method="ner", reference=[]),
            1, "t",
        )


def test_ner_entity_missing_fields():
    with pytest.raises(DatasetValidationError, match="text.*type"):
        validate_case(
            _minimal_case(
                category="ner",
                scoring_method="ner",
                reference=[{"text": "X"}],
            ),
            1, "t",
        )


# ===========================================================
# 23. Missing code tests rejected
# ===========================================================


def test_code_tests_missing():
    with pytest.raises(DatasetValidationError, match="code_tests.*non-empty tests"):
        validate_case(
            _minimal_case(
                category="code_generation",
                scoring_method="code_tests",
                tests=None,
            ),
            1, "t",
        )


def test_code_tests_empty_list():
    with pytest.raises(DatasetValidationError, match="code_tests.*non-empty tests"):
        validate_case(
            _minimal_case(
                category="code_generation",
                scoring_method="code_tests",
                tests=[],
            ),
            1, "t",
        )


# ===========================================================
# 24. Invalid code-test object rejected
# ===========================================================


def test_code_test_not_object():
    with pytest.raises(DatasetValidationError, match="tests\\[0\\].*must be an object"):
        validate_case(
            _minimal_case(
                category="code_generation",
                scoring_method="code_tests",
                tests=["not an object"],
            ),
            1, "t",
        )


def test_code_test_missing_fields():
    with pytest.raises(DatasetValidationError, match="tests\\[0\\].*call.*expected"):
        validate_case(
            _minimal_case(
                category="code_generation",
                scoring_method="code_tests",
                tests=[{"call": "f()"}],
            ),
            1, "t",
        )


# ===========================================================
# 25. Logic case without reference or accepted answer rejected
# ===========================================================


def test_logic_requires_reference_or_answers():
    with pytest.raises(DatasetValidationError, match="logic.*requires"):
        validate_case(
            _minimal_case(
                category="logic",
                scoring_method="logic",
                reference=None,
                acceptable_answers=None,
            ),
            1, "t",
        )


# ===========================================================
# 26. Dataset summary is correct
# ===========================================================


def test_dataset_summary_correct(tmp_path):
    f = tmp_path / "ds.jsonl"
    _write_jsonl(f, [
        _minimal_case(task_id="a", category="factual", difficulty="easy"),
        _minimal_case(task_id="b", category="factual", difficulty="medium"),
        _minimal_case(task_id="c", category="math", difficulty="hard", scoring_method="numeric", reference=5),
    ])

    cases = load_dataset(f)
    summary = dataset_summary(cases)

    assert summary["total_cases"] == 3
    assert summary["by_category"] == {"factual": 2, "math": 1}
    assert summary["by_difficulty"] == {"easy": 1, "hard": 1, "medium": 1}
    assert summary["duplicate_count"] == 0


# ===========================================================
# 27. Smoke dataset contains exactly 16 cases
# ===========================================================


def test_smoke_dataset_16_cases():
    cases = load_dataset(SMOKE_PATH)
    assert len(cases) == 16


# ===========================================================
# 28. Smoke dataset contains exactly 2 cases per category
# ===========================================================


def test_smoke_dataset_2_per_category():
    cases = load_dataset(SMOKE_PATH)
    summary = dataset_summary(cases)
    for cat in VALID_CATEGORIES:
        assert summary["by_category"].get(cat) == 2, f"Expected 2 for {cat}"


# ===========================================================
# 29. Smoke dataset itself validates successfully
# ===========================================================


def test_smoke_dataset_validates():
    cases = validate_dataset(SMOKE_PATH)
    assert len(cases) == 16
    summary = dataset_summary(cases)
    assert summary["duplicate_count"] == 0


# ===========================================================
# 30. Existing telemetry CLI still works
# ===========================================================


def test_existing_telemetry_cli(tmp_path):
    # Create a minimal routing log
    log = tmp_path / "routing.jsonl"
    log.write_text(
        json.dumps({"route_source": "tool", "task_type": "math", "success": True}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.json"

    result = subprocess.run(
        [sys.executable, "-m", "benchmark", "aggregate", "--input", str(log), "--output", str(output)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0
    assert output.exists()


def test_telemetry_cli_backwards_compat(tmp_path):
    """The old --input --output interface without subcommand still works."""
    log = tmp_path / "routing.jsonl"
    log.write_text(
        json.dumps({"route_source": "local", "task_type": "factual", "success": True}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.json"

    result = subprocess.run(
        [sys.executable, "-m", "benchmark", "--input", str(log), "--output", str(output)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0
    assert output.exists()


# ===========================================================
# 31. All previous tests continue to pass (meta-test)
# ===========================================================


def test_validate_dataset_cli(tmp_path):
    """The validate-dataset CLI works on the smoke dataset."""
    result = subprocess.run(
        [sys.executable, "-m", "benchmark", "validate-dataset", "--input", str(SMOKE_PATH)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0
    assert "Dataset valid" in result.stdout
    assert "Total cases: 16" in result.stdout


def test_validate_dataset_cli_invalid(tmp_path):
    """The validate-dataset CLI exits non-zero on invalid input."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"task_id": "", "category": "x"}\n', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "benchmark", "validate-dataset", "--input", str(bad)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode != 0
