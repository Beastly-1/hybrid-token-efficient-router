"""Tests for benchmark/code_execution.py and code_tests scorer integration."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark.code_execution import (
    CodeExecutionResult,
    ParsedCall,
    execute_code_tests,
    parse_all_test_calls,
    parse_test_call,
)
from benchmark.code_validation import validate_code_answer
from benchmark.dataset import BenchmarkCase, load_dataset
from benchmark.scorers import ScoreResult, score_answer


SMOKE_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "smoke" / "tasks.jsonl"


def _case(**overrides) -> BenchmarkCase:
    base = {
        "task_id": "t1",
        "category": "code_generation",
        "difficulty": "easy",
        "prompt": "Write a function",
        "scoring_method": "code_tests",
        "tests": [{"call": "add(2, 3)", "expected": 5}],
    }
    base.update(overrides)
    return BenchmarkCase(**base)


# ===========================================================
# Static Integration (1-3)
# ===========================================================


class TestStaticIntegration:
    def test_valid_code_passes_static_validation(self):
        """1. Valid code passes existing static validation."""
        case = _case()
        result = validate_code_answer(case, "def add(a, b):\n    return a + b")
        assert result.valid is True

    def test_unsafe_code_never_executed(self):
        """2. Invalid or unsafe code is never executed."""
        case = _case()
        # Code with import should fail statically, never reach execution
        result = score_answer(case, "import os\ndef add(a, b):\n    return a + b")
        assert result.status == "invalid_answer"
        assert "unsafe" in result.details.get("validation_status", "") or "prohibited" in result.details.get("reason", "")

    def test_missing_function_returns_invalid(self):
        """3. Missing required function returns invalid_answer."""
        case = _case()
        result = score_answer(case, "def subtract(a, b):\n    return a - b")
        assert result.status == "invalid_answer"
        assert "missing" in result.details.get("reason", "")


# ===========================================================
# Call Parsing (4-10)
# ===========================================================


class TestCallParsing:
    def test_positional_literal_arguments(self):
        """4. Positional literal arguments."""
        result = parse_test_call("add(2, 3)")
        assert result is not None
        assert result.function_name == "add"
        assert result.args == [2, 3]
        assert result.kwargs == {}

    def test_list_dict_tuple_literal_arguments(self):
        """5. List/dict/tuple literal arguments."""
        result = parse_test_call("find_max([1, 5, 2])")
        assert result is not None
        assert result.args == [[1, 5, 2]]

        result2 = parse_test_call("process({'key': 'val'})")
        assert result2 is not None
        assert result2.args == [{"key": "val"}]

        result3 = parse_test_call("unpack((1, 2, 3))")
        assert result3 is not None
        assert result3.args == [(1, 2, 3)]

    def test_keyword_arguments(self):
        """6. Keyword arguments."""
        result = parse_test_call('greet(name="Maya")')
        assert result is not None
        assert result.kwargs == {"name": "Maya"}

    def test_attribute_call_rejected(self):
        """7. Attribute call rejected."""
        result = parse_test_call("obj.method(1)")
        assert result is None

    def test_starred_argument_rejected(self):
        """8. Starred argument rejected."""
        result = parse_test_call("add(*[1, 2])")
        assert result is None

    def test_non_literal_argument_rejected(self):
        """9. Non-literal argument rejected (variable name)."""
        result = parse_test_call("add(x, y)")
        assert result is None

    def test_malformed_call_rejected(self):
        """10. Malformed call rejected."""
        result = parse_test_call("not a call at all ???")
        assert result is None

        # Expression that's not a call
        result2 = parse_test_call("42")
        assert result2 is None

    def test_kwargs_expansion_rejected(self):
        """Reject **kwargs expansion."""
        result = parse_test_call("add(**{'a': 1})")
        assert result is None


# ===========================================================
# Execution (11-20)
# ===========================================================


class TestExecution:
    def test_all_tests_pass(self):
        """11. All tests pass."""
        case = _case(tests=[
            {"call": "add(2, 3)", "expected": 5},
            {"call": "add(0, 0)", "expected": 0},
        ])
        result = execute_code_tests(case, "def add(a, b):\n    return a + b")
        assert result.status == "completed"
        assert result.passed == 2
        assert result.failed == 0

    def test_some_tests_fail(self):
        """12. Some tests fail."""
        case = _case(tests=[
            {"call": "add(2, 3)", "expected": 5},
            {"call": "add(1, 1)", "expected": 99},  # wrong expected
        ])
        result = execute_code_tests(case, "def add(a, b):\n    return a + b")
        assert result.status == "completed"
        assert result.passed == 1
        assert result.failed == 1

    def test_all_tests_fail(self):
        """13. All tests fail."""
        case = _case(tests=[
            {"call": "add(2, 3)", "expected": 99},
            {"call": "add(1, 1)", "expected": 99},
        ])
        result = execute_code_tests(case, "def add(a, b):\n    return a + b")
        assert result.status == "completed"
        assert result.passed == 0
        assert result.failed == 2

    def test_candidate_exception_fails_test(self):
        """14. Candidate exception fails the relevant test."""
        case = _case(tests=[
            {"call": "add(2, 3)", "expected": 5},
        ])
        code = "def add(a, b):\n    raise ValueError('oops')"
        result = execute_code_tests(case, code)
        assert result.status == "completed"
        assert result.passed == 0
        assert result.failed == 1
        test_results = result.details.get("test_results", [])
        assert test_results[0]["reason"] == "exception"

    def test_infinite_loop_timeout(self):
        """15. Infinite loop is terminated by timeout."""
        case = _case(tests=[{"call": "add(1, 1)", "expected": 2}])
        code = "def add(a, b):\n    while True:\n        pass"
        result = execute_code_tests(case, code, timeout=1.0)
        assert result.timed_out is True
        assert result.status == "timeout"

    def test_syntax_invalid_not_executed(self):
        """16. Syntax-invalid code is not executed."""
        case = _case()
        result = execute_code_tests(case, "def add(a, b)\n    return a + b")
        assert result.status == "invalid_test"
        assert result.details.get("validation_status") == "syntax_error"

    def test_candidate_stdout_no_effect(self):
        """17. Candidate stdout does not affect scoring."""
        case = _case(tests=[{"call": "add(2, 3)", "expected": 5}])
        code = "def add(a, b):\n    print('HELLO WORLD' * 1000)\n    return a + b"
        result = execute_code_tests(case, code)
        assert result.status == "completed"
        assert result.passed == 1

    def test_temp_files_cleaned_up(self):
        """18. Temporary execution files are cleaned up."""
        case = _case(tests=[{"call": "add(1, 1)", "expected": 2}])
        code = "def add(a, b):\n    return a + b"

        # Track temp dirs created
        original_mkdtemp = tempfile.mkdtemp
        created_dirs = []

        def tracking_mkdtemp(*args, **kwargs):
            d = original_mkdtemp(*args, **kwargs)
            created_dirs.append(d)
            return d

        with patch("benchmark.code_execution.tempfile.mkdtemp", side_effect=tracking_mkdtemp):
            execute_code_tests(case, code)

        # All created dirs should be cleaned up
        for d in created_dirs:
            assert not Path(d).exists()

    def test_runs_in_child_process(self):
        """19. Candidate code runs in a child process, not the parent."""
        case = _case(tests=[{"call": "add(1, 1)", "expected": 2}])
        # If this ran in parent, it would set a module-level variable
        code = "import sys\n"  # This will fail static validation (import)
        # Use safe code that proves isolation differently
        code_safe = "def add(a, b):\n    return a + b"
        result = execute_code_tests(case, code_safe)
        assert result.status == "completed"
        # Verify no 'candidate' module in parent
        assert "candidate" not in sys.modules

    def test_shell_false_used(self):
        """20. shell=False is used (verified by checking subprocess.run call)."""
        case = _case(tests=[{"call": "add(1, 1)", "expected": 2}])
        code = "def add(a, b):\n    return a + b"

        original_run = subprocess.run
        shell_values = []

        def tracking_run(*args, **kwargs):
            shell_values.append(kwargs.get("shell", "NOT_SET"))
            return original_run(*args, **kwargs)

        with patch("benchmark.code_execution.subprocess.run", side_effect=tracking_run):
            execute_code_tests(case, code)

        assert all(v is False for v in shell_values)


# ===========================================================
# Comparison (21-27)
# ===========================================================


class TestComparison:
    def test_integer_result(self):
        """21. Integer result."""
        case = _case(tests=[{"call": "add(2, 3)", "expected": 5}])
        result = execute_code_tests(case, "def add(a, b):\n    return a + b")
        assert result.passed == 1

    def test_string_result(self):
        """22. String result."""
        case = _case(tests=[{"call": "greet('hi')", "expected": "hello hi"}])
        code = "def greet(name):\n    return 'hello ' + name"
        result = execute_code_tests(case, code)
        assert result.passed == 1

    def test_list_result(self):
        """23. List result."""
        case = _case(tests=[{"call": "make_list(3)", "expected": [0, 1, 2]}])
        code = "def make_list(n):\n    return list(range(n))"
        result = execute_code_tests(case, code)
        assert result.passed == 1

    def test_dict_result(self):
        """24. Dictionary result."""
        case = _case(tests=[{"call": "make_dict('a', 1)", "expected": {"a": 1}}])
        code = "def make_dict(k, v):\n    return {k: v}"
        result = execute_code_tests(case, code)
        assert result.passed == 1

    def test_bool_true_not_equal_int_1(self):
        """25. Boolean True is not accepted as integer 1."""
        case = _case(tests=[{"call": "get_val()", "expected": 1}])
        code = "def get_val():\n    return True"
        result = execute_code_tests(case, code)
        assert result.passed == 0
        assert result.failed == 1

    def test_bool_false_not_equal_int_0(self):
        """26. Boolean False is not accepted as integer 0."""
        case = _case(tests=[{"call": "get_val()", "expected": 0}])
        code = "def get_val():\n    return False"
        result = execute_code_tests(case, code)
        assert result.passed == 0
        assert result.failed == 1

    def test_non_serialisable_result_fails(self):
        """27. Non-serialisable result fails cleanly."""
        case = _case(tests=[{"call": "get_val()", "expected": "something"}])
        # Return a lambda (not serialisable, but comparison will just fail)
        code = "def get_val():\n    return lambda: None"
        result = execute_code_tests(case, code)
        assert result.passed == 0
        assert result.failed == 1


# ===========================================================
# Scorer Integration (28-36)
# ===========================================================


class TestScorerIntegration:
    def test_perfect_score(self):
        """28. Perfect code_tests result gives score 1.0 and correct=True."""
        case = _case(tests=[
            {"call": "add(2, 3)", "expected": 5},
            {"call": "add(0, 0)", "expected": 0},
        ])
        result = score_answer(case, "def add(a, b):\n    return a + b")
        assert result.status == "scored"
        assert result.score == 1.0
        assert result.correct is True

    def test_partial_pass_fractional_score(self):
        """29. Partial pass gives fractional score."""
        case = _case(tests=[
            {"call": "add(2, 3)", "expected": 5},
            {"call": "add(1, 1)", "expected": 99},  # will fail
            {"call": "add(0, 0)", "expected": 0},
        ])
        result = score_answer(case, "def add(a, b):\n    return a + b")
        assert result.status == "scored"
        assert result.score == pytest.approx(2.0 / 3.0)
        assert result.correct is False

    def test_timeout_gives_zero(self):
        """30. Timeout gives score 0.0 and correct=False."""
        case = _case(tests=[{"call": "add(1, 1)", "expected": 2}])
        code = "def add(a, b):\n    while True:\n        pass"
        # Use very short timeout
        with patch("benchmark.code_execution._DEFAULT_TIMEOUT_SECONDS", 0.5):
            result = score_answer(case, code)
        assert result.status == "scored"
        assert result.score == 0.0
        assert result.correct is False
        assert result.details.get("timed_out") is True

    def test_static_rejection_invalid_answer(self):
        """31. Static rejection gives invalid_answer."""
        case = _case()
        result = score_answer(case, "import os\ndef add(a, b): return a + b")
        assert result.status == "invalid_answer"
        assert result.score is None
        assert result.correct is None

    def test_smoke_cases_scoreable(self):
        """32. Both code_tests smoke cases are scoreable."""
        cases = load_dataset(SMOKE_PATH)
        code_cases = [c for c in cases if c.scoring_method == "code_tests"]
        assert len(code_cases) == 2

        # code_generation_001: add
        add_case = next(c for c in code_cases if c.task_id == "code_generation_001")
        result = score_answer(add_case, "def add(a, b):\n    return a + b")
        assert result.status == "scored"
        assert result.score == 1.0
        assert result.correct is True

        # code_generation_002: is_palindrome
        pal_case = next(c for c in code_cases if c.task_id == "code_generation_002")
        code = (
            "def is_palindrome(s):\n"
            "    cleaned = s.replace(' ', '').lower()\n"
            "    return cleaned == cleaned[::-1]"
        )
        result = score_answer(pal_case, code)
        assert result.status == "scored"
        assert result.score == 1.0
        assert result.correct is True

    def test_all_16_smoke_cases_supported(self):
        """33. All 16 smoke cases now use supported scoring methods."""
        cases = load_dataset(SMOKE_PATH)
        assert len(cases) == 16
        for case in cases:
            # Every method should produce a non-unsupported result with valid input
            if case.scoring_method == "code_tests":
                result = score_answer(case, "def add(a, b): return a + b\ndef is_palindrome(s): return True")
                # Should be scored (not unsupported)
                assert result.status in ("scored", "invalid_answer"), f"{case.task_id}: {result.status}"
            else:
                result = score_answer(case, "test")
                assert result.status != "unsupported", f"{case.task_id}: {result.status}"

    def test_existing_scorers_unchanged(self):
        """34. Existing scorers remain unchanged."""
        # Exact
        case = BenchmarkCase(task_id="e1", category="factual", difficulty="easy",
                             prompt="test", scoring_method="exact", reference="Au")
        assert score_answer(case, "Au").score == 1.0

        # Numeric
        case = BenchmarkCase(task_id="n1", category="math", difficulty="easy",
                             prompt="test", scoring_method="numeric", reference=42)
        assert score_answer(case, "42").score == 1.0

        # NER
        case = BenchmarkCase(task_id="r1", category="ner", difficulty="easy",
                             prompt="test", scoring_method="ner",
                             reference=[{"text": "Alice", "type": "PERSON"}])
        assert score_answer(case, [{"text": "Alice", "type": "PERSON"}]).score == 1.0

    def test_no_fireworks_call(self):
        """35. No Fireworks call is made."""
        import benchmark.code_execution as mod
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "fireworks" not in source.lower()
        assert "requests" not in source
        assert "httpx" not in source

    def test_no_local_model_loaded(self):
        """36. Local model is not loaded."""
        import benchmark.code_execution as mod
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "openvino" not in source.lower()
        assert "LocalModel" not in source
