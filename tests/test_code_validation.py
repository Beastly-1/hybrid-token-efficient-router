"""Tests for benchmark/code_validation.py."""

from pathlib import Path

import pytest

from benchmark.code_validation import (
    CodeValidationResult,
    extract_python_code,
    expected_function_names,
    validate_code_answer,
    _MAX_SOURCE_CHARS,
    _MAX_AST_NODES,
)
from benchmark.dataset import BenchmarkCase, load_dataset
from benchmark.scorers import score_answer


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
# Candidate Extraction
# ===========================================================


class TestExtraction:
    def test_plain_python_source(self):
        """Plain Python source string."""
        result = extract_python_code("def add(a, b):\n    return a + b")
        assert result.valid is True
        assert result.source_code == "def add(a, b):\n    return a + b"

    def test_python_fenced_block(self):
        """Python fenced block."""
        code = "```python\ndef add(a, b):\n    return a + b\n```"
        result = extract_python_code(code)
        assert result.valid is True
        assert result.source_code == "def add(a, b):\n    return a + b"

    def test_py_fenced_block(self):
        """py fenced block."""
        code = "```py\ndef add(a, b):\n    return a + b\n```"
        result = extract_python_code(code)
        assert result.valid is True
        assert "def add" in result.source_code

    def test_generic_fenced_block(self):
        """Generic fenced block (no language)."""
        code = "```\ndef add(a, b):\n    return a + b\n```"
        result = extract_python_code(code)
        assert result.valid is True
        assert "def add" in result.source_code

    def test_explanation_plus_one_code_block(self):
        """Explanation text followed by exactly one code block."""
        code = "Here is the solution:\n\n```python\ndef add(a, b):\n    return a + b\n```\n"
        result = extract_python_code(code)
        assert result.valid is True
        assert result.source_code == "def add(a, b):\n    return a + b"

    def test_dict_with_code_field(self):
        """Dictionary with 'code' field."""
        result = extract_python_code({"code": "def add(a, b): return a + b"})
        assert result.valid is True
        assert "def add" in result.source_code

    def test_dict_with_answer_field(self):
        """Dictionary with 'answer' field."""
        result = extract_python_code({"answer": "def add(a, b): return a + b"})
        assert result.valid is True
        assert "def add" in result.source_code

    def test_code_field_preferred_over_answer(self):
        """'code' field preferred when both exist."""
        result = extract_python_code({
            "code": "def code_ver(): pass",
            "answer": "def answer_ver(): pass",
        })
        assert result.valid is True
        assert "code_ver" in result.source_code

    def test_empty_candidate_rejected(self):
        """Empty string rejected."""
        result = extract_python_code("")
        assert result.valid is False
        assert result.status == "invalid_answer"

    def test_whitespace_only_rejected(self):
        """Whitespace-only string rejected."""
        result = extract_python_code("   \n  \t  ")
        assert result.valid is False
        assert result.status == "invalid_answer"

    def test_none_rejected(self):
        """None rejected."""
        result = extract_python_code(None)
        assert result.valid is False
        assert result.status == "invalid_answer"

    def test_boolean_rejected(self):
        """Boolean rejected."""
        result = extract_python_code(True)
        assert result.valid is False
        assert result.status == "invalid_answer"

    def test_number_rejected(self):
        """Number rejected."""
        result = extract_python_code(42)
        assert result.valid is False
        assert result.status == "invalid_answer"

    def test_list_rejected(self):
        """List rejected."""
        result = extract_python_code(["def add(a, b): pass"])
        assert result.valid is False
        assert result.status == "invalid_answer"

    def test_dict_without_supported_field_rejected(self):
        """Dict without code/answer field rejected."""
        result = extract_python_code({"source": "def add(a, b): pass"})
        assert result.valid is False
        assert result.status == "invalid_answer"

    def test_multiple_code_blocks_rejected(self):
        """Multiple code blocks rejected."""
        code = "```python\ndef a(): pass\n```\n\n```python\ndef b(): pass\n```"
        result = extract_python_code(code)
        assert result.valid is False
        assert "multiple" in result.details.get("reason", "")

    def test_unterminated_code_fence_rejected(self):
        """Unterminated code fence rejected."""
        code = "```python\ndef add(a, b): return a + b"
        result = extract_python_code(code)
        assert result.valid is False
        assert "unterminated" in result.details.get("reason", "")

    def test_unsupported_fence_language_rejected(self):
        """Unsupported fence language rejected."""
        for lang in ("javascript", "java", "cpp", "bash"):
            code = f"```{lang}\nconsole.log('hi')\n```"
            result = extract_python_code(code)
            assert result.valid is False
            assert "unsupported fence language" in result.details.get("reason", "")

    def test_preserves_internal_indentation(self):
        """Internal indentation is preserved."""
        src = "def f():\n    if True:\n        return 1"
        code = f"```python\n{src}\n```"
        result = extract_python_code(code)
        assert result.valid is True
        assert "    if True:" in result.source_code
        assert "        return 1" in result.source_code


# ===========================================================
# Syntax Validation
# ===========================================================


class TestSyntax:
    def test_valid_function_parses(self):
        """Valid function parses successfully."""
        result = validate_code_answer(_case(), "def add(a, b):\n    return a + b")
        assert result.valid is True
        assert result.status == "valid"

    def test_syntax_error_reports_line_and_column(self):
        """Syntax error reports line and column."""
        result = validate_code_answer(_case(), "def add(a, b)\n    return a + b")
        assert result.valid is False
        assert result.status == "syntax_error"
        assert "syntax_line" in result.details
        assert "syntax_column" in result.details
        assert "syntax_message" in result.details

    def test_code_never_executed(self):
        """Code is never executed during validation (side-effect code)."""
        # This code would raise if executed
        dangerous = "def add(a, b):\n    return a + b\n\nraise RuntimeError('executed!')"
        # Should parse fine (valid syntax) but the raise is not a safety issue
        # since we don't block raise statements
        result = validate_code_answer(_case(), dangerous)
        # It parses and passes safety (raise is allowed)
        assert result.valid is True


# ===========================================================
# Required Functions
# ===========================================================


class TestRequiredFunctions:
    def test_required_top_level_function_found(self):
        """Required top-level function found."""
        result = validate_code_answer(_case(), "def add(a, b):\n    return a + b")
        assert result.valid is True
        assert "add" in result.details.get("defined_functions", [])

    def test_missing_required_function_rejected(self):
        """Missing required function rejected."""
        result = validate_code_answer(_case(), "def subtract(a, b):\n    return a - b")
        assert result.valid is False
        assert result.status == "invalid_answer"
        assert "add" in result.details.get("missing_functions", [])

    def test_multiple_required_functions(self):
        """Multiple required functions supported."""
        case = _case(tests=[
            {"call": "add(1, 2)", "expected": 3},
            {"call": "multiply(2, 3)", "expected": 6},
        ])
        code = "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b"
        result = validate_code_answer(case, code)
        assert result.valid is True

    def test_nested_function_does_not_satisfy(self):
        """Nested function does not satisfy requirement."""
        code = "def wrapper():\n    def add(a, b):\n        return a + b\n    return add"
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert "add" in result.details.get("missing_functions", [])

    def test_class_method_does_not_satisfy(self):
        """Class method does not satisfy requirement."""
        code = "class Math:\n    def add(self, a, b):\n        return a + b"
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert "add" in result.details.get("missing_functions", [])

    def test_additional_functions_permitted(self):
        """Additional functions are permitted."""
        code = "def helper(x):\n    return x\n\ndef add(a, b):\n    return helper(a) + helper(b)"
        result = validate_code_answer(_case(), code)
        assert result.valid is True

    def test_invalid_test_call_syntax_rejected(self):
        """Invalid test-call syntax is ignored (no function extracted)."""
        names = expected_function_names(_case(tests=[{"call": "???invalid", "expected": 1}]))
        assert len(names) == 0

    def test_attribute_based_test_call_rejected(self):
        """Attribute-based test call rejected (not a simple name)."""
        names = expected_function_names(_case(tests=[{"call": "solution.add(2, 3)", "expected": 5}]))
        assert len(names) == 0

    def test_indexed_call_target_rejected(self):
        """Indexed-call target rejected."""
        names = expected_function_names(_case(tests=[{"call": "values[0]()", "expected": 1}]))
        assert len(names) == 0

    def test_expected_function_names_basic(self):
        """Basic function name extraction."""
        case = _case(tests=[
            {"call": "add(2, 3)", "expected": 5},
            {"call": "add(-1, 4)", "expected": 3},
        ])
        names = expected_function_names(case)
        assert names == {"add"}


# ===========================================================
# Safety
# ===========================================================


class TestSafety:
    def test_import_rejected(self):
        """import statement rejected."""
        code = "import os\ndef add(a, b):\n    return a + b"
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert result.status == "unsafe_code"
        assert result.details.get("prohibited_node") == "Import"

    def test_from_import_rejected(self):
        """from ... import rejected."""
        code = "from os import path\ndef add(a, b):\n    return a + b"
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert result.status == "unsafe_code"
        assert result.details.get("prohibited_node") == "ImportFrom"

    def test_eval_rejected(self):
        """eval() call rejected."""
        code = "def add(a, b):\n    return eval(f'{a}+{b}')"
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert result.status == "unsafe_code"
        assert result.details.get("prohibited_call") == "eval"

    def test_exec_rejected(self):
        """exec() call rejected."""
        code = "def add(a, b):\n    exec('pass')\n    return a + b"
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert result.status == "unsafe_code"
        assert result.details.get("prohibited_call") == "exec"

    def test_open_rejected(self):
        """open() call rejected."""
        code = "def add(a, b):\n    open('file.txt')\n    return a + b"
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert result.status == "unsafe_code"
        assert result.details.get("prohibited_call") == "open"

    def test_dunder_import_rejected(self):
        """__import__() call rejected."""
        code = "def add(a, b):\n    __import__('os')\n    return a + b"
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert result.status == "unsafe_code"

    def test_system_attribute_call_rejected(self):
        """os.system-style attribute call rejected even without import."""
        code = "def add(a, b):\n    os.system('ls')\n    return a + b"
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert result.status == "unsafe_code"
        assert result.details.get("prohibited_call") == "system"

    def test_dunder_attribute_access_rejected(self):
        """Dunder attribute access rejected."""
        code = "def add(a, b):\n    x = a.__class__\n    return a + b"
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert result.status == "unsafe_code"
        assert result.details.get("prohibited_attribute") == "__class__"

    def test_globals_call_rejected(self):
        """globals() call rejected."""
        code = "def add(a, b):\n    g = globals()\n    return a + b"
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert result.status == "unsafe_code"
        assert result.details.get("prohibited_call") == "globals"

    def test_getattr_rejected(self):
        """getattr() call rejected."""
        code = "def add(a, b):\n    return getattr(a, '__add__')(b)"
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert result.status == "unsafe_code"

    def test_global_statement_rejected(self):
        """global statement rejected."""
        code = "x = 0\ndef add(a, b):\n    global x\n    return a + b"
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert result.status == "unsafe_code"
        assert result.details.get("prohibited_node") == "Global"

    def test_nonlocal_statement_rejected(self):
        """nonlocal statement rejected."""
        code = "def outer():\n    x = 0\n    def add(a, b):\n        nonlocal x\n        return a + b"
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert result.status == "unsafe_code"
        assert result.details.get("prohibited_node") == "Nonlocal"

    def test_normal_loop_accepted(self):
        """Normal loop accepted."""
        code = "def add(a, b):\n    total = 0\n    for i in range(a):\n        total += 1\n    return total + b"
        result = validate_code_answer(_case(), code)
        assert result.valid is True

    def test_recursion_accepted(self):
        """Recursion accepted."""
        code = "def add(a, b):\n    if b == 0:\n        return a\n    return add(a + 1, b - 1)"
        result = validate_code_answer(_case(), code)
        assert result.valid is True

    def test_list_comprehension_accepted(self):
        """List comprehension accepted."""
        code = "def add(a, b):\n    return sum([x for x in [a, b]])"
        result = validate_code_answer(_case(), code)
        assert result.valid is True

    def test_safe_builtins_accepted(self):
        """Safe built-ins accepted."""
        code = "def add(a, b):\n    nums = [a, b]\n    return sum(sorted(nums))"
        result = validate_code_answer(_case(), code)
        assert result.valid is True


# ===========================================================
# Size Limits
# ===========================================================


class TestLimits:
    def test_excess_source_characters_rejected(self):
        """Excess source characters rejected."""
        code = "def add(a, b):\n    return a + b\n" + "# padding\n" * 10000
        assert len(code) > _MAX_SOURCE_CHARS
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert result.status == "too_large"
        assert result.details.get("size_limit") == "max_source_chars"

    def test_excess_ast_nodes_rejected(self):
        """Excess AST nodes rejected."""
        # Generate code with many AST nodes but within char limit.
        # Each 'a=1;' is ~4 chars but produces multiple AST nodes.
        # Use semicolons to pack many statements in fewer chars.
        lines = ["def add(a, b):\n    return a + b"]
        # Each line "a=1" is 4 chars + newline, produces ~3 AST nodes (Assign, Name, Constant)
        needed = _MAX_AST_NODES // 3 + 1
        for i in range(needed):
            lines.append(f"a={i}")
        code = "\n".join(lines)
        # Ensure we're within char limit but over node limit
        assert len(code) <= _MAX_SOURCE_CHARS
        result = validate_code_answer(_case(), code)
        assert result.valid is False
        assert result.status == "too_large"
        assert result.details.get("size_limit") == "max_ast_nodes"


# ===========================================================
# Integration
# ===========================================================


class TestIntegration:
    def test_smoke_case_001_valid_candidate(self):
        """code_generation_001 passes static validation with valid code."""
        cases = load_dataset(SMOKE_PATH)
        case = next(c for c in cases if c.task_id == "code_generation_001")
        code = "def add(a, b):\n    return a + b"
        result = validate_code_answer(case, code)
        assert result.valid is True
        assert result.status == "valid"
        assert "add" in result.details.get("required_functions", [])

    def test_smoke_case_002_valid_candidate(self):
        """code_generation_002 passes static validation with valid code."""
        cases = load_dataset(SMOKE_PATH)
        case = next(c for c in cases if c.task_id == "code_generation_002")
        code = (
            "def is_palindrome(s):\n"
            "    cleaned = s.replace(' ', '').lower()\n"
            "    return cleaned == cleaned[::-1]"
        )
        result = validate_code_answer(case, code)
        assert result.valid is True
        assert result.status == "valid"
        assert "is_palindrome" in result.details.get("required_functions", [])

    def test_smoke_case_001_invalid_code_rejected(self):
        """Invalid code for code_generation_001 is rejected."""
        cases = load_dataset(SMOKE_PATH)
        case = next(c for c in cases if c.task_id == "code_generation_001")
        # Missing the required function
        result = validate_code_answer(case, "def subtract(a, b): return a - b")
        assert result.valid is False

    def test_smoke_case_002_invalid_code_rejected(self):
        """Invalid code for code_generation_002 is rejected."""
        cases = load_dataset(SMOKE_PATH)
        case = next(c for c in cases if c.task_id == "code_generation_002")
        # Syntax error
        result = validate_code_answer(case, "def is_palindrome(s)\n    return True")
        assert result.valid is False

    def test_score_answer_now_supported(self):
        """score_answer now returns scored for code_tests."""
        case = _case()
        result = score_answer(case, "def add(a, b): return a + b")
        assert result.status == "scored"
        assert result.score == 1.0

    def test_existing_scorers_unchanged(self):
        """Existing summarization and NER scorers remain unchanged."""
        # Summarization
        from benchmark.dataset import BenchmarkCase as BC
        sum_case = BC(
            task_id="s1", category="summarization", difficulty="easy",
            prompt="Summarize", scoring_method="summarization",
            reference="The cat sat on the mat.",
            constraints={"max_words": 10},
        )
        result = score_answer(sum_case, "The cat sat on the mat.")
        assert result.status == "scored"
        assert result.score is not None

        # NER
        ner_case = BC(
            task_id="n1", category="ner", difficulty="easy",
            prompt="Extract", scoring_method="ner",
            reference=[{"text": "Alice", "type": "PERSON"}],
        )
        result = score_answer(ner_case, [{"text": "Alice", "type": "PERSON"}])
        assert result.status == "scored"
        assert result.score == 1.0

    def test_no_fireworks_call(self):
        """No Fireworks call is made."""
        import benchmark.code_validation as mod
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "fireworks" not in source.lower()
        assert "requests" not in source
        assert "httpx" not in source

    def test_no_local_model_loaded(self):
        """Local model is not loaded."""
        import benchmark.code_validation as mod
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "openvino" not in source.lower()
        assert "LocalModel" not in source

    def test_no_code_executed(self):
        """No candidate code is executed (no eval/exec in module)."""
        import benchmark.code_validation as mod
        source = Path(mod.__file__).read_text(encoding="utf-8")
        # The module should not use eval() or exec() itself
        # (ast.parse is fine, it's not execution)
        assert "eval(" not in source
        assert "exec(" not in source
