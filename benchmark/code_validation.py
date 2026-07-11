"""Code answer extraction and static validation for code_tests benchmark cases.

This module provides deterministic helpers that:
1. Extract Python source code from a candidate answer
2. Parse it using ast
3. Validate structural and safety requirements
4. Return a typed validation result

IMPORTANT: AST validation alone is NOT a secure sandbox. Subprocess isolation
and runtime limits are required before executing candidate code. This module
does NOT execute candidate code.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from benchmark.dataset import BenchmarkCase

# ===================================================================
# Constants
# ===================================================================

_MAX_SOURCE_CHARS = 20_000
_MAX_AST_NODES = 5_000

# Dangerous built-in/helper calls
_DANGEROUS_CALLS = frozenset({
    "eval", "exec", "compile", "open", "input", "__import__",
    "breakpoint", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "help", "dir",
})

# OS/filesystem/process calls.
# Note: 'replace' is excluded because str.replace() is ubiquitous in normal
# Python. The import prohibition already blocks os.replace() usage.
_DANGEROUS_SYSTEM_CALLS = frozenset({
    "system", "popen", "spawn", "fork",
    "remove", "unlink", "rmdir", "mkdir", "makedirs",
    "rename", "chdir",
})

_ALL_DANGEROUS_CALLS = _DANGEROUS_CALLS | _DANGEROUS_SYSTEM_CALLS

# Dangerous dunder attributes
_DUNDER_PATTERN = re.compile(r"^__.*__$")

# Fenced code block pattern
_FENCE_PATTERN = re.compile(
    r"```(\w*)\s*\n(.*?)```", re.DOTALL
)

# Supported fence language identifiers
_SUPPORTED_FENCE_LANGS = frozenset({"python", "py", ""})


# ===================================================================
# Result Type
# ===================================================================


@dataclass
class CodeValidationResult:
    """Typed result from code validation.

    Status values:
    - "valid": code passed all static checks
    - "invalid_answer": candidate could not be interpreted as code
    - "syntax_error": code has a syntax error
    - "unsafe_code": code uses prohibited constructs
    - "too_large": code exceeds size limits
    """

    valid: bool
    status: str
    source_code: Optional[str]
    details: dict = field(default_factory=dict)


# ===================================================================
# Code Extraction
# ===================================================================


def extract_python_code(candidate: Any) -> CodeValidationResult:
    """Extract Python source code from a candidate answer.

    Supported formats:
    1. Plain Python source string
    2. Python/py/generic fenced block
    3. String with explanation + exactly one code block
    4. Dict with "code" or "answer" key ("code" preferred)

    Rejects: None, booleans, numbers, lists, dicts without supported field,
    empty strings, multiple fenced blocks, unterminated fences, unsupported
    fence languages.
    """
    # Reject unsupported types
    if candidate is None:
        return CodeValidationResult(False, "invalid_answer", None,
                                    {"reason": "candidate is None"})
    if isinstance(candidate, (bool, int, float)):
        return CodeValidationResult(False, "invalid_answer", None,
                                    {"reason": f"unsupported type: {type(candidate).__name__}"})
    if isinstance(candidate, list):
        return CodeValidationResult(False, "invalid_answer", None,
                                    {"reason": "unsupported type: list"})

    # Dict: extract from "code" or "answer" key
    if isinstance(candidate, dict):
        source = candidate.get("code")
        if isinstance(source, str) and source.strip():
            return _extract_from_string(source)
        source = candidate.get("answer")
        if isinstance(source, str) and source.strip():
            return _extract_from_string(source)
        return CodeValidationResult(False, "invalid_answer", None,
                                    {"reason": "dict has no non-empty 'code' or 'answer' field"})

    if not isinstance(candidate, str):
        return CodeValidationResult(False, "invalid_answer", None,
                                    {"reason": f"unsupported type: {type(candidate).__name__}"})

    if not candidate.strip():
        return CodeValidationResult(False, "invalid_answer", None,
                                    {"reason": "empty string"})

    return _extract_from_string(candidate)


def _extract_from_string(text: str) -> CodeValidationResult:
    """Extract code from a string, handling fences."""
    stripped = text.strip()

    # Check for unterminated fence
    open_fences = stripped.count("```")
    if open_fences % 2 != 0:
        return CodeValidationResult(False, "invalid_answer", None,
                                    {"reason": "unterminated code fence"})

    # Find all fenced blocks
    matches = list(_FENCE_PATTERN.finditer(stripped))

    if len(matches) > 1:
        return CodeValidationResult(False, "invalid_answer", None,
                                    {"reason": "multiple code blocks found"})

    if len(matches) == 1:
        lang = matches[0].group(1).lower()
        if lang not in _SUPPORTED_FENCE_LANGS:
            return CodeValidationResult(False, "invalid_answer", None,
                                        {"reason": f"unsupported fence language: {lang}"})
        source = matches[0].group(2).strip()
        if not source:
            return CodeValidationResult(False, "invalid_answer", None,
                                        {"reason": "empty code block"})
        return CodeValidationResult(True, "valid", source)

    # No fences: treat entire string as Python source
    return CodeValidationResult(True, "valid", stripped)


# ===================================================================
# Size Limits
# ===================================================================


def _check_size_limits(source: str, node_count: int) -> Optional[CodeValidationResult]:
    """Return a failure result if size limits are exceeded, else None."""
    if len(source) > _MAX_SOURCE_CHARS:
        return CodeValidationResult(False, "too_large", source, {
            "reason": "source exceeds character limit",
            "size_limit": "max_source_chars",
            "limit": _MAX_SOURCE_CHARS,
            "actual": len(source),
        })
    if node_count > _MAX_AST_NODES:
        return CodeValidationResult(False, "too_large", source, {
            "reason": "AST exceeds node limit",
            "size_limit": "max_ast_nodes",
            "limit": _MAX_AST_NODES,
            "actual": node_count,
        })
    return None


def _count_ast_nodes(tree: ast.AST) -> int:
    """Count all nodes in an AST."""
    return sum(1 for _ in ast.walk(tree))


# ===================================================================
# Syntax Validation
# ===================================================================


def _parse_source(source: str) -> tuple[Optional[ast.Module], Optional[CodeValidationResult]]:
    """Parse source code. Returns (tree, None) on success or (None, error) on failure."""
    try:
        tree = ast.parse(source, mode="exec")
        return tree, None
    except SyntaxError as e:
        return None, CodeValidationResult(False, "syntax_error", source, {
            "reason": "syntax error",
            "syntax_line": e.lineno,
            "syntax_column": e.offset,
            "syntax_message": e.msg if e.msg else str(e),
        })


# ===================================================================
# Static Safety Policy
# ===================================================================


def _check_safety(tree: ast.Module, source: str) -> Optional[CodeValidationResult]:
    """Check for prohibited AST constructs. Returns failure or None."""
    for node in ast.walk(tree):
        # Reject import/importfrom/global/nonlocal statements
        if isinstance(node, ast.Import):
            return CodeValidationResult(False, "unsafe_code", source, {
                "reason": "prohibited construct",
                "prohibited_node": "Import",
            })
        if isinstance(node, ast.ImportFrom):
            return CodeValidationResult(False, "unsafe_code", source, {
                "reason": "prohibited construct",
                "prohibited_node": "ImportFrom",
            })
        if isinstance(node, ast.Global):
            return CodeValidationResult(False, "unsafe_code", source, {
                "reason": "prohibited construct",
                "prohibited_node": "Global",
            })
        if isinstance(node, ast.Nonlocal):
            return CodeValidationResult(False, "unsafe_code", source, {
                "reason": "prohibited construct",
                "prohibited_node": "Nonlocal",
            })

        # Reject dangerous function calls
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name and name in _ALL_DANGEROUS_CALLS:
                return CodeValidationResult(False, "unsafe_code", source, {
                    "reason": "prohibited call",
                    "prohibited_call": name,
                })

        # Reject dunder attribute access
        if isinstance(node, ast.Attribute):
            if _DUNDER_PATTERN.match(node.attr):
                return CodeValidationResult(False, "unsafe_code", source, {
                    "reason": "prohibited dunder attribute access",
                    "prohibited_attribute": node.attr,
                })

    return None


# ===================================================================
# Expected Function Names
# ===================================================================


def expected_function_names(case: BenchmarkCase) -> set[str]:
    """Extract required function names from case.tests[*].call fields.

    Supports only direct function calls (simple identifier). Rejects:
    - Attribute-based calls (obj.method())
    - Indexed targets (values[0]())
    - Non-call expressions

    Returns a set of function name strings.
    """
    names: set[str] = set()
    if not case.tests:
        return names

    for test in case.tests:
        if not isinstance(test, dict):
            continue
        call_str = test.get("call")
        if not isinstance(call_str, str) or not call_str.strip():
            continue
        name = _extract_call_name(call_str)
        if name is not None:
            names.add(name)

    return names


def _extract_call_name(call_str: str) -> Optional[str]:
    """Extract the root function name from a call expression string.

    Returns the name for simple calls like 'add(2, 3)'.
    Returns None for attribute calls, indexed targets, or non-calls.
    """
    try:
        tree = ast.parse(call_str.strip(), mode="eval")
    except SyntaxError:
        return None

    expr = tree.body
    if not isinstance(expr, ast.Call):
        return None
    func = expr.func
    if not isinstance(func, ast.Name):
        return None
    return func.id


def _get_top_level_functions(tree: ast.Module) -> set[str]:
    """Get names of top-level function definitions (not nested, not methods)."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


# ===================================================================
# Public Validator
# ===================================================================


def validate_code_answer(case: BenchmarkCase, candidate: Any) -> CodeValidationResult:
    """Validate a candidate code answer against a code_tests benchmark case.

    Flow:
    1. Confirm scoring_method == "code_tests"
    2. Extract source
    3. Parse source (syntax check)
    4. Apply size limits
    5. Apply static safety checks
    6. Extract required function names from case.tests
    7. Confirm all required functions are defined at top level

    Does NOT execute candidate code.

    AsyncFunctionDef: accepted as a valid top-level function definition.
    """
    if case.scoring_method != "code_tests":
        return CodeValidationResult(False, "invalid_answer", None, {
            "reason": f"scoring_method is '{case.scoring_method}', expected 'code_tests'",
        })

    # Step 1: Extract
    extraction = extract_python_code(candidate)
    if not extraction.valid:
        return extraction

    source = extraction.source_code
    assert source is not None  # guaranteed by valid extraction

    # Step 2: Parse
    tree, parse_error = _parse_source(source)
    if parse_error is not None:
        return parse_error

    assert tree is not None

    # Step 3: Size limits
    node_count = _count_ast_nodes(tree)
    size_error = _check_size_limits(source, node_count)
    if size_error is not None:
        return size_error

    # Step 4: Safety
    safety_error = _check_safety(tree, source)
    if safety_error is not None:
        return safety_error

    # Step 5: Required functions
    required = expected_function_names(case)
    defined = _get_top_level_functions(tree)
    missing = required - defined

    if missing:
        return CodeValidationResult(False, "invalid_answer", source, {
            "reason": "missing required functions",
            "missing_functions": sorted(missing),
            "required_functions": sorted(required),
            "defined_functions": sorted(defined),
        })

    # All checks passed
    return CodeValidationResult(True, "valid", source, {
        "character_count": len(source),
        "ast_node_count": node_count,
        "required_functions": sorted(required),
        "defined_functions": sorted(defined),
        "safety_checks_passed": True,
    })
