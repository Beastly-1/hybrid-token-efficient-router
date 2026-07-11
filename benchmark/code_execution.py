"""Isolated subprocess execution for code_tests benchmark cases.

This module executes candidate code in a temporary child process with:
- Subprocess isolation (shell=False, -I -S flags)
- Timeout protection
- Environment filtering (no secrets exposed)
- Platform-specific resource limits (best-effort on POSIX)
- Structured JSON result communication

IMPORTANT: This is defence-in-depth isolation for authored benchmark cases,
NOT a perfect security sandbox. The static safety checks in code_validation.py
provide the first layer; subprocess isolation provides the second.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from benchmark.code_validation import validate_code_answer, CodeValidationResult
from benchmark.dataset import BenchmarkCase

# ===================================================================
# Constants
# ===================================================================

_DEFAULT_TIMEOUT_SECONDS = 5.0
_MAX_TIMEOUT_SECONDS = 30.0
_MAX_OUTPUT_BYTES = 64 * 1024  # 64 KB max from child stdout

# Safe environment keys to pass to child process
_SAFE_ENV_KEYS = frozenset({
    "PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "LANG", "LC_ALL",
    "PYTHONPATH", "PYTHONHASHSEED",
})


# ===================================================================
# Result Type
# ===================================================================


@dataclass
class CodeExecutionResult:
    """Result from executing candidate code against benchmark tests."""

    status: str  # completed, timeout, runtime_error, invalid_test, infrastructure_error
    passed: int
    failed: int
    total: int
    timed_out: bool
    details: dict = field(default_factory=dict)


# ===================================================================
# Test-Call Parsing
# ===================================================================


@dataclass
class ParsedCall:
    """A parsed benchmark test call with literal arguments."""

    function_name: str
    args: list
    kwargs: dict


def parse_test_call(call_str: str) -> Optional[ParsedCall]:
    """Parse a benchmark test call string into structured literal arguments.

    Supports: add(2, 3), find_max([1, 5, 2]), greet(name="Maya")
    Rejects: attribute calls, indexed calls, starred args, **kwargs,
             non-literal arguments.

    Uses ast.parse + ast.literal_eval on individual argument nodes.
    Never uses eval().
    """
    try:
        tree = ast.parse(call_str.strip(), mode="eval")
    except SyntaxError:
        return None

    expr = tree.body
    if not isinstance(expr, ast.Call):
        return None

    # Must be a simple name
    if not isinstance(expr.func, ast.Name):
        return None

    func_name = expr.func.id

    # Reject starred arguments
    if expr.starargs if hasattr(expr, 'starargs') else False:
        return None
    for arg in expr.args:
        if isinstance(arg, ast.Starred):
            return None

    # Reject **kwargs expansion
    for kw in expr.keywords:
        if kw.arg is None:  # **kwargs
            return None

    # Parse positional args as literals
    args = []
    for arg in expr.args:
        try:
            val = ast.literal_eval(arg)
        except (ValueError, TypeError):
            return None
        args.append(val)

    # Parse keyword args as literals
    kwargs = {}
    for kw in expr.keywords:
        try:
            val = ast.literal_eval(kw.value)
        except (ValueError, TypeError):
            return None
        kwargs[kw.arg] = val

    return ParsedCall(function_name=func_name, args=args, kwargs=kwargs)


def parse_all_test_calls(case: BenchmarkCase) -> tuple[Optional[list[ParsedCall]], Optional[str]]:
    """Parse all test calls from a benchmark case.

    Returns (parsed_calls, None) on success or (None, error_message) on failure.
    """
    if not case.tests or not isinstance(case.tests, list):
        return None, "no tests defined"

    parsed = []
    for i, test in enumerate(case.tests):
        if not isinstance(test, dict):
            return None, f"test[{i}] is not a dict"
        call_str = test.get("call")
        if not isinstance(call_str, str) or not call_str.strip():
            return None, f"test[{i}] has no valid 'call' field"
        if "expected" not in test:
            return None, f"test[{i}] has no 'expected' field"

        result = parse_test_call(call_str)
        if result is None:
            return None, f"test[{i}] call '{call_str}' could not be parsed as a literal function call"
        parsed.append(result)

    return parsed, None


# ===================================================================
# Child Runner Script Generation
# ===================================================================


def _generate_runner_script(
    parsed_calls: list[ParsedCall],
    expected_values: list,
    result_path: str,
) -> str:
    """Generate the child runner script source code."""
    # Serialize test data as JSON for the child to load
    test_data = []
    for call, expected in zip(parsed_calls, expected_values):
        test_data.append({
            "function_name": call.function_name,
            "args": call.args,
            "kwargs": call.kwargs,
            "expected": expected,
        })

    test_data_json = json.dumps(test_data)
    result_path_escaped = result_path.replace("\\", "\\\\")

    script = textwrap.dedent(f"""\
        import json
        import sys
        import traceback

        # Apply resource limits on POSIX
        try:
            import resource
            # CPU time: 10 seconds
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
            # File size: 1 MB
            resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
            # Open files: 32
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        except (ImportError, ValueError, OSError):
            pass

        # Redirect stdout/stderr to suppress candidate output
        class _Null:
            def write(self, *a): pass
            def flush(self): pass
        sys.stdout = _Null()
        sys.stderr = _Null()

        result_path = "{result_path_escaped}"
        test_data = json.loads('''{test_data_json}''')

        results = []
        try:
            # Load candidate module
            import importlib.util
            spec = importlib.util.spec_from_file_location("candidate", "candidate.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            for i, td in enumerate(test_data):
                func_name = td["function_name"]
                args = td["args"]
                kwargs = td["kwargs"]
                expected = td["expected"]

                try:
                    func = getattr(mod, func_name, None)
                    if func is None:
                        results.append({{"index": i, "passed": False, "reason": "function_not_found"}})
                        continue
                    actual = func(*args, **kwargs)

                    # Type-sensitive comparison: bool vs int
                    if isinstance(expected, bool) or isinstance(actual, bool):
                        passed = (type(actual) is type(expected)) and (actual == expected)
                    else:
                        passed = (actual == expected)

                    if passed:
                        results.append({{"index": i, "passed": True}})
                    else:
                        actual_repr = repr(actual)[:200]
                        results.append({{"index": i, "passed": False, "reason": "value_mismatch", "actual": actual_repr}})
                except Exception as e:
                    results.append({{"index": i, "passed": False, "reason": "exception", "exception_type": type(e).__name__}})

        except Exception as e:
            # Module-level error
            for i in range(len(test_data)):
                if i >= len(results):
                    results.append({{"index": i, "passed": False, "reason": "module_error", "exception_type": type(e).__name__}})

        # Write result
        output = {{
            "status": "completed",
            "results": results,
            "total": len(test_data),
        }}
        with open(result_path, "w") as f:
            json.dump(output, f)
    """)
    return script


# ===================================================================
# Subprocess Execution
# ===================================================================


def _build_safe_env() -> dict:
    """Build a minimal environment for the child process."""
    env = {}
    for key in _SAFE_ENV_KEYS:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    return env


def execute_code_tests(
    case: BenchmarkCase,
    candidate: Any,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> CodeExecutionResult:
    """Execute candidate code against benchmark tests in an isolated subprocess.

    Flow:
    1. Validate candidate statically
    2. Parse all test calls
    3. Create temp dir, write candidate + runner
    4. Launch subprocess with isolation flags
    5. Read structured result
    6. Clean up

    Returns CodeExecutionResult with pass/fail counts and details.
    """
    # Clamp timeout
    timeout = min(max(0.1, timeout), _MAX_TIMEOUT_SECONDS)

    # Step 1: Static validation
    validation = validate_code_answer(case, candidate)
    if not validation.valid:
        return CodeExecutionResult(
            status="invalid_test" if validation.status == "invalid_answer" and "scoring_method" in validation.details.get("reason", "") else "invalid_test",
            passed=0, failed=0, total=0, timed_out=False,
            details={"validation_status": validation.status, "validation_details": validation.details},
        )

    source_code = validation.source_code

    # Step 2: Parse test calls
    parsed_calls, parse_error = parse_all_test_calls(case)
    if parsed_calls is None:
        return CodeExecutionResult(
            status="invalid_test",
            passed=0, failed=0, total=0, timed_out=False,
            details={"reason": parse_error},
        )

    total = len(parsed_calls)
    expected_values = [t["expected"] for t in case.tests]

    # Step 3: Create temp directory and write files
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="code_exec_")
        tmp_path = Path(tmp_dir)

        candidate_path = tmp_path / "candidate.py"
        candidate_path.write_text(source_code, encoding="utf-8")

        result_file = tmp_path / "result.json"
        runner_script = _generate_runner_script(parsed_calls, expected_values, str(result_file))

        runner_path = tmp_path / "runner.py"
        runner_path.write_text(runner_script, encoding="utf-8")

        # Step 4: Launch subprocess
        cmd = [sys.executable, "-I", "-S", str(runner_path)]
        env = _build_safe_env()

        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp_dir,
                timeout=timeout,
                capture_output=True,
                shell=False,
                stdin=subprocess.DEVNULL,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return CodeExecutionResult(
                status="timeout",
                passed=0, failed=total, total=total, timed_out=True,
                details={"execution_status": "timeout", "timeout_seconds": timeout},
            )

        # Step 5: Read result
        if not result_file.exists():
            stderr_snippet = proc.stderr[:500].decode("utf-8", errors="replace") if proc.stderr else ""
            return CodeExecutionResult(
                status="runtime_error",
                passed=0, failed=total, total=total, timed_out=False,
                details={
                    "execution_status": "runtime_error",
                    "reason": "no result file produced",
                    "returncode": proc.returncode,
                    "stderr_snippet": stderr_snippet[:200],
                },
            )

        try:
            result_data = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            return CodeExecutionResult(
                status="infrastructure_error",
                passed=0, failed=total, total=total, timed_out=False,
                details={"reason": f"could not read result: {type(e).__name__}"},
            )

        # Parse results
        test_results = result_data.get("results", [])
        passed = sum(1 for r in test_results if r.get("passed"))
        failed = total - passed

        return CodeExecutionResult(
            status="completed",
            passed=passed, failed=failed, total=total, timed_out=False,
            details={
                "execution_status": "completed",
                "passed_tests": passed,
                "failed_tests": failed,
                "total_tests": total,
                "timed_out": False,
                "test_results": test_results,
            },
        )

    except OSError as e:
        return CodeExecutionResult(
            status="infrastructure_error",
            passed=0, failed=0, total=total, timed_out=False,
            details={"reason": f"infrastructure error: {type(e).__name__}: {e}"},
        )
    finally:
        # Step 6: Cleanup
        if tmp_dir:
            _cleanup_dir(Path(tmp_dir))


def _cleanup_dir(path: Path) -> None:
    """Best-effort recursive cleanup of temporary directory."""
    try:
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
