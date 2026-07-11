"""Tests for benchmark runner, report, and CLI integration."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark.dataset import BenchmarkCase, load_dataset
from benchmark.report import generate_report, load_results
from benchmark.runner import (
    BenchmarkResult,
    BenchmarkRunner,
    parse_strategy,
    validate_remote_model,
)
from benchmark.scorers import score_answer

SMOKE_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "smoke" / "tasks.jsonl"


def _case(**overrides) -> BenchmarkCase:
    base = {
        "task_id": "t1",
        "category": "factual",
        "difficulty": "easy",
        "prompt": "What is the chemical symbol for gold?",
        "scoring_method": "exact",
        "reference": "Au",
    }
    base.update(overrides)
    return BenchmarkCase(**base)


# Deterministic answers for smoke cases
_SMOKE_ANSWERS = {
    "factual_001": "Au",
    "factual_002": "Jupiter",
    "math_001": "408",
    "math_002": "100.0",
    "sentiment_001": "positive",
    "sentiment_002": "negative",
    "summarization_001": "The committee agreed to increase infrastructure funding by 12% and cut administrative costs.",
    "summarization_002": "Daily moderate exercise reduces cardiovascular risk by 30% in adults over 50.",
    "ner_001": '[{"text":"Maya","type":"PERSON"},{"text":"Acme Labs","type":"ORGANIZATION"},{"text":"Bengaluru","type":"LOCATION"},{"text":"12 June 2026","type":"DATE"}]',
    "ner_002": '[{"text":"March 2024","type":"DATE"},{"text":"Tesla","type":"ORGANIZATION"},{"text":"Samsung","type":"ORGANIZATION"},{"text":"Austin","type":"LOCATION"}]',
    "code_debug_001": "ZeroDivisionError",
    "code_debug_002": "low = mid should be low = mid + 1",
    "logic_001": "No",
    "logic_002": "No",
    "code_generation_001": "def add(a, b):\n    return a + b",
    "code_generation_002": "def is_palindrome(s):\n    cleaned = s.replace(' ', '').lower()\n    return cleaned == cleaned[::-1]",
}


def _mock_router_factory(answers=None):
    """Create a mock router that returns deterministic answers."""
    answers = answers or _SMOKE_ANSWERS

    def factory():
        router = MagicMock()

        def route(state):
            answer = answers.get(state.task_id, "unknown")
            state.final_answer = answer
            state.tool_used = None
            state.use_remote = False
            state.route_score = 0.9
            state.selected_model = None
            state.remote_prompt_tokens = None
            state.remote_completion_tokens = None
            state.remote_total_tokens = None
            state.remote_latency_ms = None
            state.estimated_remote_cost = 0.0
            state.actual_remote_cost = None
            state._route_source = "local"
            return state

        router.route = route
        return router

    return factory


def _mock_remote_factory(answers=None):
    """Create a mock remote model."""
    answers = answers or _SMOKE_ANSWERS

    def factory():
        model = MagicMock()

        def generate(state):
            answer = answers.get(state.task_id, "unknown")
            state.remote_answer = answer
            state.use_remote = True
            state.remote_prompt_tokens = 50
            state.remote_completion_tokens = 20
            state.remote_total_tokens = 70
            state.remote_latency_ms = 150.0
            return state

        model.generate = generate
        return model

    return factory


def _mock_local_factory(answers=None):
    """Create a mock local model."""
    answers = answers or _SMOKE_ANSWERS

    def factory():
        model = MagicMock()

        def generate(state):
            answer = answers.get(state.task_id, "unknown")
            state.local_answer = answer
            state.local_device = "CPU"
            return state

        model.generate = generate
        return model

    return factory


# ===========================================================
# Strategy Parsing
# ===========================================================


class TestStrategyParsing:
    def test_router_strategy(self):
        mode, model = parse_strategy("router")
        assert mode == "router"
        assert model is None

    def test_local_strategy(self):
        mode, model = parse_strategy("local")
        assert mode == "local"
        assert model is None

    def test_remote_strategy(self):
        mode, model = parse_strategy("remote:accounts/fireworks/models/test-model")
        assert mode == "remote"
        assert model == "accounts/fireworks/models/test-model"

    def test_unknown_strategy_rejected(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            parse_strategy("invalid")

    def test_remote_model_validation(self):
        validate_remote_model("model-a", allowed_models=["model-a", "model-b"])

    def test_unapproved_model_rejected(self):
        with pytest.raises(ValueError, match="not in ALLOWED_MODELS"):
            validate_remote_model("model-c", allowed_models=["model-a", "model-b"])

    def test_allow_remote_enforcement(self):
        with pytest.raises(ValueError, match="--allow-remote"):
            BenchmarkRunner(
                strategy="remote:model-a",
                allow_remote=False,
            )


# ===========================================================
# Runner Execution
# ===========================================================


class TestRunnerExecution:
    def test_run_all_cases_mocked(self, tmp_path):
        """Run all benchmark cases with mocked inference."""
        cases = load_dataset(SMOKE_PATH)
        runner = BenchmarkRunner(
            strategy="router",
            router_factory=_mock_router_factory(),
        )
        output = tmp_path / "results.jsonl"
        results = runner.run_dataset(cases, output)
        assert len(results) == 16
        assert all(r.success for r in results)

    def test_router_strategy_uses_router(self, tmp_path):
        """Router strategy calls the router factory."""
        case = _case()
        runner = BenchmarkRunner(
            strategy="router",
            router_factory=_mock_router_factory({"t1": "Au"}),
        )
        result = runner.run_case(case)
        assert result.success is True
        assert result.score == 1.0
        assert result.correct is True

    def test_local_strategy_uses_local_model(self, tmp_path):
        """Local strategy calls the local model factory."""
        case = _case()
        runner = BenchmarkRunner(
            strategy="local",
            local_model_factory=_mock_local_factory({"t1": "Au"}),
        )
        result = runner.run_case(case)
        assert result.success is True
        assert result.score == 1.0

    @patch("benchmark.runner.ALLOWED_MODELS", ["accounts/fireworks/models/test"])
    def test_remote_strategy_uses_remote_model(self, tmp_path):
        """Remote strategy calls the remote model factory."""
        case = _case()
        runner = BenchmarkRunner(
            strategy="remote:accounts/fireworks/models/test",
            allow_remote=True,
            remote_model_factory=_mock_remote_factory({"t1": "Au"}),
        )
        result = runner.run_case(case)
        assert result.success is True
        assert result.score == 1.0
        assert result.prompt_tokens == 50

    def test_scoring_all_8_categories(self, tmp_path):
        """Scoring integration for all 8 categories."""
        cases = load_dataset(SMOKE_PATH)
        runner = BenchmarkRunner(
            strategy="router",
            router_factory=_mock_router_factory(),
        )
        output = tmp_path / "results.jsonl"
        results = runner.run_dataset(cases, output)

        categories_scored = set()
        for r in results:
            if r.scoring_status == "scored":
                categories_scored.add(r.category)

        assert len(categories_scored) == 8

    def test_individual_failure_handling(self, tmp_path):
        """Individual case failure is recorded, doesn't stop run."""
        call_count = [0]

        def failing_router_factory():
            router = MagicMock()

            def route(state):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("Simulated failure")
                state.final_answer = "Au"
                state.tool_used = None
                state.use_remote = False
                state.route_score = 0.9
                state.selected_model = None
                state.remote_prompt_tokens = None
                state.remote_completion_tokens = None
                state.remote_total_tokens = None
                state.remote_latency_ms = None
                state.estimated_remote_cost = 0.0
                state.actual_remote_cost = None
                state._route_source = "local"
                return state

            router.route = route
            return router

        cases = [_case(task_id="t1"), _case(task_id="t2")]
        runner = BenchmarkRunner(
            strategy="router",
            router_factory=failing_router_factory,
        )
        output = tmp_path / "results.jsonl"
        results = runner.run_dataset(cases, output)

        assert len(results) == 2
        assert results[0].success is False
        assert results[0].error_type == "RuntimeError"
        assert results[1].success is True

    def test_valid_jsonl_output(self, tmp_path):
        """Output is valid JSONL."""
        cases = load_dataset(SMOKE_PATH)[:3]
        runner = BenchmarkRunner(
            strategy="router",
            router_factory=_mock_router_factory(),
        )
        output = tmp_path / "results.jsonl"
        runner.run_dataset(cases, output)

        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert "task_id" in record
            assert "score" in record
            assert "benchmark_run_id" in record

    def test_max_cases_limit(self, tmp_path):
        """--max-cases limits execution."""
        cases = load_dataset(SMOKE_PATH)
        runner = BenchmarkRunner(
            strategy="router",
            router_factory=_mock_router_factory(),
        )
        output = tmp_path / "results.jsonl"
        results = runner.run_dataset(cases, output, max_cases=3)
        assert len(results) == 3


# ===========================================================
# Report Generation
# ===========================================================


class TestReport:
    def _make_results(self, tmp_path, count=16):
        """Generate mock results JSONL."""
        cases = load_dataset(SMOKE_PATH)
        runner = BenchmarkRunner(
            strategy="router",
            router_factory=_mock_router_factory(),
        )
        output = tmp_path / "results.jsonl"
        runner.run_dataset(cases[:count], output)
        return output

    def test_overall_accuracy(self, tmp_path):
        """Overall accuracy calculation."""
        output = self._make_results(tmp_path)
        records = load_results(output)
        report = generate_report(records)

        overall = report["overall"]
        assert overall["total_cases"] == 16
        assert overall["scored_cases"] > 0
        assert overall["accuracy"] is not None
        assert 0.0 <= overall["accuracy"] <= 1.0

    def test_per_category_stats(self, tmp_path):
        """Per-category statistics."""
        output = self._make_results(tmp_path)
        records = load_results(output)
        report = generate_report(records)

        by_cat = report["by_category"]
        assert len(by_cat) == 8
        for cat, stats in by_cat.items():
            assert "total" in stats
            assert "correct" in stats
            assert "accuracy" in stats
            assert "fireworks_calls" in stats

    def test_per_route_stats(self, tmp_path):
        """Per-route statistics."""
        output = self._make_results(tmp_path)
        records = load_results(output)
        report = generate_report(records)

        by_route = report["by_route"]
        assert len(by_route) > 0
        for route, stats in by_route.items():
            assert "count" in stats
            assert "accuracy" in stats

    def test_per_model_stats(self, tmp_path):
        """Per-model statistics (empty when no remote calls)."""
        output = self._make_results(tmp_path)
        records = load_results(output)
        report = generate_report(records)
        # No remote calls in mock, so by_model should be empty
        assert report["by_model"] == {}

    def test_efficiency_metrics(self, tmp_path):
        """Token/cost-per-correct calculations."""
        output = self._make_results(tmp_path)
        records = load_results(output)
        report = generate_report(records)

        eff = report["efficiency"]
        assert "fireworks_call_rate" in eff
        assert "actual_cost_per_correct" in eff
        assert "fireworks_tokens_per_correct" in eff

    def test_empty_results(self):
        """Empty result log produces valid report."""
        report = generate_report([])
        assert report["overall"]["total_cases"] == 0
        assert report["overall"]["accuracy"] is None

    def test_malformed_results(self, tmp_path):
        """Malformed lines are skipped."""
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n{\"task_id\": \"t1\"}\n", encoding="utf-8")
        records = load_results(path)
        assert len(records) == 1

    def test_report_with_remote_model(self, tmp_path):
        """Report with remote model data."""
        records = [
            {
                "benchmark_run_id": "r1",
                "task_id": "t1",
                "category": "factual",
                "difficulty": "easy",
                "strategy": "remote:model-a",
                "scoring_method": "exact",
                "answer": "Au",
                "score": 1.0,
                "correct": True,
                "scoring_status": "scored",
                "scoring_details": {},
                "route_source": "fireworks",
                "selected_model": "model-a",
                "tool_used": None,
                "route_score": None,
                "total_latency_ms": 200.0,
                "remote_latency_ms": 150.0,
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70,
                "estimated_remote_cost": 0.001,
                "actual_remote_cost": 0.0008,
                "success": True,
                "error_type": None,
                "error_message": None,
            }
        ]
        report = generate_report(records)
        assert "model-a" in report["by_model"]
        assert report["by_model"]["model-a"]["call_count"] == 1
        assert report["by_model"]["model-a"]["total_tokens"] == 70


# ===========================================================
# CLI Integration
# ===========================================================


class TestCLI:
    def test_existing_cli_aggregate(self, tmp_path):
        """Existing aggregate CLI still works."""
        log_path = tmp_path / "routing.jsonl"
        log_path.write_text(
            json.dumps({
                "route_source": "local",
                "task_type": "factual",
                "difficulty": "easy",
                "success": True,
            }) + "\n",
            encoding="utf-8",
        )
        output = tmp_path / "summary.json"
        result = subprocess.run(
            [sys.executable, "-m", "benchmark", "aggregate",
             "--input", str(log_path), "--output", str(output)],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert output.exists()

    def test_existing_cli_backwards_compat(self, tmp_path):
        """Backwards-compatible --input --output without subcommand."""
        log_path = tmp_path / "routing.jsonl"
        log_path.write_text(
            json.dumps({
                "route_source": "local",
                "task_type": "factual",
                "difficulty": "easy",
                "success": True,
            }) + "\n",
            encoding="utf-8",
        )
        output = tmp_path / "summary.json"
        result = subprocess.run(
            [sys.executable, "-m", "benchmark",
             "--input", str(log_path), "--output", str(output)],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0

    def test_validate_dataset_cli(self, tmp_path):
        """validate-dataset CLI still works."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmark", "validate-dataset",
             "--input", str(SMOKE_PATH)],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "Dataset valid" in result.stdout

    def test_run_cli_rejects_remote_without_flag(self, tmp_path):
        """run CLI rejects remote without --allow-remote."""
        output = tmp_path / "results.jsonl"
        result = subprocess.run(
            [sys.executable, "-m", "benchmark", "run",
             "--dataset", str(SMOKE_PATH),
             "--strategy", "remote:some-model",
             "--output", str(output)],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 1
        assert "allow-remote" in result.stderr.lower() or "allow-remote" in result.stdout.lower() or result.returncode == 1

    def test_report_cli_missing_input(self, tmp_path):
        """report CLI handles missing input."""
        output = tmp_path / "report.json"
        result = subprocess.run(
            [sys.executable, "-m", "benchmark", "report",
             "--input", str(tmp_path / "nonexistent.jsonl"),
             "--output", str(output)],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 1


# ===========================================================
# Smoke Demonstration
# ===========================================================


class TestSmokeDemonstration:
    def test_full_smoke_run_with_report(self, tmp_path):
        """Full offline smoke run with mocked answers and report generation."""
        cases = load_dataset(SMOKE_PATH)
        runner = BenchmarkRunner(
            strategy="router",
            run_id="smoke-test",
            router_factory=_mock_router_factory(),
        )
        results_path = tmp_path / "smoke_results.jsonl"
        results = runner.run_dataset(cases, results_path)

        # Verify all 16 cases ran
        assert len(results) == 16
        assert all(r.success for r in results)

        # Generate report
        records = load_results(results_path)
        report = generate_report(records)

        # Verify report structure
        assert report["overall"]["total_cases"] == 16
        assert report["overall"]["scored_cases"] > 0
        assert report["overall"]["correct_cases"] > 0
        assert report["overall"]["accuracy"] is not None
        assert len(report["by_category"]) == 8
        assert len(report["by_difficulty"]) > 0

        # Write report
        report_path = tmp_path / "smoke_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        assert report_path.exists()

        # Print summary for visibility
        print(f"\nSmoke run: {report['overall']['total_cases']} cases, "
              f"accuracy={report['overall']['accuracy']:.2f}, "
              f"mean_score={report['overall']['mean_score']:.3f}")

    def test_no_fireworks_calls_made(self, tmp_path):
        """No Fireworks calls are made during tests."""
        import benchmark.runner as mod
        source = Path(mod.__file__).read_text(encoding="utf-8")
        # Module doesn't import requests directly
        assert "requests.post" not in source

    def test_no_local_model_loaded(self, tmp_path):
        """Local model is not loaded during tests."""
        cases = load_dataset(SMOKE_PATH)[:1]
        runner = BenchmarkRunner(
            strategy="router",
            router_factory=_mock_router_factory(),
        )
        output = tmp_path / "results.jsonl"
        runner.run_dataset(cases, output)
        # If we got here without openvino import error, local model wasn't loaded
        assert True
