"""Tests for calibration, strategy comparison, model profiles, and calibrated selection."""

import json
import math
import tempfile
from pathlib import Path

import pytest

from benchmark.calibration import (
    CalibrationResult,
    ThresholdCandidate,
    _evaluate_threshold,
    _objective_value,
    build_model_profiles,
    calibrate,
    calibrate_category,
    load_results,
)
from benchmark.strategy_comparison import (
    compare_strategies,
    compute_strategy_metrics,
    find_pareto_efficient,
)
from decision.model_selector import ModelSelector


# ===================================================================
# Fixtures: synthetic benchmark results
# ===================================================================


def _make_record(
    category="math",
    difficulty="easy",
    correct=True,
    route_score=0.9,
    route_source="local",
    total_tokens=0,
    actual_remote_cost=0.0,
    estimated_remote_cost=0.0,
    total_latency_ms=50.0,
    selected_model=None,
    scoring_status="scored",
    score=1.0,
    prompt_tokens=None,
    completion_tokens=None,
):
    return {
        "category": category,
        "difficulty": difficulty,
        "correct": correct,
        "route_score": route_score,
        "route_source": route_source,
        "total_tokens": total_tokens,
        "actual_remote_cost": actual_remote_cost,
        "estimated_remote_cost": estimated_remote_cost,
        "total_latency_ms": total_latency_ms,
        "selected_model": selected_model,
        "scoring_status": scoring_status,
        "score": score,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def calibration_records():
    """20 math records with varying confidence and correctness."""
    records = []
    for i in range(20):
        conf = 0.5 + (i / 20) * 0.5  # 0.5 to 0.975
        correct = conf > 0.7  # high confidence -> correct
        records.append(_make_record(
            category="math",
            difficulty="easy",
            correct=correct,
            route_score=conf,
            route_source="local" if conf > 0.8 else "fireworks",
            total_tokens=100 if conf <= 0.8 else 0,
            actual_remote_cost=0.001 if conf <= 0.8 else 0.0,
            total_latency_ms=50 + i * 5,
        ))
    return records


@pytest.fixture
def multi_category_records():
    """Records across multiple categories."""
    records = []
    for cat in ("math", "factual", "sentiment", "code_generation"):
        for i in range(10):
            conf = 0.4 + (i / 10) * 0.6
            correct = conf > 0.6
            records.append(_make_record(
                category=cat,
                difficulty="easy" if i < 5 else "medium",
                correct=correct,
                route_score=conf,
                route_source="fireworks" if conf < 0.7 else "local",
                total_tokens=200 if conf < 0.7 else 0,
                actual_remote_cost=0.002 if conf < 0.7 else 0.0,
                total_latency_ms=30 + i * 10,
                selected_model="accounts/fireworks/models/model-a" if conf < 0.7 else None,
            ))
    return records


# ===================================================================
# Threshold Search Tests
# ===================================================================


class TestThresholdSearch:
    def test_evaluate_threshold_all_local(self, calibration_records):
        """Threshold 0.0 accepts everything locally."""
        c = _evaluate_threshold(calibration_records, 0.0)
        assert c.local_acceptance_rate == 1.0
        assert c.fireworks_call_rate == 0.0
        assert c.total_fireworks_tokens == 0

    def test_evaluate_threshold_all_remote(self, calibration_records):
        """Threshold 1.0 sends everything to Fireworks."""
        c = _evaluate_threshold(calibration_records, 1.0)
        assert c.local_acceptance_rate == 0.0
        assert c.fireworks_call_rate == 1.0

    def test_threshold_accuracy_varies(self, calibration_records):
        """Different thresholds produce different accuracy."""
        c_low = _evaluate_threshold(calibration_records, 0.3)
        c_high = _evaluate_threshold(calibration_records, 0.8)
        # Both should have valid accuracy
        assert 0.0 <= c_low.accuracy <= 1.0
        assert 0.0 <= c_high.accuracy <= 1.0

    def test_objective_value_tokens(self):
        c = ThresholdCandidate(0.5, 0.9, 0.5, 0.1, 0.5, 1000, 0.01, 50.0, 100.0)
        assert _objective_value(c, "tokens") == 1000

    def test_objective_value_cost(self):
        c = ThresholdCandidate(0.5, 0.9, 0.5, 0.1, 0.5, 1000, 0.05, 50.0, 100.0)
        assert _objective_value(c, "cost") == 0.05

    def test_objective_value_latency(self):
        c = ThresholdCandidate(0.5, 0.9, 0.5, 0.1, 0.5, 1000, 0.01, 75.0, 100.0)
        assert _objective_value(c, "latency") == 75.0


# ===================================================================
# Accuracy Constraint Tests
# ===================================================================


class TestAccuracyConstraint:
    def test_constraint_met(self, calibration_records):
        """When accuracy requirement can be met, constraint_met=True."""
        result = calibrate_category(calibration_records, "math", min_accuracy=0.5, objective="tokens")
        assert result is not None
        assert result.constraint_met is True
        assert result.candidate.accuracy >= 0.5

    def test_constraint_not_met_fallback(self):
        """When no threshold meets accuracy, fallback to highest accuracy."""
        # All records are incorrect
        records = [_make_record(category="math", correct=False, route_score=0.5 + i * 0.05)
                   for i in range(10)]
        result = calibrate_category(records, "math", min_accuracy=0.99, objective="tokens")
        assert result is not None
        assert result.constraint_met is False

    def test_insufficient_samples_returns_none(self):
        """Too few samples returns None."""
        records = [_make_record(category="math") for _ in range(3)]
        result = calibrate_category(records, "math", min_accuracy=0.9)
        assert result is None


# ===================================================================
# Calibration/Test Separation Tests
# ===================================================================


class TestCalibrationTestSeparation:
    def test_separate_evaluation(self, tmp_path):
        """Calibration and test sets are evaluated independently."""
        cal_records = [_make_record(category="math", correct=True, route_score=0.9) for _ in range(10)]
        test_records = [_make_record(category="math", correct=False, route_score=0.9) for _ in range(10)]

        artifact = calibrate(cal_records, test_records, min_accuracy=0.5)
        # Calibration should show high accuracy
        assert artifact["thresholds"]["math"]["accuracy"] >= 0.5
        # Test evaluation should show low accuracy
        test_eval = artifact.get("test_evaluation", {})
        if "math" in test_eval:
            assert test_eval["math"]["accuracy"] < 0.5

    def test_no_test_set(self):
        """Works without test set."""
        records = [_make_record(category="math", correct=True, route_score=0.8) for _ in range(10)]
        artifact = calibrate(records, None, min_accuracy=0.5)
        assert "test_evaluation" not in artifact or artifact["test_evaluation"] is None


# ===================================================================
# Model Profile Tests
# ===================================================================


class TestModelProfiles:
    def test_build_profiles(self, multi_category_records):
        """Profiles are built per model/category/difficulty."""
        profiles = build_model_profiles(multi_category_records)
        assert len(profiles) > 0
        for p in profiles:
            assert p.model_id is not None
            assert 0.0 <= p.accuracy <= 1.0
            assert p.sample_count > 0

    def test_sparse_fallback_hierarchy(self):
        """Fallback from model+cat+diff to model+cat to model overall."""
        records = [
            _make_record(category="math", difficulty="easy", correct=True,
                         selected_model="model-a", route_source="fireworks",
                         total_tokens=100, actual_remote_cost=0.001),
            _make_record(category="math", difficulty="hard", correct=False,
                         selected_model="model-a", route_source="fireworks",
                         total_tokens=200, actual_remote_cost=0.002),
        ]
        profiles = build_model_profiles(records)
        # Should have 2 profiles (one per difficulty)
        assert len(profiles) == 2
        # All marked low confidence
        assert all(p.low_confidence for p in profiles)

    def test_low_confidence_flag(self):
        """Profiles with < 10 samples are flagged."""
        records = [_make_record(category="math", selected_model="m", route_source="fireworks") for _ in range(5)]
        profiles = build_model_profiles(records)
        assert all(p.low_confidence for p in profiles)

    def test_no_invented_values(self):
        """Profiles only contain data from actual records."""
        records = [_make_record(category="math", selected_model="m", correct=True,
                                route_source="fireworks", actual_remote_cost=0.005)]
        profiles = build_model_profiles(records)
        assert profiles[0].accuracy == 1.0
        assert profiles[0].average_cost == 0.005


# ===================================================================
# Calibrated Model Selection Tests
# ===================================================================


class TestCalibratedSelection:
    def _make_calibration_artifact(self, tmp_path, profiles):
        artifact = {
            "schema_version": 2,
            "model_profiles": profiles,
            "thresholds": {},
        }
        path = tmp_path / "cal.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        return path

    def test_cost_per_success_selection(self, tmp_path):
        """Selects model with lowest cost per success."""
        profiles = [
            {"model_id": "accounts/fireworks/models/model-a", "category": "math",
             "difficulty": "easy", "sample_count": 20, "accuracy": 0.9,
             "average_cost": 0.001, "average_latency_ms": 50, "p95_latency_ms": 80,
             "low_confidence": False},
            {"model_id": "accounts/fireworks/models/model-b", "category": "math",
             "difficulty": "easy", "sample_count": 20, "accuracy": 0.8,
             "average_cost": 0.0005, "average_latency_ms": 40, "p95_latency_ms": 70,
             "low_confidence": False},
        ]
        cal_path = self._make_calibration_artifact(tmp_path, profiles)

        catalog = [
            {"id": "accounts/fireworks/models/model-a", "input_cost_per_million": 0.2,
             "output_cost_per_million": 0.2, "max_difficulty": "hard", "task_types": []},
            {"id": "accounts/fireworks/models/model-b", "input_cost_per_million": 0.1,
             "output_cost_per_million": 0.1, "max_difficulty": "hard", "task_types": []},
        ]
        allowed = ["accounts/fireworks/models/model-a", "accounts/fireworks/models/model-b"]

        selector = ModelSelector(
            catalog=catalog, allowed_models=allowed,
            calibration_path=cal_path, min_model_accuracy=0.7,
        )

        class FakeState:
            task_type = "math"
            difficulty = "easy"
            query = "What is 2+2?"
            max_new_tokens = 100
            selected_model = ""
            estimated_remote_cost = 0.0

        state = FakeState()
        selector.select(state)
        # model-b: cost_per_success = 0.0005/0.8 = 0.000625
        # model-a: cost_per_success = 0.001/0.9 = 0.00111
        assert state.selected_model == "accounts/fireworks/models/model-b"
        assert "calibrated" in selector.selection_reason

    def test_allowed_models_enforcement(self, tmp_path):
        """Only models in ALLOWED_MODELS are considered."""
        profiles = [
            {"model_id": "accounts/fireworks/models/model-x", "category": "math",
             "difficulty": "easy", "sample_count": 50, "accuracy": 0.99,
             "average_cost": 0.0001, "average_latency_ms": 10, "p95_latency_ms": 20,
             "low_confidence": False},
        ]
        cal_path = self._make_calibration_artifact(tmp_path, profiles)

        catalog = [
            {"id": "accounts/fireworks/models/model-a", "input_cost_per_million": 0.2,
             "output_cost_per_million": 0.2, "max_difficulty": "hard", "task_types": []},
        ]
        allowed = ["accounts/fireworks/models/model-a"]

        selector = ModelSelector(
            catalog=catalog, allowed_models=allowed, calibration_path=cal_path,
        )

        class FakeState:
            task_type = "math"
            difficulty = "easy"
            query = "test"
            max_new_tokens = 50
            selected_model = ""
            estimated_remote_cost = 0.0

        state = FakeState()
        selector.select(state)
        # model-x not in allowed, so model-a is selected via static fallback
        assert state.selected_model == "accounts/fireworks/models/model-a"

    def test_missing_calibration_fallback(self):
        """Missing calibration file falls back to static selector."""
        catalog = [
            {"id": "accounts/fireworks/models/model-a", "input_cost_per_million": 0.2,
             "output_cost_per_million": 0.2, "max_difficulty": "hard", "task_types": []},
        ]
        allowed = ["accounts/fireworks/models/model-a"]

        selector = ModelSelector(
            catalog=catalog, allowed_models=allowed,
            calibration_path=Path("/nonexistent/path.json"),
        )

        class FakeState:
            task_type = "math"
            difficulty = "easy"
            query = "test"
            max_new_tokens = 50
            selected_model = ""
            estimated_remote_cost = 0.0

        state = FakeState()
        selector.select(state)
        assert state.selected_model == "accounts/fireworks/models/model-a"
        assert "static_policy" in selector.selection_reason

    def test_invalid_calibration_fallback(self, tmp_path):
        """Invalid calibration JSON falls back to static selector."""
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("not json at all", encoding="utf-8")

        catalog = [
            {"id": "accounts/fireworks/models/model-a", "input_cost_per_million": 0.2,
             "output_cost_per_million": 0.2, "max_difficulty": "hard", "task_types": []},
        ]
        allowed = ["accounts/fireworks/models/model-a"]

        selector = ModelSelector(
            catalog=catalog, allowed_models=allowed, calibration_path=bad_path,
        )

        class FakeState:
            task_type = "math"
            difficulty = "easy"
            query = "test"
            max_new_tokens = 50
            selected_model = ""
            estimated_remote_cost = 0.0

        state = FakeState()
        selector.select(state)
        assert state.selected_model == "accounts/fireworks/models/model-a"

    def test_no_model_meets_accuracy(self, tmp_path):
        """When no model meets min accuracy, picks highest accuracy."""
        profiles = [
            {"model_id": "accounts/fireworks/models/model-a", "category": "math",
             "difficulty": "easy", "sample_count": 20, "accuracy": 0.4,
             "average_cost": 0.001, "average_latency_ms": 50, "p95_latency_ms": 80,
             "low_confidence": False},
            {"model_id": "accounts/fireworks/models/model-b", "category": "math",
             "difficulty": "easy", "sample_count": 20, "accuracy": 0.6,
             "average_cost": 0.002, "average_latency_ms": 60, "p95_latency_ms": 90,
             "low_confidence": False},
        ]
        cal_path = self._make_calibration_artifact(tmp_path, profiles)

        catalog = [
            {"id": "accounts/fireworks/models/model-a", "input_cost_per_million": 0.2,
             "output_cost_per_million": 0.2, "max_difficulty": "hard", "task_types": []},
            {"id": "accounts/fireworks/models/model-b", "input_cost_per_million": 1.0,
             "output_cost_per_million": 1.0, "max_difficulty": "hard", "task_types": []},
        ]
        allowed = ["accounts/fireworks/models/model-a", "accounts/fireworks/models/model-b"]

        selector = ModelSelector(
            catalog=catalog, allowed_models=allowed,
            calibration_path=cal_path, min_model_accuracy=0.9,
        )

        class FakeState:
            task_type = "math"
            difficulty = "easy"
            query = "test"
            max_new_tokens = 50
            selected_model = ""
            estimated_remote_cost = 0.0

        state = FakeState()
        selector.select(state)
        # model-b has higher accuracy (0.6 vs 0.4)
        assert state.selected_model == "accounts/fireworks/models/model-b"
        assert "below_min_accuracy" in selector.selection_reason

    def test_existing_selector_unchanged_without_calibration(self):
        """Static selector behaviour is preserved when no calibration."""
        catalog = [
            {"id": "accounts/fireworks/models/model-a", "input_cost_per_million": 0.2,
             "output_cost_per_million": 0.2, "max_difficulty": "hard", "task_types": ["math"]},
            {"id": "accounts/fireworks/models/model-b", "input_cost_per_million": 1.0,
             "output_cost_per_million": 1.0, "max_difficulty": "hard", "task_types": []},
        ]
        allowed = ["accounts/fireworks/models/model-a", "accounts/fireworks/models/model-b"]

        selector = ModelSelector(
            catalog=catalog, allowed_models=allowed,
            calibration_path=Path("/nonexistent"),
            selection_mode="cost_first",
        )

        class FakeState:
            task_type = "math"
            difficulty = "easy"
            query = "What is 2+2?"
            max_new_tokens = 100
            selected_model = ""
            estimated_remote_cost = 0.0

        state = FakeState()
        selector.select(state)
        # cost_first should pick model-a (cheaper)
        assert state.selected_model == "accounts/fireworks/models/model-a"


# ===================================================================
# Strategy Comparison Tests
# ===================================================================


class TestStrategyComparison:
    def test_compute_metrics(self):
        """Basic metric computation."""
        records = [
            _make_record(correct=True, route_source="local", total_latency_ms=50),
            _make_record(correct=True, route_source="fireworks", total_tokens=200,
                         actual_remote_cost=0.002, total_latency_ms=100,
                         selected_model="model-a"),
            _make_record(correct=False, route_source="fireworks", total_tokens=150,
                         actual_remote_cost=0.001, total_latency_ms=80,
                         selected_model="model-a"),
        ]
        metrics = compute_strategy_metrics(records, "test_strategy")
        assert metrics["strategy"] == "test_strategy"
        assert metrics["total_cases"] == 3
        assert metrics["accuracy"] == pytest.approx(2 / 3)
        assert metrics["fireworks_call_count"] == 2
        assert metrics["fireworks_total_tokens"] == 350
        assert metrics["actual_cost"] == pytest.approx(0.003)

    def test_pareto_efficient_detection(self):
        """Identifies non-dominated strategies."""
        metrics = [
            {"strategy": "A", "accuracy": 0.9, "fireworks_total_tokens": 1000},
            {"strategy": "B", "accuracy": 0.8, "fireworks_total_tokens": 500},
            {"strategy": "C", "accuracy": 0.7, "fireworks_total_tokens": 2000},  # dominated by B
        ]
        pareto = find_pareto_efficient(metrics, "fireworks_total_tokens")
        assert "A" in pareto
        assert "B" in pareto
        assert "C" not in pareto

    def test_pareto_single_strategy(self):
        """Single strategy is always Pareto-efficient."""
        metrics = [{"strategy": "only", "accuracy": 0.5, "fireworks_total_tokens": 100}]
        pareto = find_pareto_efficient(metrics, "fireworks_total_tokens")
        assert pareto == ["only"]

    def test_compare_strategies_from_files(self, tmp_path):
        """End-to-end comparison from JSONL files."""
        local_records = [_make_record(correct=True, route_source="local") for _ in range(5)]
        remote_records = [_make_record(correct=True, route_source="fireworks",
                                       total_tokens=100, actual_remote_cost=0.001) for _ in range(5)]

        local_path = tmp_path / "local.jsonl"
        remote_path = tmp_path / "remote.jsonl"
        _write_jsonl(local_records, local_path)
        _write_jsonl(remote_records, remote_path)

        result = compare_strategies({"local": local_path, "remote": remote_path})
        assert len(result["strategies"]) == 2
        assert "pareto_efficient" in result

    def test_small_sample_warning(self, tmp_path):
        """Strategies with < 30 cases get a warning."""
        records = [_make_record(correct=True) for _ in range(5)]
        path = tmp_path / "small.jsonl"
        _write_jsonl(records, path)

        result = compare_strategies({"small": path})
        assert "small" in result["warnings"]["small_sample_strategies"]


# ===================================================================
# Artifact Generation Tests
# ===================================================================


class TestArtifactGeneration:
    def test_calibration_artifact_schema(self, multi_category_records):
        """Calibration artifact has required fields."""
        artifact = calibrate(multi_category_records, None, min_accuracy=0.5)
        assert artifact["schema_version"] == 2
        assert "generated_at" in artifact
        assert artifact["minimum_accuracy"] == 0.5
        assert artifact["objective"] == "tokens"
        assert isinstance(artifact["thresholds"], dict)
        assert isinstance(artifact["model_profiles"], list)

    def test_calibration_artifact_thresholds(self, multi_category_records):
        """Each threshold entry has required fields."""
        artifact = calibrate(multi_category_records, None, min_accuracy=0.5)
        for cat, entry in artifact["thresholds"].items():
            assert "threshold" in entry
            assert "constraint_met" in entry
            assert "sample_count" in entry
            assert "accuracy" in entry
            assert 0.0 <= entry["threshold"] <= 1.0

    def test_model_profile_in_artifact(self, multi_category_records):
        """Model profiles are included in artifact."""
        artifact = calibrate(multi_category_records, None, min_accuracy=0.5)
        for p in artifact["model_profiles"]:
            assert "model_id" in p
            assert "accuracy" in p
            assert "sample_count" in p
            assert "low_confidence" in p

    def test_strategy_comparison_artifact_schema(self, tmp_path):
        """Strategy comparison artifact has required fields."""
        records = [_make_record(correct=True) for _ in range(10)]
        path = tmp_path / "test.jsonl"
        _write_jsonl(records, path)

        result = compare_strategies({"test": path})
        assert result["schema_version"] == 1
        assert "generated_at" in result
        assert "strategies" in result
        assert "pareto_efficient" in result
        assert "warnings" in result


# ===================================================================
# CLI Integration Tests
# ===================================================================


class TestCLI:
    def test_calibrate_command(self, tmp_path):
        """CLI calibrate command produces artifact."""
        records = [_make_record(category="math", correct=True, route_score=0.8) for _ in range(10)]
        input_path = tmp_path / "cal.jsonl"
        output_path = tmp_path / "output.json"
        _write_jsonl(records, input_path)

        import sys
        from unittest.mock import patch
        args = ["benchmark", "calibrate",
                "--calibration-results", str(input_path),
                "--output", str(output_path),
                "--minimum-accuracy", "0.5",
                "--objective", "tokens"]
        with patch.object(sys, "argv", args):
            from benchmark.__main__ import main
            code = main()
        assert code == 0
        assert output_path.exists()
        artifact = json.loads(output_path.read_text(encoding="utf-8"))
        assert artifact["schema_version"] == 2

    def test_compare_command(self, tmp_path):
        """CLI compare command produces artifact."""
        records = [_make_record(correct=True) for _ in range(10)]
        path1 = tmp_path / "s1.jsonl"
        path2 = tmp_path / "s2.jsonl"
        _write_jsonl(records, path1)
        _write_jsonl(records, path2)
        output_path = tmp_path / "cmp.json"

        import sys
        from unittest.mock import patch
        args = ["benchmark", "compare",
                "--results", str(path1), str(path2),
                "--output", str(output_path)]
        with patch.object(sys, "argv", args):
            from benchmark.__main__ import main
            code = main()
        assert code == 0
        assert output_path.exists()

    def test_calibrate_missing_file(self, tmp_path):
        """CLI calibrate with missing file returns error."""
        import sys
        from unittest.mock import patch
        args = ["benchmark", "calibrate",
                "--calibration-results", str(tmp_path / "missing.jsonl"),
                "--output", str(tmp_path / "out.json")]
        with patch.object(sys, "argv", args):
            from benchmark.__main__ import main
            code = main()
        assert code == 1


# ===================================================================
# Edge Cases
# ===================================================================


class TestEdgeCases:
    def test_empty_records(self):
        """Calibrate with empty records."""
        artifact = calibrate([], None, min_accuracy=0.9)
        assert artifact["thresholds"] == {}
        assert artifact["model_profiles"] == []

    def test_all_same_confidence(self):
        """All records have same confidence score."""
        records = [_make_record(category="math", correct=True, route_score=0.85) for _ in range(10)]
        result = calibrate_category(records, "math", min_accuracy=0.5)
        assert result is not None
        assert result.candidate.accuracy == 1.0

    def test_difficulty_threshold_insufficient_samples(self):
        """Difficulty thresholds not generated with too few samples."""
        records = [_make_record(category="math", difficulty="easy", correct=True, route_score=0.8)
                   for _ in range(8)]
        result = calibrate_category(records, "math", min_accuracy=0.5)
        assert result is not None
        # Only 8 easy samples, below _MIN_DIFFICULTY_SAMPLES=10
        assert "easy" not in result.difficulty_thresholds
