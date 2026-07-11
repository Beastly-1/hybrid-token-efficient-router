"""Tests for final calibration integration: synthetic safety, splits, run-all-models."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from decision.model_selector import ModelSelector
from state import RoutingState


# ===================================================================
# Synthetic Artifact Safety
# ===================================================================


class TestSyntheticArtifactSafety:
    def _write_artifact(self, tmp_path, data_source):
        artifact = {
            "schema_version": 2,
            "data_source": data_source,
            "model_profiles": [
                {"model_id": "accounts/fireworks/models/model-a", "category": "math",
                 "difficulty": "easy", "sample_count": 50, "accuracy": 0.95,
                 "average_cost": 0.001, "average_latency_ms": 50, "p95_latency_ms": 80,
                 "low_confidence": False},
            ],
            "thresholds": {},
        }
        path = tmp_path / "cal.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        return path

    def test_synthetic_rejected_by_default(self, tmp_path):
        """Synthetic artifact is rejected when ALLOW_SYNTHETIC_CALIBRATION is not set."""
        cal_path = self._write_artifact(tmp_path, "synthetic")
        catalog = [{"id": "accounts/fireworks/models/model-a", "input_cost_per_million": 0.2,
                    "output_cost_per_million": 0.2, "max_difficulty": "hard", "task_types": []}]
        allowed = ["accounts/fireworks/models/model-a"]

        with patch.dict(os.environ, {"ALLOW_SYNTHETIC_CALIBRATION": "false"}):
            selector = ModelSelector(catalog=catalog, allowed_models=allowed, calibration_path=cal_path)

        state = RoutingState(query="test", task_type="math", difficulty="easy")
        selector.select(state)
        # Should use static policy since synthetic is rejected
        assert "static_policy" in selector.selection_reason

    def test_synthetic_allowed_with_env_flag(self, tmp_path):
        """Synthetic artifact is accepted when ALLOW_SYNTHETIC_CALIBRATION=true."""
        cal_path = self._write_artifact(tmp_path, "synthetic")
        catalog = [{"id": "accounts/fireworks/models/model-a", "input_cost_per_million": 0.2,
                    "output_cost_per_million": 0.2, "max_difficulty": "hard", "task_types": []}]
        allowed = ["accounts/fireworks/models/model-a"]

        with patch.dict(os.environ, {"ALLOW_SYNTHETIC_CALIBRATION": "true"}):
            selector = ModelSelector(catalog=catalog, allowed_models=allowed, calibration_path=cal_path)

        state = RoutingState(query="test", task_type="math", difficulty="easy")
        selector.select(state)
        assert "calibrated" in selector.selection_reason

    def test_measured_artifact_accepted(self, tmp_path):
        """Measured artifact is always accepted."""
        cal_path = self._write_artifact(tmp_path, "measured")
        catalog = [{"id": "accounts/fireworks/models/model-a", "input_cost_per_million": 0.2,
                    "output_cost_per_million": 0.2, "max_difficulty": "hard", "task_types": []}]
        allowed = ["accounts/fireworks/models/model-a"]

        with patch.dict(os.environ, {"ALLOW_SYNTHETIC_CALIBRATION": "false"}):
            selector = ModelSelector(catalog=catalog, allowed_models=allowed, calibration_path=cal_path)

        state = RoutingState(query="test", task_type="math", difficulty="easy")
        selector.select(state)
        assert "calibrated" in selector.selection_reason

    def test_missing_data_source_accepted(self, tmp_path):
        """Artifact without data_source field is accepted (legacy)."""
        artifact = {"schema_version": 2, "model_profiles": [
            {"model_id": "accounts/fireworks/models/model-a", "category": "math",
             "difficulty": "easy", "sample_count": 50, "accuracy": 0.9,
             "average_cost": 0.001, "average_latency_ms": 50, "p95_latency_ms": 80,
             "low_confidence": False}
        ], "thresholds": {}}
        path = tmp_path / "cal.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")

        catalog = [{"id": "accounts/fireworks/models/model-a", "input_cost_per_million": 0.2,
                    "output_cost_per_million": 0.2, "max_difficulty": "hard", "task_types": []}]
        allowed = ["accounts/fireworks/models/model-a"]
        selector = ModelSelector(catalog=catalog, allowed_models=allowed, calibration_path=path)

        state = RoutingState(query="test", task_type="math", difficulty="easy")
        selector.select(state)
        assert "calibrated" in selector.selection_reason


# ===================================================================
# Dataset Split
# ===================================================================


class TestDatasetSplit:
    def test_split_reproducible(self, tmp_path):
        """Same seed produces same split."""
        dataset_path = Path("benchmarks/development/tasks.jsonl")
        if not dataset_path.exists():
            pytest.skip("Development dataset not found")

        cal1 = tmp_path / "cal1.jsonl"
        test1 = tmp_path / "test1.jsonl"
        cal2 = tmp_path / "cal2.jsonl"
        test2 = tmp_path / "test2.jsonl"

        args_base = ["benchmark", "split-dataset", "--input", str(dataset_path), "--seed", "42"]

        with patch.object(sys, "argv", args_base + ["--calibration-output", str(cal1), "--test-output", str(test1)]):
            from benchmark.__main__ import main
            assert main() == 0

        with patch.object(sys, "argv", args_base + ["--calibration-output", str(cal2), "--test-output", str(test2)]):
            from benchmark.__main__ import main
            assert main() == 0

        assert cal1.read_text() == cal2.read_text()
        assert test1.read_text() == test2.read_text()

    def test_no_duplicate_ids_across_splits(self, tmp_path):
        """No task_id appears in both calibration and test."""
        dataset_path = Path("benchmarks/development/tasks.jsonl")
        if not dataset_path.exists():
            pytest.skip("Development dataset not found")

        cal_path = tmp_path / "cal.jsonl"
        test_path = tmp_path / "test.jsonl"

        with patch.object(sys, "argv", ["benchmark", "split-dataset",
                                         "--input", str(dataset_path),
                                         "--calibration-output", str(cal_path),
                                         "--test-output", str(test_path),
                                         "--seed", "99"]):
            from benchmark.__main__ import main
            assert main() == 0

        cal_ids = {json.loads(l)["task_id"] for l in cal_path.read_text().splitlines() if l.strip()}
        test_ids = {json.loads(l)["task_id"] for l in test_path.read_text().splitlines() if l.strip()}
        assert cal_ids.isdisjoint(test_ids)
        assert len(cal_ids) + len(test_ids) == 48  # total dev cases

    def test_category_balance(self, tmp_path):
        """Both splits have cases from all categories."""
        dataset_path = Path("benchmarks/development/tasks.jsonl")
        if not dataset_path.exists():
            pytest.skip("Development dataset not found")

        cal_path = tmp_path / "cal.jsonl"
        test_path = tmp_path / "test.jsonl"

        with patch.object(sys, "argv", ["benchmark", "split-dataset",
                                         "--input", str(dataset_path),
                                         "--calibration-output", str(cal_path),
                                         "--test-output", str(test_path)]):
            from benchmark.__main__ import main
            assert main() == 0

        cal_cats = {json.loads(l)["category"] for l in cal_path.read_text().splitlines() if l.strip()}
        test_cats = {json.loads(l)["category"] for l in test_path.read_text().splitlines() if l.strip()}
        assert len(cal_cats) == 8
        assert len(test_cats) == 8


# ===================================================================
# Run-All-Models
# ===================================================================


class TestRunAllModels:
    def test_refuses_without_allow_remote(self, tmp_path):
        """run-all-models refuses without --allow-remote."""
        with patch.object(sys, "argv", ["benchmark", "run-all-models",
                                         "--dataset", "benchmarks/smoke/tasks.jsonl",
                                         "--output-dir", str(tmp_path / "out")]):
            from benchmark.__main__ import main
            assert main() == 1

    def test_refuses_without_api_key(self, tmp_path):
        """run-all-models refuses without FIREWORKS_API_KEY."""
        with patch("config.FIREWORKS_API_KEY", ""), \
             patch("config.ALLOWED_MODELS", ["model-a"]):
            with patch.object(sys, "argv", ["benchmark", "run-all-models",
                                             "--dataset", "benchmarks/smoke/tasks.jsonl",
                                             "--output-dir", str(tmp_path / "out"),
                                             "--allow-remote"]):
                from benchmark.__main__ import main
                assert main() == 1

    def test_refuses_empty_allowed_models(self, tmp_path):
        """run-all-models refuses when ALLOWED_MODELS is empty."""
        with patch("config.ALLOWED_MODELS", []), \
             patch("config.FIREWORKS_API_KEY", "test-key"):
            with patch.object(sys, "argv", ["benchmark", "run-all-models",
                                             "--dataset", "benchmarks/smoke/tasks.jsonl",
                                             "--output-dir", str(tmp_path / "out"),
                                             "--allow-remote"]):
                from benchmark.__main__ import main
                assert main() == 1


# ===================================================================
# Development Dataset Validation
# ===================================================================


class TestDevelopmentDataset:
    def test_validates(self):
        """Development dataset passes validation."""
        from benchmark.dataset import load_dataset
        cases = load_dataset(Path("benchmarks/development/tasks.jsonl"))
        assert len(cases) == 48

    def test_all_categories_present(self):
        """All 8 categories are represented."""
        from benchmark.dataset import load_dataset
        cases = load_dataset(Path("benchmarks/development/tasks.jsonl"))
        cats = {c.category for c in cases}
        assert cats == {"factual", "math", "sentiment", "summarization",
                        "ner", "code_debug", "logic", "code_generation"}

    def test_all_difficulties_present(self):
        """All 3 difficulties are represented."""
        from benchmark.dataset import load_dataset
        cases = load_dataset(Path("benchmarks/development/tasks.jsonl"))
        diffs = {c.difficulty for c in cases}
        assert diffs == {"easy", "medium", "hard"}

    def test_six_per_category(self):
        """Each category has exactly 6 cases."""
        from benchmark.dataset import load_dataset
        cases = load_dataset(Path("benchmarks/development/tasks.jsonl"))
        from collections import Counter
        counts = Counter(c.category for c in cases)
        assert all(v == 6 for v in counts.values())


# ===================================================================
# Production Integration
# ===================================================================


class TestProductionIntegration:
    def test_no_calibration_at_default_path_is_safe(self, tmp_path):
        """When no calibration file exists, selector uses static policy."""
        catalog = [{"id": "accounts/fireworks/models/model-a", "input_cost_per_million": 0.2,
                    "output_cost_per_million": 0.2, "max_difficulty": "hard", "task_types": []}]
        allowed = ["accounts/fireworks/models/model-a"]
        selector = ModelSelector(
            catalog=catalog, allowed_models=allowed,
            calibration_path=tmp_path / "nonexistent.json",
        )
        state = RoutingState(query="test", task_type="math", difficulty="easy")
        selector.select(state)
        assert state.selected_model == "accounts/fireworks/models/model-a"
        assert "static_policy" in selector.selection_reason

    def test_calibration_data_source_in_artifact(self):
        """calibrate() includes data_source in output."""
        from benchmark.calibration import calibrate
        records = [{"category": "math", "difficulty": "easy", "correct": True,
                    "route_score": 0.9, "scoring_status": "scored"} for _ in range(10)]
        artifact = calibrate(records, data_source="synthetic")
        assert artifact["data_source"] == "synthetic"

        artifact2 = calibrate(records, data_source="measured")
        assert artifact2["data_source"] == "measured"
