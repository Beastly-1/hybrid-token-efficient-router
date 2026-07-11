"""Tests for the offline metrics aggregator."""

import json
import math
from pathlib import Path

import pytest

from benchmark.metrics import MetricsAggregator, percentile, _valid_numeric, _valid_token


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


def _tool_record(**overrides) -> dict:
    base = {
        "route_source": "tool",
        "task_type": "math",
        "difficulty": "easy",
        "tool_used": "calculator",
        "success": True,
        "total_latency_ms": 2.0,
        "remote_prompt_tokens": None,
        "remote_completion_tokens": None,
        "remote_total_tokens": None,
        "remote_latency_ms": None,
        "remote_usage_source": None,
        "estimated_remote_cost": None,
        "actual_remote_cost": None,
        "remote_cost_source": None,
        "selected_model": None,
    }
    base.update(overrides)
    return base


def _local_record(**overrides) -> dict:
    base = {
        "route_source": "local",
        "task_type": "factual",
        "difficulty": "easy",
        "success": True,
        "total_latency_ms": 50.0,
        "remote_prompt_tokens": None,
        "remote_completion_tokens": None,
        "remote_total_tokens": None,
        "remote_latency_ms": None,
        "remote_usage_source": None,
        "estimated_remote_cost": None,
        "actual_remote_cost": None,
        "remote_cost_source": None,
        "selected_model": None,
    }
    base.update(overrides)
    return base


def _fireworks_record(**overrides) -> dict:
    base = {
        "route_source": "fireworks",
        "task_type": "factual",
        "difficulty": "medium",
        "success": True,
        "total_latency_ms": 1200.0,
        "remote_prompt_tokens": 100,
        "remote_completion_tokens": 50,
        "remote_total_tokens": 150,
        "remote_latency_ms": 1100.0,
        "remote_usage_source": "api",
        "estimated_remote_cost": 0.00012,
        "actual_remote_cost": 0.00015,
        "remote_cost_source": "catalog_api_usage",
        "selected_model": "accounts/fireworks/models/gemma-4-31b-it",
    }
    base.update(overrides)
    return base


def _error_record(**overrides) -> dict:
    base = {
        "route_source": "error",
        "task_type": "general",
        "difficulty": "medium",
        "success": False,
        "total_latency_ms": 0.5,
        "remote_prompt_tokens": None,
        "remote_completion_tokens": None,
        "remote_total_tokens": None,
        "remote_latency_ms": None,
        "remote_usage_source": None,
        "estimated_remote_cost": None,
        "actual_remote_cost": None,
        "remote_cost_source": None,
        "selected_model": None,
    }
    base.update(overrides)
    return base


# ===========================================================
# 1. Empty JSONL file
# ===========================================================


def test_empty_file(tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text("", encoding="utf-8")

    summary = MetricsAggregator(f).aggregate()

    assert summary["records"]["valid"] == 0
    assert summary["records"]["malformed"] == 0
    assert summary["requests"]["total"] == 0
    assert summary["requests"]["failure_rate"] == 0.0
    assert summary["routes"]["fireworks"]["rate"] == 0.0
    assert summary["latency_ms"]["total"]["count"] == 0
    assert summary["latency_ms"]["total"]["average"] is None


# ===========================================================
# 2. Single valid record
# ===========================================================


def test_single_record(tmp_path):
    f = tmp_path / "single.jsonl"
    _write_jsonl(f, [_tool_record()])

    summary = MetricsAggregator(f).aggregate()

    assert summary["records"]["valid"] == 1
    assert summary["requests"]["total"] == 1
    assert summary["requests"]["successful"] == 1
    assert summary["routes"]["tool"]["count"] == 1
    assert summary["routes"]["tool"]["rate"] == 1.0


# ===========================================================
# 3. Blank lines are skipped
# ===========================================================


def test_blank_lines_skipped(tmp_path):
    f = tmp_path / "blanks.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write(json.dumps(_tool_record()) + "\n")
        fh.write("   \n")
        fh.write(json.dumps(_local_record()) + "\n")
        fh.write("\n")

    summary = MetricsAggregator(f).aggregate()

    assert summary["records"]["valid"] == 2
    assert summary["records"]["malformed"] == 0
    assert summary["records"]["total_lines"] == 5


# ===========================================================
# 4. Malformed JSON lines are counted and skipped
# ===========================================================


def test_malformed_lines(tmp_path):
    f = tmp_path / "malformed.jsonl"
    _write_jsonl(f, [
        _tool_record(),
        "not valid json {{{",
        _local_record(),
        "also broken",
    ])

    summary = MetricsAggregator(f).aggregate()

    assert summary["records"]["valid"] == 2
    assert summary["records"]["malformed"] == 2
    assert summary["requests"]["total"] == 2


# ===========================================================
# 5. Tool, local, Fireworks and error routes counted correctly
# ===========================================================


def test_route_counts(tmp_path):
    f = tmp_path / "routes.jsonl"
    _write_jsonl(f, [
        _tool_record(),
        _tool_record(),
        _local_record(),
        _fireworks_record(),
        _fireworks_record(),
        _fireworks_record(),
        _error_record(),
    ])

    summary = MetricsAggregator(f).aggregate()

    assert summary["routes"]["tool"]["count"] == 2
    assert summary["routes"]["local"]["count"] == 1
    assert summary["routes"]["fireworks"]["count"] == 3
    assert summary["routes"]["error"]["count"] == 1


# ===========================================================
# 6. Route rates are calculated correctly
# ===========================================================


def test_route_rates(tmp_path):
    f = tmp_path / "rates.jsonl"
    _write_jsonl(f, [_tool_record()] * 3 + [_fireworks_record()] * 7)

    summary = MetricsAggregator(f).aggregate()

    assert summary["routes"]["tool"]["rate"] == pytest.approx(0.3)
    assert summary["routes"]["fireworks"]["rate"] == pytest.approx(0.7)


# ===========================================================
# 7. Fireworks call count and rate are correct
# ===========================================================


def test_fireworks_call_count_and_rate(tmp_path):
    f = tmp_path / "fw.jsonl"
    _write_jsonl(f, [_tool_record()] * 2 + [_fireworks_record()] * 3)

    summary = MetricsAggregator(f).aggregate()

    assert summary["fireworks"]["call_count"] == 3
    assert summary["fireworks"]["call_rate"] == pytest.approx(0.6)


# ===========================================================
# 8. Prompt, completion and total tokens summed correctly
# ===========================================================


def test_token_sums(tmp_path):
    f = tmp_path / "tokens.jsonl"
    _write_jsonl(f, [
        _fireworks_record(remote_prompt_tokens=100, remote_completion_tokens=50, remote_total_tokens=150),
        _fireworks_record(remote_prompt_tokens=200, remote_completion_tokens=80, remote_total_tokens=280),
    ])

    summary = MetricsAggregator(f).aggregate()

    assert summary["fireworks"]["prompt_tokens"] == 300
    assert summary["fireworks"]["completion_tokens"] == 130
    assert summary["fireworks"]["total_tokens"] == 430


# ===========================================================
# 9. Missing usage is counted correctly
# ===========================================================


def test_missing_usage_counted(tmp_path):
    f = tmp_path / "usage.jsonl"
    _write_jsonl(f, [
        _fireworks_record(remote_prompt_tokens=100, remote_completion_tokens=50, remote_total_tokens=150),
        _fireworks_record(remote_prompt_tokens=None, remote_completion_tokens=None, remote_total_tokens=None),
    ])

    summary = MetricsAggregator(f).aggregate()

    assert summary["fireworks"]["usage_available_count"] == 1
    assert summary["fireworks"]["usage_missing_count"] == 1


# ===========================================================
# 10. Tool and local records do not contribute Fireworks tokens
# ===========================================================


def test_non_fireworks_no_tokens(tmp_path):
    f = tmp_path / "no_fw.jsonl"
    _write_jsonl(f, [_tool_record(), _local_record()])

    summary = MetricsAggregator(f).aggregate()

    assert summary["fireworks"]["prompt_tokens"] == 0
    assert summary["fireworks"]["completion_tokens"] == 0
    assert summary["fireworks"]["total_tokens"] == 0


# ===========================================================
# 11. Estimated and actual costs summed separately
# ===========================================================


def test_costs_summed_separately(tmp_path):
    f = tmp_path / "costs.jsonl"
    _write_jsonl(f, [
        _fireworks_record(estimated_remote_cost=0.001, actual_remote_cost=0.0012),
        _fireworks_record(estimated_remote_cost=0.002, actual_remote_cost=0.0025),
    ])

    summary = MetricsAggregator(f).aggregate()

    assert summary["fireworks"]["estimated_cost_total"] == pytest.approx(0.003)
    assert summary["fireworks"]["actual_cost_total"] == pytest.approx(0.0037)


# ===========================================================
# 12. Missing actual cost is not treated as zero
# ===========================================================


def test_missing_actual_cost_not_zero(tmp_path):
    f = tmp_path / "missing_cost.jsonl"
    _write_jsonl(f, [
        _fireworks_record(actual_remote_cost=0.001),
        _fireworks_record(actual_remote_cost=None, remote_cost_source="unavailable"),
    ])

    summary = MetricsAggregator(f).aggregate()

    assert summary["fireworks"]["actual_cost_total"] == pytest.approx(0.001)
    assert summary["fireworks"]["actual_cost_available_count"] == 1
    assert summary["fireworks"]["actual_cost_unavailable_count"] == 1


# ===========================================================
# 13. Explicit actual cost 0.0 is accepted
# ===========================================================


def test_zero_actual_cost_accepted(tmp_path):
    f = tmp_path / "zero_cost.jsonl"
    _write_jsonl(f, [
        _fireworks_record(actual_remote_cost=0.0, remote_cost_source="catalog_api_usage"),
    ])

    summary = MetricsAggregator(f).aggregate()

    assert summary["fireworks"]["actual_cost_total"] == 0.0
    assert summary["fireworks"]["actual_cost_available_count"] == 1
    assert summary["fireworks"]["actual_cost_unavailable_count"] == 0


# ===========================================================
# 14. NaN, infinity, negative values and booleans rejected
# ===========================================================


def test_invalid_numerics_rejected(tmp_path):
    f = tmp_path / "invalid.jsonl"
    _write_jsonl(f, [
        _fireworks_record(
            total_latency_ms=float("nan"),
            remote_latency_ms=float("inf"),
            actual_remote_cost=-1.0,
            remote_prompt_tokens=True,
            remote_completion_tokens=-5,
        ),
    ])

    summary = MetricsAggregator(f).aggregate()

    assert summary["latency_ms"]["total"]["count"] == 0
    assert summary["latency_ms"]["fireworks"]["count"] == 0
    assert summary["fireworks"]["prompt_tokens"] == 0
    assert summary["fireworks"]["completion_tokens"] == 0
    assert summary["fireworks"]["actual_cost_total"] == 0.0
    assert summary["fireworks"]["actual_cost_unavailable_count"] == 1


# ===========================================================
# 15. Average latency is correct
# ===========================================================


def test_average_latency(tmp_path):
    f = tmp_path / "lat.jsonl"
    _write_jsonl(f, [
        _tool_record(total_latency_ms=10.0),
        _tool_record(total_latency_ms=20.0),
        _tool_record(total_latency_ms=30.0),
    ])

    summary = MetricsAggregator(f).aggregate()

    assert summary["latency_ms"]["total"]["average"] == pytest.approx(20.0)


# ===========================================================
# 16. p50 and p95 are correct
# ===========================================================


def test_percentiles_empty():
    assert percentile([], 50) is None
    assert percentile([], 95) is None


def test_percentiles_one_value():
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 95) == 42.0


def test_percentiles_two_values():
    s = [10.0, 20.0]
    p50 = percentile(s, 50)
    assert p50 is not None
    assert 10.0 <= p50 <= 20.0


def test_percentiles_multiple_unsorted():
    values = sorted([100.0, 10.0, 50.0, 80.0, 20.0, 90.0, 30.0, 60.0, 70.0, 40.0])
    p50 = percentile(values, 50)
    p95 = percentile(values, 95)
    assert p50 is not None
    assert p95 is not None
    # p50 should be around the median
    assert 40.0 <= p50 <= 60.0
    # p95 should be near the top
    assert p95 >= 90.0


def test_percentiles_in_summary(tmp_path):
    f = tmp_path / "perc.jsonl"
    records = [_tool_record(total_latency_ms=float(i)) for i in range(1, 101)]
    _write_jsonl(f, records)

    summary = MetricsAggregator(f).aggregate()

    assert summary["latency_ms"]["total"]["p50"] == pytest.approx(50.5, abs=1.0)
    assert summary["latency_ms"]["total"]["p95"] == pytest.approx(95.5, abs=1.0)


# ===========================================================
# 17. Missing latency values do not crash aggregation
# ===========================================================


def test_missing_latency_no_crash(tmp_path):
    f = tmp_path / "no_lat.jsonl"
    _write_jsonl(f, [
        _fireworks_record(total_latency_ms=None, remote_latency_ms=None),
        _fireworks_record(total_latency_ms=100.0, remote_latency_ms=80.0),
    ])

    summary = MetricsAggregator(f).aggregate()

    assert summary["latency_ms"]["total"]["count"] == 1
    assert summary["latency_ms"]["fireworks"]["count"] == 1


# ===========================================================
# 18. Per-task-type statistics are correct
# ===========================================================


def test_per_task_type(tmp_path):
    f = tmp_path / "tasks.jsonl"
    _write_jsonl(f, [
        _tool_record(task_type="math"),
        _fireworks_record(task_type="math", remote_prompt_tokens=50, remote_completion_tokens=20, remote_total_tokens=70),
        _local_record(task_type="sentiment"),
        _fireworks_record(task_type="sentiment", remote_prompt_tokens=80, remote_completion_tokens=30, remote_total_tokens=110),
    ])

    summary = MetricsAggregator(f).aggregate()

    assert "math" in summary["by_task_type"]
    assert summary["by_task_type"]["math"]["total"] == 2
    assert summary["by_task_type"]["math"]["tool_routes"] == 1
    assert summary["by_task_type"]["math"]["fireworks_routes"] == 1
    assert summary["by_task_type"]["math"]["prompt_tokens"] == 50

    assert "sentiment" in summary["by_task_type"]
    assert summary["by_task_type"]["sentiment"]["total"] == 2
    assert summary["by_task_type"]["sentiment"]["local_routes"] == 1
    assert summary["by_task_type"]["sentiment"]["fireworks_routes"] == 1


# ===========================================================
# 19. Per-model statistics are correct
# ===========================================================


def test_per_model(tmp_path):
    f = tmp_path / "models.jsonl"
    _write_jsonl(f, [
        _fireworks_record(
            selected_model="accounts/fireworks/models/gemma-4-31b-it",
            remote_prompt_tokens=100,
            remote_completion_tokens=50,
            remote_total_tokens=150,
            actual_remote_cost=0.001,
        ),
        _fireworks_record(
            selected_model="accounts/fireworks/models/minimax-m3",
            remote_prompt_tokens=200,
            remote_completion_tokens=80,
            remote_total_tokens=280,
            actual_remote_cost=0.002,
        ),
    ])

    summary = MetricsAggregator(f).aggregate()

    gemma = summary["by_model"]["accounts/fireworks/models/gemma-4-31b-it"]
    assert gemma["call_count"] == 1
    assert gemma["prompt_tokens"] == 100
    assert gemma["actual_cost_total"] == pytest.approx(0.001)

    minimax = summary["by_model"]["accounts/fireworks/models/minimax-m3"]
    assert minimax["call_count"] == 1
    assert minimax["prompt_tokens"] == 200


# ===========================================================
# 20. Exact selected model IDs are preserved
# ===========================================================


def test_exact_model_ids_preserved(tmp_path):
    f = tmp_path / "ids.jsonl"
    long_id = "accounts/fireworks/models/gemma-4-31b-it-nvfp4"
    _write_jsonl(f, [_fireworks_record(selected_model=long_id)])

    summary = MetricsAggregator(f).aggregate()

    assert long_id in summary["by_model"]


# ===========================================================
# 21. Missing selected model on Fireworks event is counted
# ===========================================================


def test_missing_model_counted(tmp_path):
    f = tmp_path / "no_model.jsonl"
    _write_jsonl(f, [
        _fireworks_record(selected_model=None),
        _fireworks_record(selected_model=""),
    ])

    summary = MetricsAggregator(f).aggregate()

    assert summary["fireworks"]["missing_model_count"] == 2
    assert summary["by_model"] == {}


# ===========================================================
# 22. Output JSON is valid
# ===========================================================


def test_output_json_valid(tmp_path):
    f = tmp_path / "valid.jsonl"
    _write_jsonl(f, [_tool_record(), _fireworks_record()])

    summary = MetricsAggregator(f).aggregate()
    output = json.dumps(summary)

    # Should round-trip cleanly
    parsed = json.loads(output)
    assert parsed["schema_version"] == 1


# ===========================================================
# 23. Output parent directory is created
# ===========================================================


def test_output_dir_created(tmp_path):
    f = tmp_path / "input.jsonl"
    _write_jsonl(f, [_tool_record()])

    output = tmp_path / "sub" / "deep" / "summary.json"

    summary = MetricsAggregator(f).aggregate()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary), encoding="utf-8")

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1


# ===========================================================
# 24. Missing input file produces a clear failure
# ===========================================================


def test_missing_input_file(tmp_path):
    missing = tmp_path / "nonexistent.jsonl"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        MetricsAggregator(missing)


# ===========================================================
# Additional: _valid_numeric and _valid_token edge cases
# ===========================================================


def test_valid_numeric_rejects_bool():
    assert _valid_numeric(True) is None
    assert _valid_numeric(False) is None


def test_valid_numeric_rejects_nan():
    assert _valid_numeric(float("nan")) is None


def test_valid_numeric_rejects_inf():
    assert _valid_numeric(float("inf")) is None


def test_valid_numeric_rejects_negative():
    assert _valid_numeric(-1.0) is None


def test_valid_numeric_accepts_zero():
    assert _valid_numeric(0.0) == 0.0


def test_valid_token_rejects_bool():
    assert _valid_token(True) is None
    assert _valid_token(False) is None


def test_valid_token_rejects_negative():
    assert _valid_token(-1) is None


def test_valid_token_accepts_zero():
    assert _valid_token(0) == 0
