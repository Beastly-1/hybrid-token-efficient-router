"""Benchmark accuracy report generator.

Reads benchmark-result JSONL and produces a structured JSON report with
overall, per-category, per-difficulty, per-route, and per-model statistics.
"""

import json
import math
from pathlib import Path
from typing import Any, Optional


def _percentile(values: list[float], p: float) -> Optional[float]:
    """Compute percentile using nearest-rank method."""
    if not values:
        return None
    sorted_vals = sorted(values)
    k = max(0, min(len(sorted_vals) - 1, int(math.ceil(p / 100 * len(sorted_vals))) - 1))
    return sorted_vals[k]


def _safe_div(a: float, b: float) -> Optional[float]:
    """Safe division returning None on zero denominator."""
    return a / b if b > 0 else None


def load_results(path: Path) -> list[dict]:
    """Load benchmark result JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return records


def generate_report(records: list[dict]) -> dict:
    """Generate a comprehensive accuracy report from benchmark results."""
    report: dict[str, Any] = {}

    # Overall
    total = len(records)
    scored = [r for r in records if r.get("scoring_status") == "scored"]
    correct = [r for r in scored if r.get("correct") is True]
    invalid_answer = [r for r in records if r.get("scoring_status") == "invalid_answer"]
    unsupported = [r for r in records if r.get("scoring_status") == "unsupported"]
    failures = [r for r in records if r.get("success") is False]

    scores = [r["score"] for r in scored if r.get("score") is not None]

    report["overall"] = {
        "total_cases": total,
        "scored_cases": len(scored),
        "correct_cases": len(correct),
        "accuracy": _safe_div(len(correct), len(scored)),
        "mean_score": _safe_div(sum(scores), len(scores)) if scores else None,
        "invalid_answer_count": len(invalid_answer),
        "unsupported_count": len(unsupported),
        "execution_failure_count": len(failures),
    }

    # By category
    report["by_category"] = _by_group(records, "category")

    # By difficulty
    report["by_difficulty"] = _by_group_simple(records, "difficulty")

    # By route
    report["by_route"] = _by_route(records)

    # By model
    report["by_model"] = _by_model(records)

    # Efficiency metrics
    report["efficiency"] = _efficiency_metrics(records, scored, correct)

    return report


def _by_group(records: list[dict], key: str) -> dict:
    """Group by a key with full statistics."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        g = r.get(key, "unknown")
        groups.setdefault(g, []).append(r)

    result = {}
    for name, group in sorted(groups.items()):
        scored = [r for r in group if r.get("scoring_status") == "scored"]
        correct = [r for r in scored if r.get("correct") is True]
        scores = [r["score"] for r in scored if r.get("score") is not None]
        fw_calls = [r for r in group if r.get("route_source") == "fireworks"]
        fw_tokens = sum(r.get("total_tokens") or 0 for r in fw_calls)
        costs = [r.get("actual_remote_cost") for r in group if r.get("actual_remote_cost") is not None]
        latencies = [r.get("total_latency_ms") for r in group if r.get("total_latency_ms") is not None]

        result[name] = {
            "total": len(group),
            "correct": len(correct),
            "accuracy": _safe_div(len(correct), len(scored)),
            "mean_score": _safe_div(sum(scores), len(scores)) if scores else None,
            "fireworks_calls": len(fw_calls),
            "total_fireworks_tokens": fw_tokens,
            "actual_cost": sum(costs) if costs else None,
            "average_latency_ms": _safe_div(sum(latencies), len(latencies)) if latencies else None,
            "p95_latency_ms": _percentile(latencies, 95) if latencies else None,
        }
    return result


def _by_group_simple(records: list[dict], key: str) -> dict:
    """Group by a key with basic statistics."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        g = r.get(key, "unknown")
        groups.setdefault(g, []).append(r)

    result = {}
    for name, group in sorted(groups.items()):
        scored = [r for r in group if r.get("scoring_status") == "scored"]
        correct = [r for r in scored if r.get("correct") is True]
        scores = [r["score"] for r in scored if r.get("score") is not None]

        result[name] = {
            "total": len(group),
            "correct": len(correct),
            "accuracy": _safe_div(len(correct), len(scored)),
            "mean_score": _safe_div(sum(scores), len(scores)) if scores else None,
        }
    return result


def _by_route(records: list[dict]) -> dict:
    """Statistics by route source."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        route = r.get("route_source") or "unknown"
        groups.setdefault(route, []).append(r)

    result = {}
    for name, group in sorted(groups.items()):
        scored = [r for r in group if r.get("scoring_status") == "scored"]
        correct = [r for r in scored if r.get("correct") is True]
        scores = [r["score"] for r in scored if r.get("score") is not None]

        result[name] = {
            "count": len(group),
            "accuracy": _safe_div(len(correct), len(scored)),
            "mean_score": _safe_div(sum(scores), len(scores)) if scores else None,
        }
    return result


def _by_model(records: list[dict]) -> dict:
    """Statistics by selected model."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        model = r.get("selected_model")
        if not model:
            continue
        groups.setdefault(model, []).append(r)

    result = {}
    for name, group in sorted(groups.items()):
        scored = [r for r in group if r.get("scoring_status") == "scored"]
        correct = [r for r in scored if r.get("correct") is True]
        scores = [r["score"] for r in scored if r.get("score") is not None]
        prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in group)
        completion_tokens = sum(r.get("completion_tokens") or 0 for r in group)
        total_tokens = sum(r.get("total_tokens") or 0 for r in group)
        costs = [r.get("actual_remote_cost") for r in group if r.get("actual_remote_cost") is not None]
        latencies = [r.get("remote_latency_ms") for r in group if r.get("remote_latency_ms") is not None]

        result[name] = {
            "call_count": len(group),
            "accuracy": _safe_div(len(correct), len(scored)),
            "mean_score": _safe_div(sum(scores), len(scores)) if scores else None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "actual_cost": sum(costs) if costs else None,
            "average_remote_latency_ms": _safe_div(sum(latencies), len(latencies)) if latencies else None,
            "p95_remote_latency_ms": _percentile(latencies, 95) if latencies else None,
        }
    return result


def _efficiency_metrics(records: list[dict], scored: list[dict], correct: list[dict]) -> dict:
    """Compute efficiency metrics."""
    total = len(records)
    tool_cases = [r for r in records if r.get("route_source") == "tool"]
    local_cases = [r for r in records if r.get("route_source") == "local"]
    fw_cases = [r for r in records if r.get("route_source") == "fireworks"]

    tool_scored = [r for r in tool_cases if r.get("scoring_status") == "scored"]
    tool_correct = [r for r in tool_scored if r.get("correct") is True]

    local_scored = [r for r in local_cases if r.get("scoring_status") == "scored"]
    local_correct = [r for r in local_scored if r.get("correct") is True]

    fw_tokens_total = sum(r.get("total_tokens") or 0 for r in fw_cases)
    fw_cost_total = sum(r.get("actual_remote_cost") or 0 for r in fw_cases if r.get("actual_remote_cost") is not None)

    correct_count = len(correct)

    return {
        "tool_solve_rate": _safe_div(len(tool_correct), len(tool_scored)),
        "local_acceptance_rate": _safe_div(len(local_cases), total),
        "fireworks_call_rate": _safe_div(len(fw_cases), total),
        "fireworks_tokens_per_correct": _safe_div(fw_tokens_total, correct_count),
        "actual_cost_per_correct": _safe_div(fw_cost_total, correct_count),
    }
