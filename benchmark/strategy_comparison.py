"""Strategy comparison: compare benchmark result files across strategies.

Computes per-strategy metrics and identifies Pareto-efficient strategies
for accuracy vs tokens, cost, and latency.
"""

import json
import math
from pathlib import Path
from typing import Any, Optional


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(math.ceil(p / 100 * len(s))) - 1))
    return s[k]


def _safe_div(a: float, b: float) -> Optional[float]:
    return a / b if b > 0 else None


def load_strategy_results(path: Path) -> list[dict]:
    """Load a single strategy result JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    records.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
    return records


def compute_strategy_metrics(records: list[dict], strategy_name: str) -> dict:
    """Compute comprehensive metrics for a single strategy."""
    total = len(records)
    if total == 0:
        return {"strategy": strategy_name, "total_cases": 0}

    scored = [r for r in records if r.get("scoring_status") == "scored"]
    correct = [r for r in scored if r.get("correct") is True]
    scores = [r["score"] for r in scored if r.get("score") is not None]

    # Fireworks metrics
    fw_records = [r for r in records if r.get("route_source") == "fireworks"]
    fw_prompt = sum(r.get("prompt_tokens") or 0 for r in fw_records)
    fw_completion = sum(r.get("completion_tokens") or 0 for r in fw_records)
    fw_total = sum(r.get("total_tokens") or 0 for r in fw_records)

    # Cost
    actual_costs = [r.get("actual_remote_cost") for r in records if r.get("actual_remote_cost") is not None]
    estimated_costs = [r.get("estimated_remote_cost") for r in records if r.get("estimated_remote_cost") is not None]
    actual_cost_sum = sum(actual_costs) if actual_costs else 0.0
    estimated_cost_sum = sum(estimated_costs) if estimated_costs else 0.0

    # Latency
    latencies = [r.get("total_latency_ms") for r in records if r.get("total_latency_ms") is not None]

    # Route distribution
    route_dist = {}
    for r in records:
        src = r.get("route_source") or "unknown"
        route_dist[src] = route_dist.get(src, 0) + 1

    # Model usage
    model_dist = {}
    for r in records:
        model = r.get("selected_model")
        if model:
            model_dist[model] = model_dist.get(model, 0) + 1

    # Per-category accuracy
    by_category = {}
    cats = sorted(set(r.get("category", "unknown") for r in records))
    for cat in cats:
        cat_recs = [r for r in records if r.get("category") == cat]
        cat_scored = [r for r in cat_recs if r.get("scoring_status") == "scored"]
        cat_correct = [r for r in cat_scored if r.get("correct") is True]
        by_category[cat] = {
            "total": len(cat_recs),
            "accuracy": _safe_div(len(cat_correct), len(cat_scored)),
        }

    correct_count = len(correct)
    accuracy = _safe_div(correct_count, len(scored))

    return {
        "strategy": strategy_name,
        "total_cases": total,
        "scored_cases": len(scored),
        "correct_cases": correct_count,
        "accuracy": accuracy,
        "mean_score": _safe_div(sum(scores), len(scores)) if scores else None,
        "fireworks_call_count": len(fw_records),
        "fireworks_call_rate": _safe_div(len(fw_records), total),
        "fireworks_prompt_tokens": fw_prompt,
        "fireworks_completion_tokens": fw_completion,
        "fireworks_total_tokens": fw_total,
        "actual_cost": actual_cost_sum,
        "estimated_cost": estimated_cost_sum,
        "average_latency_ms": _safe_div(sum(latencies), len(latencies)) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 95),
        "route_distribution": route_dist,
        "model_usage": model_dist,
        "tokens_per_correct": _safe_div(fw_total, correct_count),
        "cost_per_correct": _safe_div(actual_cost_sum, correct_count),
        "by_category": by_category,
    }


def _is_dominated(point: dict, others: list[dict], accuracy_key: str, cost_key: str) -> bool:
    """Check if point is dominated (another has >= accuracy AND <= cost)."""
    a = point.get(accuracy_key) or 0.0
    c = point.get(cost_key) or 0.0
    for other in others:
        if other["strategy"] == point["strategy"]:
            continue
        oa = other.get(accuracy_key) or 0.0
        oc = other.get(cost_key) or 0.0
        if oa >= a and oc <= c and (oa > a or oc < c):
            return True
    return False


def find_pareto_efficient(metrics_list: list[dict], cost_key: str) -> list[str]:
    """Find Pareto-efficient strategies for accuracy vs a cost metric."""
    valid = [m for m in metrics_list if m.get("accuracy") is not None and m.get(cost_key) is not None]
    if not valid:
        return []
    return [
        m["strategy"] for m in valid
        if not _is_dominated(m, valid, "accuracy", cost_key)
    ]


def compare_strategies(result_files: dict[str, Path]) -> dict:
    """Compare multiple strategy result files.

    Args:
        result_files: mapping of strategy_name -> Path to JSONL results

    Returns:
        Comparison artifact dict.
    """
    all_metrics = []
    for name, path in result_files.items():
        records = load_strategy_results(path)
        metrics = compute_strategy_metrics(records, name)
        all_metrics.append(metrics)

    # Pareto frontiers
    pareto_tokens = find_pareto_efficient(all_metrics, "fireworks_total_tokens")
    pareto_cost = find_pareto_efficient(all_metrics, "actual_cost")
    pareto_latency = find_pareto_efficient(all_metrics, "average_latency_ms")

    # Small sample warning
    small_sample = [m["strategy"] for m in all_metrics if (m.get("total_cases") or 0) < 30]

    import time
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "strategies": all_metrics,
        "pareto_efficient": {
            "accuracy_vs_tokens": pareto_tokens,
            "accuracy_vs_cost": pareto_cost,
            "accuracy_vs_latency": pareto_latency,
        },
        "warnings": {
            "small_sample_strategies": small_sample,
            "note": "Statistical significance not claimed for small sample sizes."
        },
    }
