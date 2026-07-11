"""Offline aggregation of routing JSONL logs into a telemetry summary."""

import json
import math
from pathlib import Path
from typing import Optional


def _valid_numeric(value) -> Optional[float]:
    """Return value as float only if finite and non-negative, not a bool."""
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value) if value >= 0 else None


def _valid_token(value) -> Optional[int]:
    """Return value only if it is a non-negative integer, not a bool."""
    if isinstance(value, bool):
        return None
    if not isinstance(value, int):
        return None
    return value if value >= 0 else None


def percentile(sorted_values: list[float], p: float) -> Optional[float]:
    """Nearest-rank percentile on a pre-sorted list.

    Uses the 'exclusive' interpolation method:
    rank = p/100 * (n + 1), then linearly interpolate between adjacent values.
    For edge cases, clamp to the first or last value.
    """
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return sorted_values[0]
    rank = p / 100.0 * (n + 1)
    if rank <= 1:
        return sorted_values[0]
    if rank >= n:
        return sorted_values[n - 1]
    lower = int(rank) - 1
    frac = rank - int(rank)
    return sorted_values[lower] + frac * (sorted_values[lower + 1] - sorted_values[lower])


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _latency_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "average": None, "p50": None, "p95": None}
    s = sorted(values)
    return {
        "count": len(s),
        "average": sum(s) / len(s),
        "p50": percentile(s, 50),
        "p95": percentile(s, 95),
    }


class MetricsAggregator:
    """Reads a routing JSONL file and produces a telemetry summary."""

    def __init__(self, input_path: Path):
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")
        self._input_path = input_path

    def aggregate(self) -> dict:
        records = []
        malformed = 0
        total_lines = 0

        with self._input_path.open("r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError:
                    malformed += 1

        valid = len(records)
        total_requests = valid

        # Success/failure
        successful = sum(1 for r in records if r.get("success") is True)
        failed = sum(1 for r in records if r.get("success") is False)

        # Routes
        route_counts: dict[str, int] = {}
        for r in records:
            src = r.get("route_source", "unknown") or "unknown"
            route_counts[src] = route_counts.get(src, 0) + 1

        tool_count = route_counts.get("tool", 0)
        local_count = route_counts.get("local", 0)
        fireworks_count = route_counts.get("fireworks", 0)
        error_count = route_counts.get("error", 0)

        # Fireworks token aggregation
        prompt_tokens_total = 0
        completion_tokens_total = 0
        fw_total_tokens_total = 0
        usage_available = 0
        usage_missing = 0

        # Fireworks cost aggregation
        estimated_cost_total = 0.0
        actual_cost_total = 0.0
        actual_cost_available = 0
        actual_cost_unavailable = 0

        # Latency
        all_latencies: list[float] = []
        fw_latencies: list[float] = []

        # Per-task
        by_task: dict[str, dict] = {}
        # Per-model
        by_model: dict[str, dict] = {}
        fw_missing_model = 0

        for r in records:
            route_source = r.get("route_source", "unknown") or "unknown"

            # Total latency
            lat = _valid_numeric(r.get("total_latency_ms"))
            if lat is not None:
                all_latencies.append(lat)

            # Task type grouping
            task_type = r.get("task_type") or "unknown"
            if not isinstance(task_type, str) or not task_type.strip():
                task_type = "unknown"
            if task_type not in by_task:
                by_task[task_type] = self._empty_task_stats()
            ts = by_task[task_type]
            ts["total"] += 1
            if r.get("success") is True:
                ts["successful"] += 1
            elif r.get("success") is False:
                ts["failed"] += 1
            ts[f"{route_source}_routes"] = ts.get(f"{route_source}_routes", 0) + 1
            if lat is not None:
                ts["_latencies"].append(lat)

            # Fireworks-specific
            if route_source == "fireworks":
                # Remote latency
                rlat = _valid_numeric(r.get("remote_latency_ms"))
                if rlat is not None:
                    fw_latencies.append(rlat)

                # Tokens
                pt = _valid_token(r.get("remote_prompt_tokens"))
                ct = _valid_token(r.get("remote_completion_tokens"))
                tt = _valid_token(r.get("remote_total_tokens"))

                if pt is not None or ct is not None or tt is not None:
                    usage_available += 1
                    if pt is not None:
                        prompt_tokens_total += pt
                    if ct is not None:
                        completion_tokens_total += ct
                    if tt is not None:
                        fw_total_tokens_total += tt
                else:
                    usage_missing += 1

                # Estimated cost
                ec = _valid_numeric(r.get("estimated_remote_cost"))
                if ec is not None:
                    estimated_cost_total += ec

                # Actual cost
                ac = _valid_numeric(r.get("actual_remote_cost"))
                cost_source = r.get("remote_cost_source")
                if ac is not None:
                    actual_cost_total += ac
                    actual_cost_available += 1
                else:
                    actual_cost_unavailable += 1

                # Per-task Fireworks stats
                if pt is not None:
                    ts["prompt_tokens"] += pt
                if ct is not None:
                    ts["completion_tokens"] += ct
                if tt is not None:
                    ts["total_tokens"] += tt
                if ec is not None:
                    ts["estimated_cost_total"] += ec
                if ac is not None:
                    ts["actual_cost_total"] += ac

                # Per-model
                model_id = r.get("selected_model")
                if model_id and isinstance(model_id, str) and model_id.strip():
                    if model_id not in by_model:
                        by_model[model_id] = self._empty_model_stats()
                    ms = by_model[model_id]
                    ms["call_count"] += 1
                    if r.get("success") is True:
                        ms["successful"] += 1
                    elif r.get("success") is False:
                        ms["failed"] += 1
                    if pt is not None:
                        ms["prompt_tokens"] += pt
                    if ct is not None:
                        ms["completion_tokens"] += ct
                    if tt is not None:
                        ms["total_tokens"] += tt
                    if ec is not None:
                        ms["estimated_cost_total"] += ec
                    if ac is not None:
                        ms["actual_cost_total"] += ac
                        ms["actual_cost_available"] += 1
                    else:
                        ms["actual_cost_unavailable"] += 1
                    if pt is not None or ct is not None or tt is not None:
                        ms["usage_available"] += 1
                    else:
                        ms["usage_missing"] += 1
                    if rlat is not None:
                        ms["_latencies"].append(rlat)
                else:
                    fw_missing_model += 1

        # Finalize per-task
        by_task_final = {}
        for task, ts in by_task.items():
            lats = ts.pop("_latencies")
            ts["fireworks_call_rate"] = _safe_rate(
                ts.get("fireworks_routes", 0), ts["total"]
            )
            lat_stats = _latency_stats(lats)
            ts["average_total_latency_ms"] = lat_stats["average"]
            ts["p95_total_latency_ms"] = lat_stats["p95"]
            # Clean up internal route keys into standard form
            ts["tool_routes"] = ts.get("tool_routes", 0)
            ts["local_routes"] = ts.get("local_routes", 0)
            ts["fireworks_routes"] = ts.get("fireworks_routes", 0)
            ts["error_routes"] = ts.get("error_routes", 0)
            by_task_final[task] = ts

        # Finalize per-model
        by_model_final = {}
        for model_id, ms in by_model.items():
            lats = ms.pop("_latencies")
            lat_stats = _latency_stats(lats)
            ms["average_remote_latency_ms"] = lat_stats["average"]
            ms["p95_remote_latency_ms"] = lat_stats["p95"]
            by_model_final[model_id] = ms

        return {
            "schema_version": 1,
            "source_file": str(self._input_path),
            "records": {
                "valid": valid,
                "malformed": malformed,
                "total_lines": total_lines,
            },
            "requests": {
                "total": total_requests,
                "successful": successful,
                "failed": failed,
                "failure_rate": _safe_rate(failed, total_requests),
            },
            "routes": {
                "tool": {"count": tool_count, "rate": _safe_rate(tool_count, total_requests)},
                "local": {"count": local_count, "rate": _safe_rate(local_count, total_requests)},
                "fireworks": {"count": fireworks_count, "rate": _safe_rate(fireworks_count, total_requests)},
                "error": {"count": error_count, "rate": _safe_rate(error_count, total_requests)},
            },
            "fireworks": {
                "call_count": fireworks_count,
                "call_rate": _safe_rate(fireworks_count, total_requests),
                "prompt_tokens": prompt_tokens_total,
                "completion_tokens": completion_tokens_total,
                "total_tokens": fw_total_tokens_total,
                "usage_available_count": usage_available,
                "usage_missing_count": usage_missing,
                "estimated_cost_total": estimated_cost_total,
                "actual_cost_total": actual_cost_total,
                "actual_cost_available_count": actual_cost_available,
                "actual_cost_unavailable_count": actual_cost_unavailable,
                "missing_model_count": fw_missing_model,
            },
            "latency_ms": {
                "total": _latency_stats(all_latencies),
                "fireworks": _latency_stats(fw_latencies),
            },
            "by_task_type": by_task_final,
            "by_model": by_model_final,
        }

    @staticmethod
    def _empty_task_stats() -> dict:
        return {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "tool_routes": 0,
            "local_routes": 0,
            "fireworks_routes": 0,
            "error_routes": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_total": 0.0,
            "actual_cost_total": 0.0,
            "_latencies": [],
        }

    @staticmethod
    def _empty_model_stats() -> dict:
        return {
            "call_count": 0,
            "successful": 0,
            "failed": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_total": 0.0,
            "actual_cost_total": 0.0,
            "actual_cost_available": 0,
            "actual_cost_unavailable": 0,
            "usage_available": 0,
            "usage_missing": 0,
            "_latencies": [],
        }
