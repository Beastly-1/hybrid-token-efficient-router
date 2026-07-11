"""Structured JSONL logger for routing events."""

import json
import time
from pathlib import Path
from typing import Optional

from config import LOG_DIR, PROJECT_ROOT
from state import RoutingState


class RoutingLogger:
    """Appends one JSON object per routing request to a JSONL file."""

    def __init__(self, log_path: Optional[Path] = None):
        self._log_path = log_path or (PROJECT_ROOT / LOG_DIR / "routing.jsonl")

    def log_event(
        self,
        state: RoutingState,
        route_source: str,
        total_latency_ms: float,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Write one routing event. Never raises."""
        try:
            record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "task_id": state.task_id or None,
                "task_type": state.task_type,
                "difficulty": state.difficulty,
                "tool_candidate": state.tool_candidate,
                "tool_used": state.tool_used,
                "route_source": route_source,
                "use_remote": state.use_remote,
                "selected_model": state.selected_model or None,
                "route_score": state.route_score,
                "routing_reason": state.routing_reason or None,
                "estimated_remote_cost": state.estimated_remote_cost or None,
                "total_latency_ms": round(total_latency_ms, 2),
                "remote_prompt_tokens": state.remote_prompt_tokens,
                "remote_completion_tokens": state.remote_completion_tokens,
                "remote_total_tokens": state.remote_total_tokens,
                "remote_latency_ms": state.remote_latency_ms,
                "remote_usage_source": state.remote_usage_source,
                "actual_remote_cost": state.actual_remote_cost,
                "remote_cost_source": state.remote_cost_source,
                "success": success,
                "error_type": error,
            }
            self._write(record)
        except Exception:
            pass

    def _write(self, record: dict) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
