import json
import time
from pathlib import Path

import pytest

from benchmark.logger import RoutingLogger
from router import HybridRouter
from state import RoutingState


# ===========================================================
# Helpers
# ===========================================================


class FakeLocalModel:
    def generate(self, state):
        state.local_answer = "A concise local answer."
        return state


class UnavailableLocalModel:
    def generate(self, state):
        raise RuntimeError("local model unavailable")


class FakeRemoteModel:
    def generate(self, state):
        state.remote_answer = "Fireworks answer"
        state.use_remote = True
        return state


def _read_events(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


# ===========================================================
# 1. Logger creates the parent directory
# ===========================================================


def test_logger_creates_parent_directory(tmp_path):
    log_file = tmp_path / "subdir" / "nested" / "routing.jsonl"
    logger = RoutingLogger(log_path=log_file)

    state = RoutingState(query="hello")
    logger.log_event(state, route_source="tool", total_latency_ms=1.0)

    assert log_file.exists()


# ===========================================================
# 2. Logger writes valid JSONL
# ===========================================================


def test_logger_writes_valid_jsonl(tmp_path):
    log_file = tmp_path / "routing.jsonl"
    logger = RoutingLogger(log_path=log_file)

    state = RoutingState(query="test", task_type="math", difficulty="easy")
    logger.log_event(state, route_source="tool", total_latency_ms=5.5)

    events = _read_events(log_file)
    assert len(events) == 1
    assert events[0]["task_type"] == "math"
    assert events[0]["route_source"] == "tool"


# ===========================================================
# 3. Multiple events are appended as separate lines
# ===========================================================


def test_multiple_events_appended(tmp_path):
    log_file = tmp_path / "routing.jsonl"
    logger = RoutingLogger(log_path=log_file)

    for i in range(3):
        state = RoutingState(query=f"q{i}", task_type="factual")
        logger.log_event(state, route_source="local", total_latency_ms=float(i))

    events = _read_events(log_file)
    assert len(events) == 3
    assert events[2]["total_latency_ms"] == 2.0


# ===========================================================
# 4. Optional missing values become null
# ===========================================================


def test_missing_values_are_null(tmp_path):
    log_file = tmp_path / "routing.jsonl"
    logger = RoutingLogger(log_path=log_file)

    state = RoutingState(query="q")
    logger.log_event(state, route_source="local", total_latency_ms=1.0)

    event = _read_events(log_file)[0]
    assert event["task_id"] is None
    assert event["tool_candidate"] is None
    assert event["tool_used"] is None
    assert event["selected_model"] is None
    assert event["error_type"] is None


# ===========================================================
# 5. Sensitive content is not written
# ===========================================================


def test_no_sensitive_content_in_log(tmp_path):
    log_file = tmp_path / "routing.jsonl"
    logger = RoutingLogger(log_path=log_file)

    state = RoutingState(
        query="Tell me a secret",
        system_prompt="You are a helpful assistant with key=sk-abc123",
    )
    state.local_answer = "The answer is 42"
    logger.log_event(state, route_source="local", total_latency_ms=1.0)

    raw = log_file.read_text(encoding="utf-8")
    assert "Tell me a secret" not in raw
    assert "sk-abc123" not in raw
    assert "The answer is 42" not in raw


# ===========================================================
# 6. Tool route produces exactly one event
# ===========================================================


def test_tool_route_one_event(tmp_path):
    log_file = tmp_path / "routing.jsonl"
    logger = RoutingLogger(log_path=log_file)

    router = HybridRouter(logger=logger)
    state = router.route(RoutingState(query="calculate 2 + 3 * 4"))

    events = _read_events(log_file)
    assert len(events) == 1
    assert events[0]["route_source"] == "tool"
    assert events[0]["tool_used"] == "calculator"
    assert events[0]["success"] is True


# ===========================================================
# 7. Accepted local route produces exactly one event
# ===========================================================


def test_local_route_one_event(tmp_path):
    log_file = tmp_path / "routing.jsonl"
    logger = RoutingLogger(log_path=log_file)

    router = HybridRouter(logger=logger)
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="What is the capital of France?"))

    events = _read_events(log_file)
    assert len(events) == 1
    assert events[0]["route_source"] == "local"
    assert events[0]["success"] is True


# ===========================================================
# 8. Fireworks route produces exactly one event
# ===========================================================


def test_fireworks_route_one_event(tmp_path):
    log_file = tmp_path / "routing.jsonl"
    logger = RoutingLogger(log_path=log_file)

    router = HybridRouter(logger=logger)
    router.local_model = UnavailableLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="What is the capital of France?"))

    events = _read_events(log_file)
    assert len(events) == 1
    assert events[0]["route_source"] == "fireworks"
    assert events[0]["use_remote"] is True


# ===========================================================
# 9. total_latency_ms is numeric and non-negative
# ===========================================================


def test_latency_is_numeric_and_non_negative(tmp_path):
    log_file = tmp_path / "routing.jsonl"
    logger = RoutingLogger(log_path=log_file)

    router = HybridRouter(logger=logger)
    router.route(RoutingState(query="calculate 1 + 1"))

    event = _read_events(log_file)[0]
    assert isinstance(event["total_latency_ms"], (int, float))
    assert event["total_latency_ms"] >= 0


# ===========================================================
# 10. Logging failure does not break routing
# ===========================================================


def test_logging_failure_does_not_break_route(tmp_path):
    # Point to an invalid path that cannot be created
    bad_path = Path("/\x00invalid/routing.jsonl")
    logger = RoutingLogger(log_path=bad_path)

    router = HybridRouter(logger=logger)
    state = router.route(RoutingState(query="calculate 2 + 2"))

    # Router still works correctly
    assert state.final_answer == "4"
    assert state.tool_used == "calculator"


# ===========================================================
# 11. Fireworks JSONL events contain remote telemetry fields
# ===========================================================


class FakeRemoteModelWithUsage:
    def generate(self, state):
        state.remote_answer = "Fireworks answer"
        state.use_remote = True
        state.remote_prompt_tokens = 50
        state.remote_completion_tokens = 25
        state.remote_total_tokens = 75
        state.remote_latency_ms = 120.5
        state.remote_usage_source = "api"
        return state


def test_fireworks_event_contains_remote_telemetry(tmp_path):
    log_file = tmp_path / "routing.jsonl"
    logger = RoutingLogger(log_path=log_file)

    router = HybridRouter(logger=logger)
    router.local_model = UnavailableLocalModel()
    router.remote_model = FakeRemoteModelWithUsage()

    router.route(RoutingState(query="What is the capital of France?"))

    event = _read_events(log_file)[0]
    assert event["remote_prompt_tokens"] == 50
    assert event["remote_completion_tokens"] == 25
    assert event["remote_total_tokens"] == 75
    assert event["remote_latency_ms"] == 120.5
    assert event["remote_usage_source"] == "api"


# ===========================================================
# 12. Tool and local events have null remote-token fields
# ===========================================================


def test_tool_event_has_null_remote_fields(tmp_path):
    log_file = tmp_path / "routing.jsonl"
    logger = RoutingLogger(log_path=log_file)

    router = HybridRouter(logger=logger)
    router.route(RoutingState(query="calculate 5 + 5"))

    event = _read_events(log_file)[0]
    assert event["remote_prompt_tokens"] is None
    assert event["remote_completion_tokens"] is None
    assert event["remote_total_tokens"] is None
    assert event["remote_latency_ms"] is None
    assert event["remote_usage_source"] is None


def test_local_event_has_null_remote_fields(tmp_path):
    log_file = tmp_path / "routing.jsonl"
    logger = RoutingLogger(log_path=log_file)

    router = HybridRouter(logger=logger)
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    router.route(RoutingState(query="What is the capital of France?"))

    event = _read_events(log_file)[0]
    assert event["remote_prompt_tokens"] is None
    assert event["remote_completion_tokens"] is None
    assert event["remote_total_tokens"] is None
    assert event["remote_latency_ms"] is None
    assert event["remote_usage_source"] is None
