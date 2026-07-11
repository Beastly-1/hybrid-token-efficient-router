import pytest

from models.remote_model import RemoteModel
from state import RoutingState


# ===========================================================
# Helpers
# ===========================================================


class FakeResponse:
    def __init__(self, body=None):
        self._body = body or {"choices": [{"message": {"content": "final answer"}}]}

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class FailingResponse:
    def raise_for_status(self):
        import requests
        raise requests.HTTPError("500 Server Error")

    def json(self):
        return {}


# ===========================================================
# Existing tests (preserved)
# ===========================================================


def test_remote_model_sends_analysis_and_returns_final_answer(monkeypatch):
    monkeypatch.setattr("models.remote_model.FIREWORKS_API_KEY", "test-key")
    captured = {}

    def post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    state = RoutingState(
        query="Explain recursion.",
        system_prompt="Be concise.",
        task_type="factual",
        difficulty="easy",
        max_new_tokens=96,
    )

    state = RemoteModel(post=post).generate(state)

    assert state.remote_answer == "final answer"
    assert state.use_remote
    assert captured["json"]["messages"][1]["content"] == "Explain recursion."
    assert "task type: factual" in captured["json"]["messages"][0]["content"]


def test_remote_model_includes_local_tool_result(monkeypatch):
    monkeypatch.setattr("models.remote_model.FIREWORKS_API_KEY", "test-key")
    captured = {}

    def post(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    state = RoutingState(
        query="What is 2 + 2?",
        system_prompt="Be concise.",
        tool_used="calculator",
        final_answer="4",
        max_new_tokens=96,
    )

    RemoteModel(post=post).generate(state)

    system_message = captured["json"]["messages"][0]["content"]
    assert "deterministic tool: calculator" in system_message
    assert "deterministic tool result: 4" in system_message


# ===========================================================
# 1. Full API usage is captured correctly
# ===========================================================


def test_full_usage_captured(monkeypatch):
    monkeypatch.setattr("models.remote_model.FIREWORKS_API_KEY", "test-key")

    body = {
        "choices": [{"message": {"content": "answer"}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }

    state = RoutingState(query="q", max_new_tokens=96)
    state = RemoteModel(post=lambda *a, **k: FakeResponse(body)).generate(state)

    assert state.remote_prompt_tokens == 100
    assert state.remote_completion_tokens == 50
    assert state.remote_total_tokens == 150
    assert state.remote_usage_source == "api"


# ===========================================================
# 2. Missing total_tokens is derived
# ===========================================================


def test_missing_total_tokens_derived(monkeypatch):
    monkeypatch.setattr("models.remote_model.FIREWORKS_API_KEY", "test-key")

    body = {
        "choices": [{"message": {"content": "answer"}}],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 20,
        },
    }

    state = RoutingState(query="q", max_new_tokens=96)
    state = RemoteModel(post=lambda *a, **k: FakeResponse(body)).generate(state)

    assert state.remote_prompt_tokens == 80
    assert state.remote_completion_tokens == 20
    assert state.remote_total_tokens == 100
    assert state.remote_usage_source == "derived"


# ===========================================================
# 3. Completely missing usage
# ===========================================================


def test_missing_usage_produces_nulls(monkeypatch):
    monkeypatch.setattr("models.remote_model.FIREWORKS_API_KEY", "test-key")

    body = {"choices": [{"message": {"content": "answer"}}]}

    state = RoutingState(query="q", max_new_tokens=96)
    state = RemoteModel(post=lambda *a, **k: FakeResponse(body)).generate(state)

    assert state.remote_prompt_tokens is None
    assert state.remote_completion_tokens is None
    assert state.remote_total_tokens is None
    assert state.remote_usage_source == "missing"


# ===========================================================
# 4. Malformed usage does not break a valid answer
# ===========================================================


def test_malformed_usage_does_not_crash(monkeypatch):
    monkeypatch.setattr("models.remote_model.FIREWORKS_API_KEY", "test-key")

    body = {
        "choices": [{"message": {"content": "answer"}}],
        "usage": "not a dict",
    }

    state = RoutingState(query="q", max_new_tokens=96)
    state = RemoteModel(post=lambda *a, **k: FakeResponse(body)).generate(state)

    assert state.remote_answer == "answer"
    assert state.remote_usage_source == "missing"


# ===========================================================
# 5. Negative token values are rejected
# ===========================================================


def test_negative_tokens_rejected(monkeypatch):
    monkeypatch.setattr("models.remote_model.FIREWORKS_API_KEY", "test-key")

    body = {
        "choices": [{"message": {"content": "answer"}}],
        "usage": {
            "prompt_tokens": -5,
            "completion_tokens": 20,
            "total_tokens": -1,
        },
    }

    state = RoutingState(query="q", max_new_tokens=96)
    state = RemoteModel(post=lambda *a, **k: FakeResponse(body)).generate(state)

    assert state.remote_prompt_tokens is None
    assert state.remote_completion_tokens == 20
    assert state.remote_total_tokens is None  # negative total rejected


# ===========================================================
# 6. String token values are not accepted
# ===========================================================


def test_string_tokens_not_accepted(monkeypatch):
    monkeypatch.setattr("models.remote_model.FIREWORKS_API_KEY", "test-key")

    body = {
        "choices": [{"message": {"content": "answer"}}],
        "usage": {
            "prompt_tokens": "100",
            "completion_tokens": "50",
            "total_tokens": "150",
        },
    }

    state = RoutingState(query="q", max_new_tokens=96)
    state = RemoteModel(post=lambda *a, **k: FakeResponse(body)).generate(state)

    assert state.remote_prompt_tokens is None
    assert state.remote_completion_tokens is None
    assert state.remote_total_tokens is None


# ===========================================================
# 7. remote_latency_ms is numeric and non-negative
# ===========================================================


def test_remote_latency_is_numeric(monkeypatch):
    monkeypatch.setattr("models.remote_model.FIREWORKS_API_KEY", "test-key")

    state = RoutingState(query="q", max_new_tokens=96)
    state = RemoteModel(post=lambda *a, **k: FakeResponse()).generate(state)

    assert isinstance(state.remote_latency_ms, float)
    assert state.remote_latency_ms >= 0


# ===========================================================
# 10. Existing behaviour and returned answer unchanged
# ===========================================================


def test_answer_unchanged_with_usage(monkeypatch):
    monkeypatch.setattr("models.remote_model.FIREWORKS_API_KEY", "test-key")

    body = {
        "choices": [{"message": {"content": "  the answer  "}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    state = RoutingState(query="q", max_new_tokens=96)
    state = RemoteModel(post=lambda *a, **k: FakeResponse(body)).generate(state)

    assert state.remote_answer == "the answer"
    assert state.use_remote is True


# ===========================================================
# 11. Failed HTTP request records latency
# ===========================================================


def test_failed_request_records_latency(monkeypatch):
    monkeypatch.setattr("models.remote_model.FIREWORKS_API_KEY", "test-key")

    state = RoutingState(query="q", max_new_tokens=96)

    with pytest.raises(RuntimeError, match="Fireworks generation failed"):
        RemoteModel(post=lambda *a, **k: FailingResponse()).generate(state)

    assert isinstance(state.remote_latency_ms, float)
    assert state.remote_latency_ms >= 0
