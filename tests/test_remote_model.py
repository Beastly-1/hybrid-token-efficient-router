from models.remote_model import RemoteModel
from state import RoutingState


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "final answer"}}]}


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
