from analyzers.pre_router import PreRouterPipeline
from state import RoutingState


def test_pre_router_prepares_code_request():
    state = PreRouterPipeline().prepare(
        RoutingState(query="Write a Python function to calculate factorial.")
    )

    assert state.input_valid
    assert state.task_type == "code_generation"
    assert state.difficulty == "hard"
    assert state.max_new_tokens > 0
    assert "runnable code" in state.system_prompt
    assert "Current date is" in state.system_prompt
    assert "Current time is" in state.system_prompt
    assert "Current datetime is" in state.system_prompt
    assert state.current_date
    assert state.current_time
    assert state.current_datetime


def test_pre_router_stops_empty_input():
    state = PreRouterPipeline().prepare(RoutingState(query="  "))

    assert not state.input_valid
    assert state.final_answer == "Query is empty."


def test_pre_router_uses_local_task_classifier(monkeypatch):
    class FakeClassifier:
        def classify(self, state):
            state.task_type = "code_generation"
            state.tool_candidate = "python_executor"
            return state

    monkeypatch.setattr(
        "analyzers.pre_router.LocalTaskClassifier",
        lambda: FakeClassifier(),
    )

    state = PreRouterPipeline().prepare(
        RoutingState(query="Write a Python script to sort a list.")
    )

    assert state.task_type == "code_generation"
    assert state.tool_candidate == "python_executor"
