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


def test_pre_router_stops_empty_input():
    state = PreRouterPipeline().prepare(RoutingState(query="  "))

    assert not state.input_valid
    assert state.final_answer == "Query is empty."
