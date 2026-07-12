from analyzers.task_analyzer import TaskAnalyzer
from decision.decision_engine import DecisionEngine
from state import RoutingState


def test_task_analyzer_flags_legal_queries_as_force_remote():
    query = (
        "Can you review this contract clause and explain any potential "
        "copyright implications?"
    )
    state = TaskAnalyzer().analyze(RoutingState(query=query))

    assert state.force_remote
    assert state.task_type == "legal"


def test_task_analyzer_does_not_force_remote_for_basic_code():
    query = "Write a function to add two numbers."
    state = TaskAnalyzer().analyze(RoutingState(query=query))

    assert not state.force_remote
    assert state.task_type == "code_generation"


def test_task_analyzer_flags_code_generation_as_force_remote():
    query = "Write a Python function to parse a CSV file."
    state = TaskAnalyzer().analyze(RoutingState(query=query))

    assert not state.force_remote
    assert state.task_type == "code_generation"


def test_task_analyzer_does_not_force_remote_for_basic_bug_fix():
    query = "Fix: IndexError at line 42"
    state = TaskAnalyzer().analyze(RoutingState(query=query))

    assert not state.force_remote
    assert state.task_type == "code_debug"
