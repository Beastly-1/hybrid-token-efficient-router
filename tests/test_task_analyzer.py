from analyzers.task_analyzer import TaskAnalyzer
from state import RoutingState


def test_task_analyzer_classifies_gdpr_business_query_as_factual():
    query = (
        "Explain the implications of the latest GDPR changes for a small e-commerce "
        "startup and suggest a compliance plan."
    )
    state = TaskAnalyzer().analyze(RoutingState(query=query))

    assert state.task_type == "factual"
    assert state.tool_candidate is None
    assert state.difficulty == "medium"


def test_task_analyzer_detects_math_by_whole_words_only():
    query = "What is cos 30 degrees?"
    state = TaskAnalyzer().analyze(RoutingState(query=query))

    assert state.task_type == "math"
    assert state.tool_candidate == "calculator"

