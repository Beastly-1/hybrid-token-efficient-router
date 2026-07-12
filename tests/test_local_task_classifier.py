from analyzers.local_task_classifier import LocalTaskClassifier
from state import RoutingState


class FakeLocalModel:
    def generate(self, state):
        state.local_answer = (
            '{"task_type":"code_generation", "tool_candidate":"python_executor", "difficulty":"hard"}'
        )
        return state


def test_local_task_classifier_refines_factual_state():
    classifier = LocalTaskClassifier(local_model=FakeLocalModel())
    state = RoutingState(query="Write a Python script to sort a list.")
    state.task_type = "factual"
    state.tool_candidate = None
    state.difficulty = "medium"

    state = classifier.classify(state)

    assert state.task_type == "code_generation"
    assert state.tool_candidate == "python_executor"
    assert state.difficulty == "hard"
