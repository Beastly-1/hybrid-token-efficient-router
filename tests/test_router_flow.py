from router import HybridRouter
from state import RoutingState


class FakeRemoteModel:
    def generate(self, state):
        state.remote_answer = "Fireworks answer"
        state.use_remote = True
        return state


class FakeLocalModel:
    def generate(self, state):
        state.local_answer = "A concise local answer."
        return state


class UnavailableLocalModel:
    def generate(self, state):
        raise RuntimeError("local model unavailable")


def test_router_sends_non_tool_request_to_final_model():
    router = HybridRouter()
    router.local_model = UnavailableLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="What is the capital of France?"))

    assert state.use_remote
    assert state.final_answer == "Fireworks answer"
    assert state.task_type == "factual"


def test_router_keeps_accepted_easy_answer_local():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="What is the capital of France?"))

    assert not state.use_remote
    assert state.final_answer == "A concise local answer."
