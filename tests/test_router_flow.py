from datetime import datetime

from router import HybridRouter
from state import RoutingState
import pytest


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


class FailingRemoteModel:
    def generate(self, state):
        raise RuntimeError("provider unavailable")


class RecordingLogger:
    def __init__(self):
        self.events = []

    def log_event(self, state, **kwargs):
        self.events.append(kwargs)


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


def test_router_logs_remote_generation_failures():
    logger = RecordingLogger()
    router = HybridRouter(logger=logger)
    router.local_model = UnavailableLocalModel()
    router.remote_model = FailingRemoteModel()

    with pytest.raises(RuntimeError, match="provider unavailable"):
        router.route(RoutingState(query="Explain recursion."))

    assert logger.events[-1]["route_source"] == "error"
    assert logger.events[-1]["success"] is False
    assert logger.events[-1]["error"] == "RuntimeError"


def test_router_handles_day_of_week_datetime_tool():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="What day of week is 2026-07-09?"))

    assert not state.use_remote
    assert state.tool_used == "datetime"
    assert state.final_answer == "Thursday"


def test_router_handles_today_datetime_tool():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="What is today's date?"))

    assert not state.use_remote
    assert state.tool_used == "datetime"
    assert state.final_answer == datetime.now().date().isoformat()


def test_router_handles_current_time_phrasing():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="What is the time right now?"))

    assert not state.use_remote
    assert state.tool_used == "datetime"
    assert len(state.final_answer) == 8


def test_router_handles_whats_the_date_phrasing():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="What's the date today?"))

    assert not state.use_remote
    assert state.tool_used == "datetime"
    assert state.final_answer == datetime.now().date().isoformat()


def test_router_handles_word_based_addition():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="What is 2 plus 2?"))

    assert not state.use_remote
    assert state.tool_used == "calculator"
    assert state.final_answer == "4"


def test_router_handles_word_based_division():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="How much is 6 divided by 3?"))

    assert not state.use_remote
    assert state.tool_used == "calculator"
    assert state.final_answer == "2.0"


def test_router_handles_check_email_phrasing():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="Check email abc@gmail.com"))

    assert not state.use_remote
    assert state.tool_used == "regex_verifier"
    assert state.final_answer == "Valid Email"


def test_router_handles_is_this_url_valid_phrasing():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="Is this URL valid https://openai.com"))

    assert not state.use_remote
    assert state.tool_used == "regex_verifier"
    assert state.final_answer == "Valid URL"


def test_router_handles_check_json_phrasing():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query='Check JSON {"name":"Jane","age":22}'))

    assert not state.use_remote
    assert state.tool_used == "json_validator"
    assert state.final_answer.startswith("Valid JSON")


def test_router_handles_date_validation_phrasing():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="Is this date valid 2026-07-12?"))

    assert not state.use_remote
    assert state.tool_used == "regex_verifier"
    assert state.final_answer == "Valid Date"


def test_router_handles_time_validation_phrasing():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="Is this time valid 17:15:33?"))

    assert not state.use_remote
    assert state.tool_used == "regex_verifier"
    assert state.final_answer == "Valid Time"


def test_router_handles_date_conversion_phrasing():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="Convert this date 2026-07-12"))

    assert not state.use_remote
    assert state.tool_used == "datetime"
    assert state.final_answer == "2026-07-12"


def test_router_handles_percent_math():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="What is 20% of 50?"))

    assert not state.use_remote
    assert state.tool_used == "calculator"
    assert state.final_answer == "10.0"


def test_router_handles_email_validation_with_punctuation():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="Please validate email abc@gmail.com."))

    assert not state.use_remote
    assert state.tool_used == "regex_verifier"
    assert state.final_answer == "Valid Email"


def test_router_handles_url_validation_with_trailing_mark():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(
        RoutingState(query="Can you verify url https://openai.com?")
    )

    assert not state.use_remote
    assert state.tool_used == "regex_verifier"
    assert state.final_answer == "Valid URL"


def test_router_handles_basic_addition_as_calculator():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="What is 2+2?"))

    assert not state.use_remote
    assert state.tool_used == "calculator"
    assert state.final_answer == "4"


def test_router_handles_basic_multiplication_as_calculator():
    router = HybridRouter()
    router.local_model = FakeLocalModel()
    router.remote_model = FakeRemoteModel()

    state = router.route(RoutingState(query="How much is 6 * 7?"))

    assert not state.use_remote
    assert state.tool_used == "calculator"
    assert state.final_answer == "42"
