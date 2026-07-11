import time
from typing import Optional

from analyzers.pre_router import PreRouterPipeline
from benchmark.logger import RoutingLogger
from config import LOCAL_MODEL_ENABLED
from decision.decision_engine import DecisionEngine
from decision.model_selector import ModelSelector
from evaluators.confidence import ConfidenceEvaluator
from models.local_model import LocalModel
from models.remote_model import RemoteModel
from state import RoutingState
from tools.tool_router import ToolRouter


class HybridRouter:
    """Uses zero-token local work first, then the cheapest sufficient fallback."""

    def __init__(self, logger: Optional[RoutingLogger] = None):
        self.pre_router = PreRouterPipeline()
        self.tool_router = ToolRouter()
        self.local_model = LocalModel()
        self.confidence = ConfidenceEvaluator()
        self.decision = DecisionEngine()
        self.model_selector = ModelSelector()
        self.remote_model = RemoteModel()
        self._logger = logger or RoutingLogger()

    def route(self, state: RoutingState) -> RoutingState:
        start_time = time.perf_counter()

        state = self.pre_router.prepare(state)

        if not state.input_valid:
            state.use_remote = False
            return self._finalize_route(state, start_time, "error", error="invalid_input")

        state = self.tool_router.execute(state)

        if state.tool_used is not None:
            state.use_remote = False
            state.routing_reason = f"Answered using tool: {state.tool_used}"
            return self._finalize_route(state, start_time, "tool")

        if LOCAL_MODEL_ENABLED:
            try:
                state = self.local_model.generate(state)
                state = self.confidence.evaluate(state)
            except RuntimeError as error:
                state.local_runtime_error = str(error)

        state = self.decision.decide(state)

        if not state.use_remote:
            return self._finalize_route(state, start_time, "local")

        state = self.model_selector.select(state)
        state = self.remote_model.generate(state)
        state.final_answer = state.remote_answer
        self._calculate_actual_cost(state)

        return self._finalize_route(state, start_time, "fireworks")

    def _calculate_actual_cost(self, state: RoutingState) -> None:
        """Set actual_remote_cost from catalog pricing and API token counts."""
        prompt = state.remote_prompt_tokens
        completion = state.remote_completion_tokens

        if (
            not isinstance(prompt, int)
            or isinstance(prompt, bool)
            or prompt < 0
            or not isinstance(completion, int)
            or isinstance(completion, bool)
            or completion < 0
            or not state.selected_model
        ):
            state.actual_remote_cost = None
            state.remote_cost_source = "unavailable"
            return

        cost = self.model_selector.calculate_actual_cost(
            state.selected_model, prompt, completion
        )
        if cost is None:
            state.actual_remote_cost = None
            state.remote_cost_source = "unavailable"
        else:
            state.actual_remote_cost = cost
            state.remote_cost_source = "catalog_api_usage"

    def _finalize_route(
        self,
        state: RoutingState,
        start_time: float,
        route_source: str,
        success: bool = True,
        error: Optional[str] = None,
    ) -> RoutingState:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._logger.log_event(
            state,
            route_source=route_source,
            total_latency_ms=elapsed_ms,
            success=success if error is None else False,
            error=error,
        )
        return state
