from analyzers.pre_router import PreRouterPipeline
from decision.decision_engine import DecisionEngine
from decision.model_selector import ModelSelector
from evaluators.confidence import ConfidenceEvaluator
from models.local_model import LocalModel
from models.remote_model import RemoteModel
from tools.tool_router import ToolRouter
from config import LOCAL_MODEL_ENABLED


class HybridRouter:
    """Uses zero-token local work first, then the cheapest sufficient fallback."""

    def __init__(self):
        self.pre_router = PreRouterPipeline()
        self.tool_router = ToolRouter()
        self.local_model = LocalModel()
        self.confidence = ConfidenceEvaluator()
        self.decision = DecisionEngine()
        self.model_selector = ModelSelector()
        self.remote_model = RemoteModel()

    def route(self, state):
        state = self.pre_router.prepare(state)

        if not state.input_valid:
            state.use_remote = False
            return state

        state = self.tool_router.execute(state)

        if state.tool_used is not None:
            state.use_remote = False
            state.routing_reason = f"Answered using tool: {state.tool_used}"
            return state

        if LOCAL_MODEL_ENABLED:
            try:
                state = self.local_model.generate(state)
                state = self.confidence.evaluate(state)
            except RuntimeError as error:
                state.local_runtime_error = str(error)

        state = self.decision.decide(state)

        if not state.use_remote:
            return state

        state = self.model_selector.select(state)
        state = self.remote_model.generate(state)
        state.final_answer = state.remote_answer

        return state