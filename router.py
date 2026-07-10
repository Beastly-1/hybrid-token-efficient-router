from analyzers.task_analyzer import TaskAnalyzer
from tools.tool_router import ToolRouter
from models.local_model import LocalModel
from evaluators.confidence import ConfidenceEvaluator
from decision.decision_engine import DecisionEngine
from models.remote_model import RemoteModel


class HybridRouter:
    """
    Complete routing pipeline.

    Query
      ↓
    Task Analyzer
      ↓
    Tool Router
      ↓
    Local Model
      ↓
    Confidence Evaluator
      ↓
    Decision Engine
      ↓
    Remote Model (Fireworks)
    """

    def __init__(self):

        self.analyzer = TaskAnalyzer()

        self.tool_router = ToolRouter()

        self.local_model = LocalModel()

        self.confidence = ConfidenceEvaluator()

        self.decision = DecisionEngine()

        self.remote_model = RemoteModel()

    def route(self, state):

        # ---------------------------------------
        # Step 1 : Analyze task
        # ---------------------------------------

        state = self.analyzer.analyze(state)

        # ---------------------------------------
        # Step 2 : Try deterministic tools first
        # ---------------------------------------

        state = self.tool_router.execute(state)

        if state.final_answer is not None:

            state.use_remote = False

            state.routing_reason = (
                f"Answered using tool: {state.tool_used}"
            )

            return state

        # ---------------------------------------
        # Step 3 : Local model
        # ---------------------------------------

        state = self.local_model.generate(state)

        # ---------------------------------------
        # Step 4 : Confidence evaluation
        # ---------------------------------------

        state = self.confidence.evaluate(state)

        # ---------------------------------------
        # Step 5 : Local vs Fireworks
        # ---------------------------------------

        state = self.decision.decide(state)

        # ---------------------------------------
        # Step 6 : Fireworks
        # ---------------------------------------

        if state.use_remote:

            state = self.remote_model.generate(state)

            state.final_answer = state.remote_answer

        else:

            state.final_answer = state.local_answer

        return state