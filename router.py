from analyzers.pre_router import PreRouterPipeline
from decision.decision_engine import DecisionEngine
from decision.model_selector import ModelSelector
from evaluators.confidence import ConfidenceEvaluator
from models.local_model import LocalModel
from models.remote_model import RemoteModel
from tools.tool_router import ToolRouter
from config import LOCAL_MODEL_ENABLED


class HybridRouter:
<<<<<<< HEAD
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
    Remote Model
    """

    def __init__(self):

        self.analyzer = TaskAnalyzer()
=======
    """Uses zero-token local work first, then the cheapest sufficient fallback."""

    def __init__(self):
        self.pre_router = PreRouterPipeline()
>>>>>>> origin/with-step-1-again
        self.tool_router = ToolRouter()
        self.local_model = LocalModel()
        self.confidence = ConfidenceEvaluator()
        self.decision = DecisionEngine()
<<<<<<< HEAD
        self.remote_model = RemoteModel()

    def route(self, state):

        # ---------------------------------------
        # Step 1 : Analyze task
        # ---------------------------------------

        state = self.analyzer.analyze(state)

        print("\n========== TASK ANALYZER ==========")
        print("Task Type      :", state.task_type)
        print("Difficulty     :", state.difficulty)
        print("Tool Candidate :", state.tool_candidate)

        # ---------------------------------------
        # Step 2 : Deterministic tools
        # ---------------------------------------

        state = self.tool_router.execute(state)

        if state.tool_used is not None:

            print("Tool Used      :", state.tool_used)

            state.use_remote = False
            state.routing_reason = (
                f"Answered using tool: {state.tool_used}"
            )

            return state

        # ---------------------------------------
        # Step 3 : Local Model
        # ---------------------------------------

        print("\nCalling Local Model...")

        state = self.local_model.generate(state)

        print("Local Answer:")
        print(state.local_answer)

        print("\nLogprobs:")
        print(state.logprobs)

        # ---------------------------------------
        # Step 4 : Confidence
        # ---------------------------------------

        state = self.confidence.evaluate(state)

        print("\nRoute Score:", state.route_score)

        # ---------------------------------------
        # Step 5 : Decision
        # ---------------------------------------
=======
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
>>>>>>> origin/with-step-1-again

        state = self.decision.decide(state)
        if not state.use_remote:
            return state

<<<<<<< HEAD
        print("Use Remote:", state.use_remote)

        # ---------------------------------------
        # Step 6 : Remote Model
        # ---------------------------------------

        if state.use_remote:

            print("\nCalling Fireworks...")

            state = self.remote_model.generate(state)
            state.final_answer = state.remote_answer

        else:

            state.final_answer = state.local_answer

        return state
=======
        state = self.model_selector.select(state)
        state = self.remote_model.generate(state)
        state.final_answer = state.remote_answer
        return state
>>>>>>> origin/with-step-1-again
