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
    Remote Model
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

        state = self.decision.decide(state)

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