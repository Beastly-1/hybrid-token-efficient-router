from analyzers.task_analyzer import TaskAnalyzer
from models.local_model import LocalModel
from evaluators.confidence import ConfidenceEvaluator
from decision.decision_engine import DecisionEngine
from models.remote_model import RemoteModel


class HybridRouter:
    """
    Main routing pipeline.

    Query
      ↓
    Task Analyzer
      ↓
    Local Model
      ↓
    Confidence Evaluator
      ↓
    Decision Engine
      ↓
    Remote Model (if required)
      ↓
    Final Answer
    """

    def __init__(self):

        self.analyzer = TaskAnalyzer()
        self.local_model = LocalModel()
        self.confidence = ConfidenceEvaluator()
        self.decision = DecisionEngine()
        self.remote_model = RemoteModel()

    def route(self, state):

        # Step 1: Analyze the task
        state = self.analyzer.analyze(state)

        # Step 2: Generate local response
        state = self.local_model.generate(state)

        # Step 3: Compute confidence
        state = self.confidence.evaluate(state)

        # Step 4: Decide local vs remote
        state = self.decision.decide(state)

        # Step 5: If needed, call Fireworks
        if state.use_remote:
            state = self.remote_model.generate(state)
            state.final_answer = state.remote_answer

        return state