from analyzers.difficulty_estimator import DifficultyEstimator
from analyzers.prompt_library import PromptLibrary
from analyzers.sanity_checks import SanityChecker
from analyzers.task_analyzer import TaskAnalyzer
from analyzers.runtime_budget import RuntimeBudgetManager


class PreRouterPipeline:
    """Runs all deterministic work required before routing decisions."""

    def __init__(self):
        self.task_detector = TaskAnalyzer()
        self.difficulty_estimator = DifficultyEstimator()
        self.prompt_library = PromptLibrary()
        self.budget_manager = RuntimeBudgetManager()
        self.sanity_checker = SanityChecker()

    def prepare(self, state):
        state = self.sanity_checker.validate_input(state)
        if not state.input_valid:
            state.final_answer = " ".join(state.input_issues)
            state.routing_reason = "Input sanity check failed."
            return state

        state = self.task_detector.analyze(state)
        state.difficulty, signals = self.difficulty_estimator.estimate(
            state.query, state.task_type
        )
        state.task_signals.update(signals)
        state.system_prompt = self.prompt_library.build(state.task_type, state.difficulty)
        state = self.budget_manager.assign(state)
        return state
