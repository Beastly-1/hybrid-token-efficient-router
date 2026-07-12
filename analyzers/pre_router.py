from analyzers.difficulty_estimator import DifficultyEstimator
from analyzers.local_task_classifier import LocalTaskClassifier
from analyzers.prompt_library import PromptLibrary
from analyzers.sanity_checks import SanityChecker
from analyzers.task_analyzer import TaskAnalyzer
from analyzers.runtime_budget import RuntimeBudgetManager
from tools.datetime_utils import DateTimeUtils


class PreRouterPipeline:
    """Runs all deterministic work required before routing decisions."""

    def __init__(self):
        self.task_detector = TaskAnalyzer()
        self.local_task_classifier = LocalTaskClassifier()
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
        state = self.local_task_classifier.classify(state)
        state.difficulty, signals = self.difficulty_estimator.estimate(
            state.query, state.task_type
        )
        state.task_signals.update(signals)

        dt = DateTimeUtils()
        state.current_date = dt.current_date()
        state.current_time = dt.current_time()
        state.current_datetime = dt.current_datetime()

        state.system_prompt = (
            self.prompt_library.build(state.task_type, state.difficulty)
            + "\n\n"
            + f"Current date is {state.current_date}. "
            + f"Current time is {state.current_time}. "
            + f"Current datetime is {state.current_datetime}."
        )
        state = self.budget_manager.assign(state)
        return state
