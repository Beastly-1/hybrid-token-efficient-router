from state import RoutingState


class TaskAnalyzer:
    """
    Classifies the incoming task before any model is called.
    This stage costs 0 tokens and helps the router make better decisions.
    """

    def analyze(self, state: RoutingState):

        query = state.query.lower()

        # -------------------------
        # Default values
        # -------------------------

        state.task_type = "general"
        state.difficulty = "medium"
        state.tool_candidate = None

        # -------------------------
        # Task Classification
        # -------------------------

        if any(word in query for word in [
            "summarize", "summary", "summarise"
        ]):
            state.task_type = "summarization"

        elif any(word in query for word in [
            "sentiment", "positive", "negative", "neutral"
        ]):
            state.task_type = "sentiment"

        elif any(word in query for word in [
            "entity", "entities", "person", "organization",
            "organisation", "location"
        ]):
            state.task_type = "ner"

        elif any(word in query for word in [
            "debug", "fix", "bug", "error", "traceback"
        ]):
            state.task_type = "code_debug"

        elif any(word in query for word in [
            "write a python function",
            "write a function",
            "implement",
            "generate code"
        ]):
            state.task_type = "code_generation"

        elif any(word in query for word in [
            "solve", "calculate", "percentage", "probability",
            "equation"
        ]):
            state.task_type = "math"

        elif any(word in query for word in [
            "logic", "puzzle", "deduce", "deduction"
        ]):
            state.task_type = "logic"

        else:
            state.task_type = "factual"

        # -------------------------
        # Difficulty Estimation
        # -------------------------

        words = len(query.split())

        if words < 10:
            state.difficulty = "easy"

        elif words < 40:
            state.difficulty = "medium"

        else:
            state.difficulty = "hard"

        # -------------------------
        # Tool Detection
        # -------------------------

        if any(op in query for op in [
            "+", "-", "*", "/", "%"
        ]):
            state.tool_candidate = "calculator"

        return state