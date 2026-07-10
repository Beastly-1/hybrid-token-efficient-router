import re

from state import RoutingState


class TaskAnalyzer:
    """
    Lightweight zero-token analyzer.

    Determines:
    - Task type
    - Difficulty
    - Candidate tool
    """

    def analyze(self, state: RoutingState):

        query = state.query
        q = query.lower()

        # -------------------------------------------------
        # Defaults
        # -------------------------------------------------

        state.task_type = "factual"
        state.difficulty = "medium"
        state.tool_candidate = None

        # =================================================
        # TOOL DETECTION
        # =================================================

        # ---------------- Calculator ----------------

        has_math_keyword = bool(
            re.search(
                r"\b(?:calculate|compute|evaluate|simplify|sqrt|sin|cos|tan|"
                r"factorial|log|exp)\b",
                q,
            )
        )
        has_numeric_expression = bool(
            re.search(r"\d\s*(?:\+|-|\*{1,2}|/|%|\^|//)\s*\d", q)
        )

        if has_math_keyword or has_numeric_expression:
            state.tool_candidate = "calculator"

        # ---------------- JSON ----------------

        elif (
            "json" in q
            and (
                "validate" in q
                or "schema" in q
                or "parse" in q
                or "pretty" in q
            )
        ):
            state.tool_candidate = "json_validator"

        # ---------------- Regex ----------------

        elif any(
            word in q
            for word in [
                "email",
                "phone",
                "url",
                "uuid",
                "ipv4",
                "regex",
                "date format",
                "time format",
            ]
        ):
            state.tool_candidate = "regex_verifier"

        # ---------------- DateTime ----------------

        elif any(
            word in q
            for word in [
                "current date",
                "today",
                "current time",
                "timestamp",
                "day of week",
                "days between",
                "add days",
                "subtract days",
                "leap year",
            ]
        ):
            state.tool_candidate = "datetime"

        # ---------------- Python ----------------

        elif any(
            word in q
            for word in [
                "run python",
                "execute python",
                "execute code",
                "run this code",
            ]
        ):
            state.tool_candidate = "python_executor"

        # ---------------- Verification ----------------

        elif (
            "verify" in q
            or "validation" in q
        ):
            state.tool_candidate = "verification"

        # =================================================
        # TASK TYPE
        # =================================================

        if any(
            word in q
            for word in [
                "summarize",
                "summarise",
                "summary",
            ]
        ):
            state.task_type = "summarization"

        elif any(
            word in q
            for word in [
                "sentiment",
                "positive",
                "negative",
                "neutral",
            ]
        ):
            state.task_type = "sentiment"

        elif any(
            word in q
            for word in [
                "entity",
                "entities",
                "person",
                "organization",
                "organisation",
                "location",
                "named entity",
            ]
        ):
            state.task_type = "ner"

        elif any(
            word in q
            for word in [
                "debug",
                "bug",
                "traceback",
                "error",
                "fix",
                "exception",
            ]
        ):
            state.task_type = "code_debug"

        elif any(
            word in q
            for word in [
                "write a function",
                "write a python function",
                "generate code",
                "implement",
                "write python",
                "write code",
            ]
        ):
            state.task_type = "code_generation"

        elif (
            state.tool_candidate == "calculator"
            or any(
                word in q
                for word in [
                    "equation",
                    "percentage",
                    "probability",
                    "integral",
                    "derivative",
                ]
            )
        ):
            state.task_type = "math"

        elif any(
            word in q
            for word in [
                "logic",
                "puzzle",
                "deduce",
                "deduction",
                "constraint",
            ]
        ):
            state.task_type = "logic"

        else:
            state.task_type = "factual"

        # =================================================
        # DIFFICULTY
        # =================================================

        words = len(query.split())

        if words <= 8:
            state.difficulty = "easy"

        elif words <= 35:
            state.difficulty = "medium"

        else:
            state.difficulty = "hard"

        # Long reasoning tasks are harder

        if state.task_type in [
            "logic",
            "code_generation",
            "code_debug",
        ]:
            state.difficulty = "hard"

        elif state.task_type in [
            "summarization",
            "ner",
        ]:
            if words > 25:
                state.difficulty = "hard"

        return state
