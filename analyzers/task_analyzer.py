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

        query = state.query.strip()
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

<<<<<<< HEAD
=======
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

>>>>>>> origin/with-step-1-again
        # ---------------- DateTime ----------------

        if any(
            phrase in q
            for phrase in [
                "current date",
                "today",
                "current time",
                "current datetime",
                "timestamp",
                "day of week",
                "days between",
                "add days",
                "subtract days",
                "leap year",
            ]
        ):
            state.tool_candidate = "datetime"

        # ---------------- Calculator ----------------

        else:

            math_keywords = [
                "calculate",
                "compute",
                "evaluate",
                "simplify",
                "solve",
                "sqrt",
                "sin",
                "cos",
                "tan",
                "factorial",
                "log",
                "log10",
                "exp",
                "equation",
                "integral",
                "derivative",
            ]

            has_math_expression = (
                re.search(r"\d+\s*[\+\*/%]\s*\d+", q) is not None
                or re.search(r"\d+\s*-\s*\d+", q) is not None
                or "**" in q
                or re.search(r"\d+\s*//\s*\d+", q) is not None
            )

            if has_math_expression or any(word in q for word in math_keywords):
                state.tool_candidate = "calculator"

            # ---------------- JSON ----------------

            elif (
                "json" in q
                and any(
                    word in q
                    for word in [
                        "validate",
                        "schema",
                        "parse",
                        "pretty",
                    ]
                )
            ):
                state.tool_candidate = "json_validator"

            # ---------------- Regex ----------------

            elif any(
                phrase in q
                for phrase in [
                    "verify email",
                    "validate email",
                    "verify phone",
                    "validate phone",
                    "verify url",
                    "validate url",
                    "verify ipv4",
                    "validate ipv4",
                    "verify uuid",
                    "validate uuid",
                    "regex pattern",
                    "regex match",
                ]
            ):
                state.tool_candidate = "regex_verifier"

            # ---------------- Python ----------------

            elif any(
                phrase in q
                for phrase in [
                    "run python",
                    "execute python",
                    "execute code",
                    "run this code",
                ]
            ):
                state.tool_candidate = "python_executor"

            # ---------------- Verification ----------------

            elif (
                "verify json" in q
                or "verification" in q
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
                "classify sentiment",
                "positive",
                "negative",
                "neutral",
            ]
        ):
            state.task_type = "sentiment"

        elif any(
            word in q
            for word in [
                "named entity",
                "extract entities",
                "extract named entities",
                "entities",
            ]
        ):
            state.task_type = "ner"

        elif any(
            word in q
            for word in [
                "debug",
                "bug",
                "traceback",
                "exception",
                "fix",
                "error",
            ]
        ):
            state.task_type = "code_debug"

        elif any(
            phrase in q
            for phrase in [
                "write a python function",
                "python function",
                "write a function",
                "write a python function",
                "generate code",
                "write code",
                "implement",
            ]
        ):
            state.task_type = "code_generation"

        elif any(
            phrase in q
            for phrase in [
                "logic",
                "puzzle",
                "deduce",
                "deduction",
                "who owns",
                "arrangement",
                "arrangements",
                "circular table",
                "opposite",
                "left of",
                "right of",
            ]
        ):
            state.task_type = "logic"

        elif (
            state.tool_candidate == "calculator"
            or (
                re.search(r"\d", q)
                and any(
                    word in q
                    for word in [
                        "equation",
                        "percentage",
                        "probability",
                        "integral",
                        "derivative",
                    ]
                )
            )
        ):
            state.task_type = "math"

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

        if state.task_type in [
            "logic",
            "code_generation",
            "code_debug",
        ]:
            state.difficulty = "hard"

        elif (
            state.task_type in [
                "summarization",
                "ner",
            ]
            and words > 25
        ):
            state.difficulty = "hard"

        return state
