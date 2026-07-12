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
        state.force_remote = False

        # =================================================
        # LEGAL QUERIES - ALWAYS REMOTE
        # =================================================

        legal_keywords = [
            "legal",
            "law",
            "lawsuit",
            "contract",
            "attorney",
            "lawyer",
            "court",
            "jurisdiction",
            "liability",
            "copyright",
            "patent",
            "trademark",
        ]
        
        if any(keyword in q for keyword in legal_keywords):
            state.force_remote = True
            state.task_type = "legal"
            return state

        # =================================================
        # TOOL DETECTION
        # =================================================

        # ---------------- DateTime ----------------

        if any(
            phrase in q
            for phrase in [
                "current date",
                "what's the date",
                "whats the date",
                "what is the date",
                "what's the date",
                "whats the date",
                "what date is it",
                "what is today's date",
                "today's date",
                "is this date valid",
                "date today",
                "show me the date",
                "tell me the date",
                "today",
                "current time",
                "is this time valid",
                "what time is it",
                "what's the time",
                "whats the time",
                "what is the time",
                "time now",
                "time right now",
                "show me the time",
                "tell me the time",
                "right now",
                "current datetime",
                "timestamp",
                "day of week",
                "what day is it",
                "what day of the week is it",
                "days between",
                "days until",
                "days since",
                "add days",
                "subtract days",
                "date format",
                "format this date",
                "convert this date",
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
                "sum",
                "add",
                "subtract",
                "multiply",
                "divide",
                "percent",
                "percentage",
                "round",
                "rounding",
                "convert",
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
                or re.search(r"\d+\s*\+\s*\d+", q) is not None
                or re.search(r"\d+\s+(plus|minus|times|multiplied by|divided by)\s+\d+", q)
                is not None
                or re.search(r"\b(sum|add|subtract|multiply|divide)\b", q) is not None
                or re.search(r"\d+\s*%", q) is not None
            )

            if has_math_expression or any(
                re.search(rf"\b{re.escape(word)}\b", q)
                for word in math_keywords
            ):
                state.tool_candidate = "calculator"

            # ---------------- JSON ----------------

            elif (
                "json" in q
                and any(
                    word in q
                    for word in [
                "validate",
                "check",
                "is this json valid",
                "schema",
                "parse",
                "pretty",
                "format",
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
                "check email",
                "is this email valid",
                "verify phone",
                "validate phone",
                "check phone",
                "is this phone valid",
                "verify url",
                "validate url",
                "check url",
                "is this url valid",
                "verify ipv4",
                "validate ipv4",
                "check ipv4",
                "verify uuid",
                "validate uuid",
                "check uuid",
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
                "run this python",
                "execute code",
                "run this code",
                "python script",
                "python code",
            ]
            ):
                state.tool_candidate = "python_executor"

            # ---------------- Verification ----------------

            elif (
                "verify json" in q
                or "verification" in q
                or "check json" in q
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
                    re.search(rf"\b{re.escape(word)}\b", q)
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
