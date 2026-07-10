import json
import re

from tools.tool_registry import ToolRegistry


class ToolRouter:
    """
    Routes deterministic queries to tools before any LLM is used.
    """

    def __init__(self):

        self.registry = ToolRegistry()

    # =======================================================
    # Main Entry
    # =======================================================

    def execute(self, state):

        query = state.query.strip()
        lower = query.lower()

        state.tool_used = None

        # ---------------------------------------------------
        # Calculator
        # ---------------------------------------------------

        if self._should_use_calculator(state):

            calculator = self.registry.get_tool("calculator")

            expression = self._extract_expression(query)

            result = calculator.calculate(expression)

            if result["success"]:

                state.final_answer = str(result["result"])
                state.tool_used = "calculator"

                return state

        # ---------------------------------------------------
        # JSON
        # ---------------------------------------------------

        if self._should_use_json(lower):

            validator = self.registry.get_tool("json_validator")

            json_text = self._extract_json(query)

            if json_text:

                valid = validator.validate(json_text)

                if valid:

                    parsed = validator.parse(json_text)

                    pretty = validator.pretty(parsed)

                    state.final_answer = (
                        "Valid JSON\n\n"
                        + pretty
                    )

                else:

                    state.final_answer = "Invalid JSON"

                state.tool_used = "json_validator"

                return state

        # ---------------------------------------------------
        # Regex
        # ---------------------------------------------------

        if self._should_use_regex(lower):

            regex = self.registry.get_tool(
                "regex_verifier"
            )

            candidate = self._extract_last_token(query)

            if (
            "verify email" in lower
            or "validate email" in lower
            or "email address" in lower
                ):

                state.final_answer = (
                    "Valid Email"
                    if regex.is_email(candidate)
                    else "Invalid Email"
                )

            elif (
                "verify phone" in lower
                or "validate phone" in lower
                or "phone number" in lower
                ):

                state.final_answer = (
                    "Valid Phone Number"
                    if regex.is_phone(candidate)
                    else "Invalid Phone Number"
                )

            elif (
                "verify url" in lower
                or "validate url" in lower
                ):  

                state.final_answer = (
                    "Valid URL"
                    if regex.is_url(candidate)
                    else "Invalid URL"
                )

            elif "ipv4" in lower:

                state.final_answer = (
                    "Valid IPv4"
                    if regex.is_ipv4(candidate)
                    else "Invalid IPv4"
                )

            elif "uuid" in lower:

                state.final_answer = (
                    "Valid UUID"
                    if regex.is_uuid(candidate)
                    else "Invalid UUID"
                )

            elif "date" in lower:

                state.final_answer = (
                    "Valid Date"
                    if regex.is_date(candidate)
                    else "Invalid Date"
                )

            elif "time" in lower:

                state.final_answer = (
                    "Valid Time"
                    if regex.is_time(candidate)
                    else "Invalid Time"
                )

            else:

                return state

            state.tool_used = "regex_verifier"

            return state

        # ---------------------------------------------------
        # DateTime
        # ---------------------------------------------------

        if self._should_use_datetime(lower):

            dt = self.registry.get_tool("datetime")

            if "current date" in lower:

                state.final_answer = dt.current_date()

            elif "current time" in lower:

                state.final_answer = dt.current_time()

            elif "current datetime" in lower:

                state.final_answer = dt.current_datetime()

            elif "timestamp" in lower:

                state.final_answer = str(
                    dt.current_timestamp()
                )

            elif "leap year" in lower:

                match = re.search(r"\d{4}", query)

                if match:

                    year = int(match.group())

                    state.final_answer = str(
                        dt.is_leap_year(year)
                    )

                else:

                    return state

            elif "day of week" in lower:

                match = re.search(
                    r"\d{4}-\d{2}-\d{2}",
                    query
                )

                if match:

                    state.final_answer = (
                        dt.day_of_week(
                            match.group()
                        )
                    )

                else:

                    return state

            else:

                return state

            state.tool_used = "datetime"

            return state
                # ---------------------------------------------------
        # Python Executor
        # ---------------------------------------------------

        if self._should_use_python(state, lower):

            executor = self.registry.get_tool(
                "python_executor"
            )

            code = self._extract_code(query)

            if code:

                result = executor.execute(code)

                if result["success"]:

                    state.final_answer = (
                        result["stdout"].strip()
                    )

                else:

                    state.final_answer = (
                        result["stderr"].strip()
                    )

                state.tool_used = "python_executor"

                return state

        # ---------------------------------------------------
        # Verification
        # ---------------------------------------------------

        if self._should_use_verification(lower):

            verification = self.registry.get_tool(
                "verification"
            )

            if "json" in lower:

                json_text = self._extract_json(query)

                if json_text:

                    state.final_answer = str(
                        verification.verify_json(
                            json_text
                        )
                    )

                    state.tool_used = "verification"

                    return state

        return state

    # ======================================================
    # Routing Decisions
    # ======================================================

    def _should_use_calculator(self, state):

        return (
            state.tool_candidate == "calculator"
            or state.task_type == "math"
        )

    def _should_use_json(self, query):

        return (
            "json" in query
            and (
                "validate" in query
                or "schema" in query
                or "parse" in query
            )
        )

    def _should_use_regex(self, query):

        return any(
            word in query
            for word in [
                "email",
                "phone",
                "url",
                "ipv4",
                "uuid",
                "date format",
                "time format",
            ]
        )

    def _should_use_datetime(self, query):

        return any(
            word in query
            for word in [
                "current date",
                "current time",
                "current datetime",
                "timestamp",
                "day of week",
                "leap year",
            ]
        )

    def _should_use_python(self, state, query):

        return (
            state.tool_candidate == "python_executor"
            or state.task_type == "code_debug"
            or "execute python" in query
            or "run python" in query
            or "execute code" in query
        )

    def _should_use_verification(self, query):

        return (
            "verify" in query
            or "validation" in query
        )

    # ======================================================
    # Helpers
    # ======================================================

    def _extract_expression(self, query):

        query = re.sub(
            r"(?i)(calculate|compute|evaluate|solve)",
            "",
            query,
        )

        return query.strip()

    def _extract_json(self, query):

        start = query.find("{")
        end = query.rfind("}")

        if start == -1 or end == -1:
            return None

        return query[start:end + 1]

    def _extract_last_token(self, query):

        tokens = query.split()

        if not tokens:
            return ""

        return tokens[-1]

    def _extract_code(self, query):

        if "```" in query:

            parts = query.split("```")

            if len(parts) >= 3:

                code = parts[1]

                if code.startswith("python"):
                    code = code[6:].strip()

                return code.strip()

        return None