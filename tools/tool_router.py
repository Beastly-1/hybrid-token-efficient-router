from datetime import datetime
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

            candidate = self._extract_regex_candidate(query, lower, regex)

            if (
            "verify email" in lower
            or "validate email" in lower
            or "check email" in lower
            or "is this email valid" in lower
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
                or "check phone" in lower
                or "is this phone valid" in lower
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
                or "check url" in lower
                or "is this url valid" in lower
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

            if "current date" in lower or "today" in lower:

                state.final_answer = dt.current_date()

            elif "is this date valid" in lower:

                candidate = self._extract_date_candidate(query)
                state.final_answer = (
                    "Valid Date"
                    if candidate and dt.day_of_week(candidate) is not None
                    else "Invalid Date"
                )

            elif "is this time valid" in lower:

                candidate = self._extract_time_candidate(query)
                state.final_answer = (
                    "Valid Time"
                    if candidate and self._is_time_candidate(candidate)
                    else "Invalid Time"
                )

            elif (
                "current time" in lower
                or "what time is it" in lower
                or "what's the time" in lower
                or "whats the time" in lower
                or "what is the time" in lower
                or "time now" in lower
                or "time right now" in lower
                or "right now" in lower
                or "show me the time" in lower
                or "tell me the time" in lower
            ):

                state.final_answer = dt.current_time()

            elif "current datetime" in lower:

                state.final_answer = dt.current_datetime()

            elif "timestamp" in lower:

                state.final_answer = str(
                    dt.current_timestamp()
                )

            elif "leap year" in lower:

                match = re.search(r"\b(\d{4})\b", query)

                if match:

                    year = int(match.group())

                    state.final_answer = str(
                        dt.is_leap_year(year)
                    )

                else:

                    return state

            elif "christmas" in lower:

                year_match = re.search(r"\b(\d{4})\b", query)
                year = int(year_match.group()) if year_match else datetime.now().year
                state.final_answer = dt.day_of_week(f"{year}-12-25")

            elif "day of week" in lower or "day of the week" in lower:

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

            elif "add days" in lower:

                date_match = re.search(r"\d{4}-\d{2}-\d{2}", query)
                days_match = re.search(r"\b(-?\d+)\b", query)

                if date_match and days_match:
                    state.final_answer = dt.add_days(
                        date_match.group(),
                        int(days_match.group())
                    )
                else:
                    return state

            elif "subtract days" in lower:

                date_match = re.search(r"\d{4}-\d{2}-\d{2}", query)
                days_match = re.search(r"\b(-?\d+)\b", query)

                if date_match and days_match:
                    state.final_answer = dt.subtract_days(
                        date_match.group(),
                        int(days_match.group())
                    )
                else:
                    return state

            elif "days between" in lower:

                dates = re.findall(r"\d{4}-\d{2}-\d{2}", query)

                if len(dates) >= 2:
                    state.final_answer = str(
                        dt.days_between(dates[0], dates[1])
                    )
                else:
                    return state

            elif (
                "format date" in lower
                or "format this date" in lower
                or "date format" in lower
                or "convert this date" in lower
            ):

                date_match = re.search(r"\d{4}-\d{2}-\d{2}", query)
                format_match = re.search(r"(%[^ ]+)", query)

                if date_match and format_match:
                    state.final_answer = dt.format_date(
                        date_match.group(),
                        format_match.group(1)
                    )
                elif date_match and "convert this date" in lower:
                    state.final_answer = date_match.group()
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
                or "check" in query
                or "is this json valid" in query
                or "schema" in query
                or "parse" in query
                or "pretty" in query
                or "format" in query
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
                "is this date valid",
                "is this time valid",
                "check email",
                "check phone",
                "check url",
                "is this email valid",
                "is this phone valid",
                "is this url valid",
                "verify date",
                "validate date",
                "check date",
                "verify time",
                "validate time",
                "check time",
            ]
        )

    def _should_use_datetime(self, query):

        return any(
            word in query
            for word in [
                "current date",
                "what's the date",
                "whats the date",
                "what is the date",
                "date today",
                "show me the date",
                "tell me the date",
                "today",
                "current time",
                "what time is it",
                "what's the time",
                "whats the time",
                "what is the time",
                "time now",
                "time right now",
                "right now",
                "show me the time",
                "tell me the time",
                "current datetime",
                "timestamp",
                "day of week",
                "what day is it",
                "what day of the week is it",
                "day of the week",
                "days between",
                "days until",
                "days since",
                "add days",
                "subtract days",
                "leap year",
                "format date",
                "format this date",
                "convert this date",
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
            or "check" in query
        )

    # ======================================================
    # Helpers
    # ======================================================

    def _extract_expression(self, query):

        query = re.sub(
            r"(?i)^\s*(what is|what's|whats|calculate|compute|evaluate|solve|how much is|how much are|can you calculate)\b[:\s]*",
            "",
            query,
        )

        query = re.sub(
            r"(?i)\b(plus)\b",
            "+",
            query,
        )
        query = re.sub(
            r"(?i)\b(minus)\b",
            "-",
            query,
        )
        query = re.sub(
            r"(?i)\b(times|multiplied by)\b",
            "*",
            query,
        )
        query = re.sub(
            r"(?i)\b(divided by)\b",
            "/",
            query,
        )

        percent_of_match = re.search(
            r"(?i)\b(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)\b",
            query,
        )
        if percent_of_match:
            left = percent_of_match.group(1)
            right = percent_of_match.group(2)
            return f"({left} / 100) * {right}"

        match = re.search(
            r"[\d\w\.\(\)\s\+\-\*/%//\^,]+",
            query,
        )

        if match:
            candidate = match.group().strip()
            if candidate:
                return candidate

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

    def _extract_regex_candidate(self, query, lower, regex):

        if "email" in lower:
            candidate = regex.extract(regex.EMAIL, query)
            return self._normalize_candidate(
                candidate or self._extract_last_token(query)
            )

        if "phone" in lower:
            candidate = regex.extract(regex.PHONE, query)
            return self._normalize_candidate(
                candidate or self._extract_last_token(query)
            )

        if "url" in lower:
            candidate = regex.extract(regex.URL, query)
            return self._normalize_candidate(
                candidate or self._extract_last_token(query)
            )

        if "ipv4" in lower:
            candidate = regex.extract(regex.IPV4, query)
            return self._normalize_candidate(
                candidate or self._extract_last_token(query)
            )

        if "uuid" in lower:
            candidate = regex.extract(regex.UUID, query)
            return self._normalize_candidate(
                candidate or self._extract_last_token(query)
            )

        if "date" in lower:
            candidate = regex.extract(regex.DATE, query)
            return self._normalize_candidate(
                candidate or self._extract_last_token(query)
            )

        if "time" in lower:
            candidate = regex.extract(regex.TIME, query)
            return self._normalize_candidate(
                candidate or self._extract_last_token(query)
            )

        return self._normalize_candidate(self._extract_last_token(query))

    def _normalize_candidate(self, candidate):

        return candidate.strip(" \t\n\r.,!?;:'\"`()[]{}<>")

    def _extract_code(self, query):

        if "```" in query:

            parts = query.split("```")

            if len(parts) >= 3:

                code = parts[1]

                if code.startswith("python"):
                    code = code[6:].strip()

                return code.strip()

        return None

    def _extract_date_candidate(self, query):

        match = re.search(r"\d{4}-\d{2}-\d{2}", query)
        return match.group() if match else None

    def _extract_time_candidate(self, query):

        match = re.search(r"\b\d{2}:\d{2}(:\d{2})?\b", query)
        return match.group() if match else None

    def _is_time_candidate(self, candidate):

        return candidate.count(":") in (1, 2)
