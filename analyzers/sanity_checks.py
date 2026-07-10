import re

from config import MAX_INPUT_CHARACTERS, MAX_PROMPT_CHARACTERS


class SanityChecker:
    """Validates bounded, usable inputs and rejects unusable local outputs."""

    def validate_input(self, state):
        query = state.query.strip()
        issues = []
        if not query:
            issues.append("Query is empty.")
        if len(query) > MAX_INPUT_CHARACTERS:
            issues.append(f"Query exceeds the {MAX_INPUT_CHARACTERS}-character limit.")
        if "\x00" in query:
            issues.append("Query contains a null character.")
        state.query = query
        state.input_issues = issues
        state.input_valid = not issues
        return state

    def validate_prompt(self, prompt: str) -> str:
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            return prompt[:MAX_PROMPT_CHARACTERS]
        return prompt

    def validate_output(self, answer: str) -> tuple[bool, str]:
        normalized = answer.strip()
        if not normalized:
            return False, "Local model returned an empty answer."
        if "\x00" in normalized:
            return False, "Local model returned an invalid character."
        if len(normalized) > 100 and len(set(re.findall(r"\w+", normalized.lower()))) < 3:
            return False, "Local model output is degenerate."
        return True, ""
