class PromptLibrary:
    """Small, task-aware system prompts for the local inference pass."""

    _TASK_INSTRUCTIONS = {
        "summarization": "Preserve the important facts and omit commentary.",
        "ner": "Return entities with their types in a compact, consistent form.",
        "sentiment": "Return the sentiment and a short justification only if requested.",
        "code_generation": "Return correct, runnable code and only necessary explanation.",
        "code_debug": "Identify the defect and provide a concrete corrected version.",
        "math": "Calculate carefully and give the final result clearly.",
        "logic": "Follow the constraints carefully and state the conclusion clearly.",
    }

    def build(self, task_type: str, difficulty: str) -> str:
        instruction = self._TASK_INSTRUCTIONS.get(
            task_type, "Answer accurately and directly."
        )
        budget_note = "Keep the answer concise." if difficulty == "easy" else "Use enough detail to be reliable."
        return (
            "You are a precise local assistant in a routing system. "
            f"{instruction} {budget_note} "
            "Do not invent facts. If the request is underspecified, say what is missing."
        )
