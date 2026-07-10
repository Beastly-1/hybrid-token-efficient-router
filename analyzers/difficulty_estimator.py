import re


class DifficultyEstimator:
    """Estimates task effort using deterministic prompt signals."""

    _HARD_TASKS = {"logic", "code_generation", "code_debug"}
    _REASONING_TERMS = {
        "analyze", "compare", "deduce", "derive", "design", "explain",
        "optimize", "prove", "reason", "step by step", "tradeoff",
    }

    def estimate(self, query: str, task_type: str) -> tuple[str, dict]:
        word_count = len(re.findall(r"\S+", query))
        line_count = query.count("\n") + 1
        has_code = "```" in query or bool(re.search(r"\b(def|class|function)\b", query))
        reasoning_hits = sum(term in query.lower() for term in self._REASONING_TERMS)

        score = 0
        score += 2 if word_count > 80 else 1 if word_count > 25 else 0
        score += 1 if line_count > 8 else 0
        score += 1 if has_code else 0
        score += min(reasoning_hits, 2)
        score += 2 if task_type in self._HARD_TASKS else 0

        difficulty = (
            "hard"
            if task_type in self._HARD_TASKS or score >= 3
            else "medium"
            if score >= 1
            else "easy"
        )
        return difficulty, {
            "word_count": word_count,
            "line_count": line_count,
            "has_code": has_code,
            "reasoning_hits": reasoning_hits,
            "difficulty_score": score,
        }
