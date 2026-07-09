import math

from config import LOGPROB_WEIGHT, ANSWER_WEIGHT


class ConfidenceEvaluator:
    """
    Computes a routing score indicating whether the
    local model's answer is trustworthy enough.

    Signals:
    1. Local model log probabilities
    2. Answer quality heuristics
    """

    def evaluate(self, state):

        logprob_score = self._logprob_score(state.logprobs)

        answer_score = self._answer_quality_score(
            state.local_answer,
            state.task_type
        )

        # -----------------------------
        # Final Route Score
        # -----------------------------

        route_score = (
            LOGPROB_WEIGHT * logprob_score +
            ANSWER_WEIGHT * answer_score
        )

        state.route_score = round(route_score, 3)

        state.confidence_details = {
            "logprob_score": round(logprob_score, 3),
            "answer_score": round(answer_score, 3),
        }

        return state

    # =========================================================

    def _logprob_score(self, logprobs):
        """
        Converts average log probability into a
        confidence value between 0 and 1.
        """

        if not logprobs:
            return 0.0

        avg = sum(logprobs) / len(logprobs)

        score = math.exp(avg)

        return max(0.0, min(score, 1.0))

    # =========================================================

    def _answer_quality_score(self, answer, task_type):
        """
        Evaluates whether the answer appears reliable.

        Uses zero-token heuristic checks.
        """

        if not answer:
            return 0.0

        answer = answer.strip()

        if answer == "":
            return 0.0

        lower = answer.lower()

        # -----------------------------------------------------
        # Hard Fail Phrases
        # -----------------------------------------------------

        bad_phrases = [
            "i don't know",
            "i dont know",
            "not sure",
            "cannot answer",
            "can't answer",
            "unknown",
            "no idea",
            "maybe",
        ]

        if any(p in lower for p in bad_phrases):
            return 0.0

        score = 1.0

        # -----------------------------------------------------
        # Repeated output penalty
        # -----------------------------------------------------

        words = lower.split()

        if len(words) > 6:
            unique_ratio = len(set(words)) / len(words)

            if unique_ratio < 0.40:
                score -= 0.25

        # -----------------------------------------------------
        # Task-specific bonuses
        # -----------------------------------------------------

        if task_type == "code_generation":
            if "def " in answer:
                score += 0.05

        elif task_type == "code_debug":
            if "return" in answer:
                score += 0.05

        elif task_type == "ner":
            if ":" in answer:
                score += 0.05

        elif task_type == "summarization":
            if len(words) > 8:
                score += 0.05

        elif task_type == "math":
            # Short numeric answers are valid.
            pass

        return max(0.0, min(score, 1.0))