from config import LOCAL_MAX_NEW_TOKENS, LOCAL_TIMEOUT_SECONDS


class RuntimeBudgetManager:
    """Sets bounded generation budgets before local inference."""

    _TOKEN_BUDGETS = {"easy": 96, "medium": 192, "hard": LOCAL_MAX_NEW_TOKENS}

    def assign(self, state):
        state.max_new_tokens = self._TOKEN_BUDGETS[state.difficulty]
        state.timeout_seconds = LOCAL_TIMEOUT_SECONDS
        return state
