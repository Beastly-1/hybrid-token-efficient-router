from config import LOCAL_ACCEPT_DIFFICULTIES, ROUTE_THRESHOLDS


class DecisionEngine:
    """
    Decides whether the locally evaluated answer is safe to keep.
    """

    def decide(self, state):

        threshold = ROUTE_THRESHOLDS.get(state.task_type, ROUTE_THRESHOLDS["default"])
        local_allowed = state.difficulty in LOCAL_ACCEPT_DIFFICULTIES
        if state.local_answer and local_allowed and state.route_score >= threshold:
            state.use_remote = False
            state.final_answer = state.local_answer
            state.routing_reason = f"Local accepted ({state.route_score:.3f})."
        else:
            state.use_remote = True
            state.routing_reason = "Local answer unavailable or did not meet the acceptance policy."

        return state
