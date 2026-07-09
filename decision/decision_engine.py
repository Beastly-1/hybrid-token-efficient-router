from config import ROUTE_THRESHOLDS


class DecisionEngine:
    """
    Decides whether to trust the local model
    or escalate to Fireworks.
    """

    def decide(self, state):

        threshold = ROUTE_THRESHOLDS.get(
            state.task_type,
            ROUTE_THRESHOLDS["default"]
        )

        if state.route_score >= threshold:
            state.use_remote = False
            state.final_answer = state.local_answer
            state.routing_reason = (
                f"Local accepted "
                f"({state.route_score:.3f} >= {threshold:.3f})"
            )

        else:
            state.use_remote = True
            state.routing_reason = (
                f"Escalated "
                f"({state.route_score:.3f} < {threshold:.3f})"
            )

        return state