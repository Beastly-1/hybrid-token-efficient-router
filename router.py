from state import RoutingState


class Router:
    """
    Main orchestration pipeline.
    """

    def route(self, query: str):

        # Create shared state
        state = RoutingState(query=query)

        # ----------------------------
        # Step 1: Analyze Query
        # ----------------------------
        self.analyze(state)

        # ----------------------------
        # Step 2: Try Tool
        # ----------------------------
        self.try_tool(state)

        # ----------------------------
        # Step 3: Local Model
        # ----------------------------
        if not state.tool_used:
            self.local_model(state)

        # ----------------------------
        # Step 4: Evaluate
        # ----------------------------
        self.evaluate(state)

        # ----------------------------
        # Step 5: Decide
        # ----------------------------
        self.decide(state)

        # ----------------------------
        # Step 6: Remote Model
        # ----------------------------
        if state.decision == "REMOTE":
            self.remote_model(state)

        # ----------------------------
        # Step 7: Finalize
        # ----------------------------
        self.finalize(state)

        return state.final_answer

    # ==================================
    # These are placeholders for now
    # ==================================

    def analyze(self, state):
        pass

    def try_tool(self, state):
        pass

    def local_model(self, state):
        pass

    def evaluate(self, state):
        pass

    def decide(self, state):
        pass

    def remote_model(self, state):
        pass

    def finalize(self, state):
        pass