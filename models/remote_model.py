class RemoteModel:
    """
    Wrapper around Fireworks AI.
    """

    def generate(self, state):
        """
        Takes a RoutingState and updates:
        - state.remote_answer
        """
        raise NotImplementedError("Implement Fireworks API here.")