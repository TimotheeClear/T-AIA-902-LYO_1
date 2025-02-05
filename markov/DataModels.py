class Experience:
    def __init__(
        self,
        state       : int,
        action      : int,
        new_state   : int,
        reward      : float,
        terminated  : bool  = False,
        truncated   : bool  = False,
    ) -> None:
        self.State      = state
        self.Action     = action
        self.New_state  = new_state
        self.Reward     = reward
        self.Terminated = terminated
        self.Truncated  = truncated
        pass
    pass

    def pretty(self):
        return f"state {self.State} + action {self.Action} => state {self.New_state} and reward {self.Reward} ({'Done' if self.Terminated or self.Truncated else 'On-Going'})"

class Episode:
    def __init__(self) -> None:
        self.Experiences    = []
        self.Terminated     = False
        self.Truncated      = False
