from markov.Policies import Policy

class RandomPolicy(Policy):
    def NextAction(self, state: int):
        return self.env.action_space.sample()  