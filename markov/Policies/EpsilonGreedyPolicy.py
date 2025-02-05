from markov.DataModels  import Experience, Episode
from markov.Policies    import Policy
import gym
import numpy as np

from markov.Training import Agent

class EpsilonGreedyPolicy(Policy):
    def __init__(
        self,
        epsilon             : float,
        exploration_policy  : Policy,
        exploitation_policy : Policy,
        learn_exploration   : bool      = True,
        logging             : bool      = False,
    ) -> None:
        super().__init__(logging)
        self.epsilon                = epsilon
        self.exploration_policy     = exploration_policy
        self.exploitation_policy    = exploitation_policy

    def InitEnv(self, env: gym.Env):
        super().InitEnv(env)
        self.exploration_policy.InitEnv(env)
        self.exploitation_policy.InitEnv(env)

    def NextAction(self, state: int):
        should_explore = not self.inference and np.random.uniform(0, 1) < self.epsilon
        if should_explore:
            # print("exploring")
            return self.exploration_policy.NextAction(state)  
        # print("exploiting")
        return self.exploitation_policy.NextAction(state)  
    
    def getNextActionFromModel(self, state: int):
        return self.exploitation_policy.NextAction(state)
    
    def learnFromExperiences(self, experiences: list[Experience]):
        self.exploration_policy.Learn(experiences)
        self.exploitation_policy.Learn(experiences)


class DecayingEpsilonGreedyPolicy(EpsilonGreedyPolicy):
    def __init__(
        self,
        epsilon             : float,
        exploration_policy  : Policy,
        exploitation_policy : Policy,
        decay_factor        : float     = 1,
        min_epsilon         : float     = 0,
        learn_exploration   : bool      = True,
        logging             : bool      = False,
    ) -> None:
        super().__init__(epsilon, exploration_policy, exploitation_policy, learn_exploration, logging)
        self.decay_factor   = decay_factor
        self.min_epsilon    = min_epsilon

    def NextAction(self, state: int):
        action = super().NextAction(state)
        self.decay()
        return action 
    
    def getNextActionFromModel(self, state: int):
        return self.exploitation_policy.NextAction(state)
    

    
    def decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_factor) 