from markov.DataModels  import Experience
from markov.Policies    import Policy 
import gym
import numpy as np

class QTablePolicy(Policy):
    def __init__(
        self,
        alpha   : float,
        gamma   : float,
        logging : bool = False,
    ):
        super().__init__(logging)
        self.alpha = alpha
        self.gamma = gamma

    def InitEnv(self, env: gym.Env):
        super().InitEnv(env)
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])

    def NextAction(self, state: int):
        return np.argmax(self.Q[state])  
    
    def learnFromExperiences(self, experiences: list[Experience]):
        for exp in experiences:
            self.learnFromExperience(exp)
    
    def learnFromExperience(self, exp : Experience):
        next_max = np.max(self.Q[exp.New_state]) 
        self.Q[exp.State, exp.Action] = self.Q[exp.State, exp.Action] + self.alpha * (exp.Reward + self.gamma * next_max - self.Q[exp.State, exp.Action])
        return
