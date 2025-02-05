from markov.DataModels  import Experience
from .._interfaces       import IActionTaker, ILearner
from contextlib         import contextmanager
from typing             import final
import gym
import numpy as np

class Policy(IActionTaker, ILearner):
    def __init__(
        self,
        logging     : bool      = False,
    ) -> None:
        self.logging    = logging
        self.inference  = False

    def toggleInference(self, activate_inference : bool):
        self.inference = activate_inference

    def InitEnv(self, env: gym.Env):
        self.env = env

    @contextmanager
    def Inference(self, toggle : bool):
        original_inference  = self.inference  
        self.inference      = toggle  
        try:
            yield  
        finally:
            self.inference  = original_inference 

    def debug(self, msg):
        if self.logging:
            print(msg)
    
    def NextAction(self, state: int):
        return self.env.action_space.sample()
    
    @final
    def Learn(self, experiences: list[Experience]):
        if self.inference:
            return
        self.learnFromExperiences(experiences)

    def learnFromExperiences(self, experiences: list[Experience]):
        pass

    # is used to get the action that the policy would take if it were in inference mode and the environment was in the given state
    # early stopping is a technique used to get a metric on the performance of the training process
    def getNextActionFromModel(self, state: int):
        return self.NextAction(state)