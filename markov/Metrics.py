from functools import reduce
from typing         import Any, Callable, List
from markov.DataModels import Experience, Episode

# === Step Metrics === #
StepMetricFunction      = Callable[[Experience], Any]

def CumulativeRewardFactory():
    cumulative_reward = 0
    def compute(e : Experience):
        nonlocal cumulative_reward 
        cumulative_reward += e.Reward
        return cumulative_reward
    return compute

# === Episode Metrics === #
EpisodeMetricFunction   = Callable[[Episode], Any]

def EpisodeReward(episode : Episode):
    return reduce(
        lambda acc, exp: acc + exp.Reward,
        episode.Experiences,
        0
    )
# === Model Readiness Metrics === #
ModelReadinessFunction   = Callable[[float], Any]

def ModelReadiness(reward: float):
    return reward
def EpisodeTerminated(episode : Episode):
    return 1 if episode.Terminated else 0
