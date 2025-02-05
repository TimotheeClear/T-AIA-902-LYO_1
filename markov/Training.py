from copy import deepcopy
from pathlib import Path

from time import sleep
from typing             import Union, Generator, Iterator

from requests import get
from markov.DataModels  import Episode, Experience
from markov.Policies.Policy    import Policy
from markov.Metrics     import EpisodeMetricFunction, StepMetricFunction, ModelReadinessFunction
from markov._interfaces import ISavableToDisk 

import gym, json
from typing import Callable, Dict, Iterator, Union, List, Optional

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        """Reset the environment to its initial state."""
        return super().reset()

    def resetOnState(self, state=None):
        """Reset the environment to a specified state."""
        # Standard reset to ensure the environment is properly initialized
        test, test2 = self.reset()
        if state is not None:
            self.state = state
            self.steps_beyond_done = None
            # Check if the environment allows direct manipulation of the state
            if hasattr(self.env, 's'):
                self.env.s = state # type: ignore
            if hasattr(self.env.unwrapped, 's'):
                self.env.unwrapped.s = state # type: ignore
            else:
                raise AttributeError("The environment does not expose 's' for direct manipulation.")
        # Return the state of the environment to confirm it's been set
        return getattr(self.env, 's', None)
    
class AgentMetrics:
    def __init__(
        self,
        step_metrics            : dict[str, StepMetricFunction]    = {},
        episode_metrics         : dict[str, EpisodeMetricFunction] = {},
        model_readiness_metrics         : dict[str, ModelReadinessFunction] = {},
    ):
        self.step_metrics       = step_metrics
        self.episode_metrics    = episode_metrics
        self.model_readiness_metrics = model_readiness_metrics

        self.results = {
            "step"      : { metric:[] for metric, _ in step_metrics.items() },
            "episode"   : { metric:[] for metric, _ in episode_metrics.items() },
            "model_readiness":{ metric:[] for metric, _ in model_readiness_metrics.items() },
        }
    
    def ComputeStepMetrics(self, e : Experience):
        for name, func  in self.step_metrics.items():
            self.results["step"][name].append(func(e))

    def ComputeEpisodeMetrics(self, e : Episode):
        for name, func in self.episode_metrics.items():
            self.results["episode"][name].append(func(e))

    def ComputeModelReadinessMettrics(self, reward : float):
        for name, func in self.model_readiness_metrics.items():
            self.results["model_readiness"][name].append(func(reward))
        
    
class AgentConfig:
    def __init__(
        self,
        policy                  : Policy,
        learning_policies       : dict[str, Policy] | None  = None,
        experience_buffer_size  : Union[int, Iterator[int], Generator[int, None, None]] = 1,
        model_readiness          : bool = False,
        model_readiness_modulo   : int = 10,  
        get_curriculum_training_callbacks: Optional[Callable[[], List[Callable[['Agent'], bool]]]] = None,
        early_stop              : bool = False
    ):
        self.model_readiness = model_readiness
        self.model_readiness_modulo = model_readiness_modulo
        self.experience_policy  = policy
        self.learning_policies  = {"experience_policy": policy} if learning_policies is None else learning_policies
        self.get_curriculum_training_callbacks = get_curriculum_training_callbacks

        if isinstance(experience_buffer_size, int):
            self.max_buffer_size_generator = self._generate_infinite(experience_buffer_size)
        else:
            self.max_buffer_size_generator = experience_buffer_size
        self.max_buffer_size = next(self.max_buffer_size_generator)
        self.early_stop = early_stop

    def _generate_infinite(self, value: int) -> Iterator[int]:
        while True:
            yield value


class Agent(
    # ISavableToDisk
):
    def __init__(
        self,
        training_env            : gym.Env,
        config                  : AgentConfig,
        metrics                 : AgentMetrics = AgentMetrics(),
        curriculum_learning     : bool = False

    ):
        config.experience_policy.InitEnv(training_env)
        for policy in config.learning_policies.values():
            policy.InitEnv(training_env)

        self.training_env = CustomEnvWrapper(training_env)


        self.config  = config
        self.curriculum_learning = curriculum_learning
        self.Metrics = metrics

    def Train(self, epochs : int) -> AgentMetrics :
        curriculum_episodes = 0
        if self.config.get_curriculum_training_callbacks:
            # get a list of experience to do
            # experience is a callback done in loop until condition is met
            curriculum_trainings = self.config.get_curriculum_training_callbacks()
            for curriculum_training in curriculum_trainings:

                while curriculum_training(self) is False:
                    # Here, each curriculum training continues until its condition is met
                    curriculum_episodes += 1
                
                    pass
                curriculum_episodes += 1
        for epoch in range(epochs - curriculum_episodes):
            epoch += curriculum_episodes

            self.experience_buffer  = []
            state, info             = self.training_env.reset()
            episode                 = Episode()

            # is training state piloted?
            # state = state or getPilotedState()
            

            while True:
                action = self.config.experience_policy.NextAction(state)
                # map action 0-> 4 ; 1->5;  2-> 0;  3-> 1; 4->2; 5->3; 
                maped_action = action + 4 if action < 2 else action - 2
                new_state, reward, terminated, truncated, info = self.training_env.step(maped_action)

                experience = Experience(
                    state       = state, 
                    action      = action, 
                    new_state   = new_state, 
                    reward      = reward, 
                    terminated  = terminated, 
                    truncated   = truncated
                )
                episode.Experiences.append(experience)

                self.experience_buffer.append(experience)
                
                self.Metrics.ComputeStepMetrics(experience)

                if len(self.experience_buffer) == self.config.max_buffer_size:
                    self.emptyExperienceBuffer()
                
                state = new_state
                
                if terminated or truncated:
                    episode.Terminated = terminated
                    episode.Truncated  = truncated
                    self.Metrics.ComputeEpisodeMetrics(episode)
                    break
            
            self.emptyExperienceBuffer()

            # get reward for all states to determine if model is done training
            if self.config.model_readiness and epoch != 0 and (epoch ) % self.config.model_readiness_modulo == 0:
                total_reward = 0
                # for initial_state in range(500):
                for initial_state in state_iterator():
                    self.training_env.resetOnState(initial_state)
                    state = initial_state

                    while True:
                        action = self.config.experience_policy.getNextActionFromModel(state)
                        # map action 0-> 4 ; 1->5;  2-> 0;  3-> 1; 4->2; 5->3; 
                        maped_action = action + 4 if action < 2 else action - 2

                        new_state, reward, terminated, truncated, info = self.training_env.step(maped_action)
                        total_reward += reward
                        state=new_state


                        if terminated or truncated:
                            episode.Terminated = terminated
                            episode.Truncated  = truncated
                            break

                self.Metrics.ComputeModelReadinessMettrics(total_reward)
                # print(f"\tEpisode n°{epoch} total reward: {total_reward}")
                if self.config.early_stop and total_reward == 2379:
                    # print(f"Model is ready after {epoch} epochs. total reward: {total_reward}")
                    return self.Metrics
                    
                
                # to ensure deterministic rewardnessmetric decomment
                # second_total_reward = 0
                # for initial_state in state_iterator():
                #     self.training_env.resetOnState(initial_state)
                #     state = initial_state
                #     while True:
                #         action = self.config.experience_policy.getNextActionFromModel(state)

                #         new_state, reward, terminated, truncated, info = self.training_env.step(action)
                #         second_total_reward += reward
                #         state=new_state

                #         if terminated or truncated:
                #             episode.Terminated = terminated
                #             episode.Truncated  = truncated
                #             break

                # assert total_reward == second_total_reward, f"Total reward is not the same. First: {total_reward}, Second: {second_total_reward}"

        return self.Metrics

    # def SaveToDisk(self, path : str):
    #     p = Path(path)
    #     with open(p / "Agent.json", mode="w") as save_file:
    #         json.dump({
    #                 "training_env"  : None,
    #                 "config"        : None,
    #                 "metrics"       : None
    #             },
    #             save_file
    #         )


    # def LoadFromDisk(self, path : str):
    #     p = Path(path)
    #     with open(p, mode="w") as save_file:


        
    def Infere(
        self, 
        env         : gym.Env, 
        policy_name : str | None    = None, 
        episodes    : int           = 1, 
        metrics     : AgentMetrics  = AgentMetrics(),
        verbose     : bool          = False
    ) -> AgentMetrics :
        policy = list(self.config.learning_policies.values())[0] if policy_name == None else self.config.learning_policies[policy_name]
        with policy.Inference(True):
            for _ in range(episodes):
                state, info = env.reset()
                episode     = Episode()

                while True:
                    action = self.config.experience_policy.NextAction(state)
                    maped_action = action + 4 if action < 2 else action - 2

                    new_state, reward, terminated, truncated, info = env.step(maped_action)

                    experience = Experience(
                        state       = state, 
                        action      = action, 
                        new_state   = new_state, 
                        reward      = reward, 
                        terminated  = terminated, 
                        truncated   = truncated
                    )
                    
                    state = new_state

                    if verbose:
                        print(experience.pretty())

                    episode.Experiences.append(experience)
                    metrics.ComputeStepMetrics(experience)

                    if terminated or truncated:
                        episode.Terminated = terminated
                        episode.Truncated  = truncated
                        metrics.ComputeEpisodeMetrics(episode)
                        break

        return metrics

    def playEpisode(self, mode : str = "train"):
        pass

    def emptyExperienceBuffer(self):
        for policy in self.config.learning_policies.values():
            policy.Learn(self.experience_buffer)
        self.experience_buffer  = []
        self.max_buffer_size    = next(self.config.max_buffer_size_generator)
    
class Academy:
    def __init__(
        self,
        env     : gym.Env,
        agents  : dict[str, AgentConfig],
        metrics : AgentMetrics | None = None
    ) -> None:
        self.agents = {
            name : Agent(
                training_env = deepcopy(env), 
                config       = config
            ) for name, config in agents.items()
        }
        if metrics is not None:
            for agent in self.agents.values():
                agent.Metrics = deepcopy(metrics) 

    #TODO multiprocessing
    def Train(self, epochs : int) -> dict[str, AgentMetrics]:
        for name, agent in self.agents.items():
            # print(f"Training agent {name}:")
            agent.Train(epochs)
        return {name : agent.Metrics for name, agent in self.agents.items()}

    def closeEnvs(self):
        for agent in self.agents.values():
            agent.training_env.close()

    #TODO multiprocessing
    def Infere(
        self, 
        env         : gym.Env, 
        agent_name  : str | None    = None, 
        policy_name : str | None    = None, 
        episodes    : int           = 1, 
        metrics     : AgentMetrics  = AgentMetrics()
    ) -> dict[str, AgentMetrics]:    
        agent_metrics = {}
        for name, agent in self.agents.items():
            if agent_name is not None and name != agent_name:
                continue

            copied_env = deepcopy(env)
            agent_metrics[name] = agent.Infere(
                env         = copied_env, 
                policy_name = policy_name,
                episodes    = episodes,
                metrics     = deepcopy(metrics),
                verbose     = agent_name is not None
            ) 
            copied_env.close()

        return agent_metrics
    
def get_drop_off_passager_callback(start_state: int, expected_reward: int = 20):
    def callback(agent: 'Agent'):
        # Use a different variable to maintain clarity and avoid accidental overwrites
        current_state = start_state
        agent.training_env.resetOnState(current_state)
        episode = Episode()
        agent.experience_buffer = []
        total_reward = 0  # To aggregate rewards if necessary

        while True:
            action = agent.config.experience_policy.NextAction(current_state)
            maped_action = action + 4 if action < 2 else action - 2

            new_state, reward, terminated, truncated, info = agent.training_env.step(maped_action)
            experience = Experience(
                state=current_state,  # Use current_state here
                action=action,
                new_state=new_state,
                reward=reward,
                terminated=terminated,
                truncated=truncated
            )
            episode.Experiences.append(experience)

            agent.experience_buffer.append(experience)
            
            agent.Metrics.ComputeStepMetrics(experience)

            if len(agent.experience_buffer) == agent.config.max_buffer_size:
                agent.emptyExperienceBuffer()
            
            current_state = new_state  # Update current_state for the next iteration
            
            if terminated or truncated:
                episode.Terminated = terminated
                episode.Truncated = truncated
                agent.Metrics.ComputeEpisodeMetrics(episode)
                break
        
        agent.emptyExperienceBuffer()
        
        # Determine the condition to return True or False based on your criteria
        return reward == expected_reward

    return callback


def get_state_from_destination(destination: int):
    position = [0, 0]
    if destination == 1:
        position = [0, 4]
    elif destination == 2:
        position = [4, 0]
    elif destination == 3:
        position = [4, 3]
    return ((position[0] * 5 + position[1]) *  5 + 4) * 4 + destination

def get_curriculum_training_callbacks():
    trainings= [
    # 1. train to drop-off passengers
    get_drop_off_passager_callback(get_state_from_destination(0), 20),
    get_drop_off_passager_callback(get_state_from_destination(1), 20),
    get_drop_off_passager_callback(get_state_from_destination(2), 20),
    get_drop_off_passager_callback(get_state_from_destination(3), 20),
    ]
    return trainings

def state_iterator():
    max_taxi_row = 4
    max_taxi_col = 4
    passenger_locations = [0, 1, 2, 3]  # Valid initial passenger locations
    destinations = [0, 1, 2, 3]         # Possible destinations

    def next_state():
        for taxi_row in range(max_taxi_row + 1):
            for taxi_col in range(max_taxi_col + 1):
                for passenger_location in passenger_locations:
                    for destination in destinations:
                        if passenger_location != destination:
                            current_state = ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
                            yield current_state

    return next_state()
