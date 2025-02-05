from typing             import Union, Generator, Iterator
from markov.DataModels  import Episode, Experience
from markov.Policies.Policy    import Policy
from markov.Metrics import EpisodeMetricFunction, StepMetricFunction

import gym


class AgentTester:
    def __init__(
        self,
        training_env            : gym.Env,
        experience_policy       : Policy,
        learning_policies       : dict[str, Policy],
        experience_buffer_size  : Union[int, Iterator[int], Generator[int, None, None]],
        step_metrics            : dict[str, StepMetricFunction]    = {},
        episode_metrics         : dict[str, EpisodeMetricFunction] = {}
    ):
        experience_policy.InitEnv(training_env)
        for policy in learning_policies.values():
            policy.InitEnv(training_env)

        self.training_env       = training_env
        self.experience_policy  = experience_policy
        self.learning_policies  = learning_policies
        self.step_metrics       = step_metrics
        self.episode_metrics    = episode_metrics

        self.metrics            = {
            "step"      : { metric:[] for metric, _ in step_metrics.items() },
            "episode"   : { metric:[] for metric, _ in episode_metrics.items() },
        }

        if isinstance(experience_buffer_size, int):
            self.max_buffer_size_generator = self._generate_infinite(experience_buffer_size)
        else:
            self.max_buffer_size_generator = experience_buffer_size
        self.max_buffer_size = next(self.max_buffer_size_generator)

    def _generate_infinite(self, value: int) -> Iterator[int]:
        while True:
            yield value
        
    def Train(self, epochs : int):
        for epoch in range(epochs):
            self.experience_buffer  = []
            state, info             = self.training_env.reset()
            episode                 = Episode()

            while True:
                action = self.experience_policy.NextAction(state)
            
                new_state, reward, terminated, truncated, info = self.training_env.step(action)

                experience = Experience(state, action, new_state, reward)
                self.experience_buffer.append(experience)
                episode.Experiences.append(experience)
                
                for name, func  in self.step_metrics.items():
                    self.metrics["step"][name].append(func(experience))

                if len(self.experience_buffer) == self.max_buffer_size:
                    self.emptyExperienceBuffer()
                
                state = new_state
                
                if terminated or truncated:
                    episode.Terminated = terminated
                    episode.Truncated  = truncated
                    for name, func in self.episode_metrics.items():
                        self.metrics["episode"][name].append(func(episode))
                    break
            
            self.emptyExperienceBuffer()

        
    def Infere(
        self, 
        env : gym.Env, 
        policy_name : str | None = None, 
        games : int = 1, 
    ):
        policy = list(self.learning_policies.values())[0] if policy_name == None else self.learning_policies[policy_name]
        with policy.Inference(True):
            for game in range(games):
                state, info = env.reset()
                while True:
                    action = self.experience_policy.NextAction(state)
                    new_state, reward, terminated, truncated, info = env.step(action)
                    state = new_state
                    if terminated or truncated:
                        break

    def emptyExperienceBuffer(self):
        for policy in self.learning_policies.values():
            policy.Learn(self.experience_buffer)
        self.experience_buffer  = []
        self.max_buffer_size    = next(self.max_buffer_size_generator)
    