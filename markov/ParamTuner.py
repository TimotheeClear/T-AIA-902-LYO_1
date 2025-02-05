from ast import Call
from cgi import print_arguments
import numpy as np
import sys, gym
sys.path.append('./')
from markov.Policies import EpsilonGreedyPolicy, RandomPolicy, QTablePolicy
from markov.Training import AgentConfig, Academy, AgentMetrics
from markov.Metrics import ModelReadiness
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
import gym
from typing import Callable, Dict, Iterator, Union, List, Optional, Tuple
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import warnings
warnings.filterwarnings('ignore')

num_cores = os.cpu_count() or 1
# Define the hyperparameter space
space: List[Union[Real, Integer]] = [
    Real(0.01, 1.0, name="alpha"),  # Learning rate
    Real(0.01, 1.0, name="epsilon"),  # Exploration rate
    Real(0.01, 1.0, name="gamma")  # Discount factor
]

# This function now expects a single dictionary of parameters
def train_model(params, nb_agents, nb_episodes):
    # Configure multiple agents with the current parameter set
    agent_configs = {}
    for i in range(nb_agents):  # Create five agents
        agent_name = f"agent_{i}"
        agent_configs[agent_name] = AgentConfig(
            policy=EpsilonGreedyPolicy(
                epsilon=params['epsilon'],
                exploration_policy=RandomPolicy(),
                exploitation_policy=QTablePolicy(alpha=params['alpha'], gamma=params['gamma'])
            ),
            model_readiness=True,
            early_stop=True
        )

    # Initialize the RL environment and academy
    academy = Academy(
        env=gym.make("Taxi-v3"),
        agents=agent_configs,
        metrics=AgentMetrics(
            model_readiness_metrics={"ModelReadiness": ModelReadiness}
        )
    )

    # Train the model for a fixed number of episodes
    training_metrics = academy.Train(nb_episodes)
    academy.closeEnvs()  # Clean up resources

    target_max_readiness = 2379  # Ideal maximum readiness
    results= []
    # Lists to store the rates of improvement and the shortfall from the target readiness for each agent.
    for agent_name, agent_results in training_metrics.items():
        model_readiness_scores = agent_results.results["model_readiness"]["ModelReadiness"]
        # get max
        max_score = max(model_readiness_scores)  # Get the highest score achieved
       # Determine the episode index when the target score was first achieved
        if target_max_readiness in model_readiness_scores:
            time= model_readiness_scores.index(target_max_readiness) + 1
        else:
            time= len(model_readiness_scores)  # Use the total episodes if the target was never reached

        quality = max_score - target_max_readiness - 1
        result = abs(quality) * time 

        results.append(result)
    

    success = np.median(results)
    print(f"Median adjusted score: {success}")
    print("")

    return float(success)

class ParamTuner:
    def __init__(
        self,
        random_state : int,
        nb_agents : int= 5,
        n_calls : int=30,
        nb_episodes : int =500,
        space : list[Real | Integer]=space,
        train_model : Callable[[Dict, int, int], float] = train_model
    ):
        self.space = space
        self.n_calls = n_calls
        self.random_state = random_state
        self.nb_agents = nb_agents
        self.nb_episodes = nb_episodes
        self.train_model = train_model
    
    def objective(self, **params):
        print('Params tested:')
        print(params)
        return self.train_model(params, self.nb_agents, self.nb_episodes)

    
    def optimize_model(self):
        decorated_objective = use_named_args(self.space)(self.objective)  # Decorate the objective method
        result = gp_minimize(decorated_objective, self.space, n_calls=self.n_calls, n_jobs=num_cores, verbose=True)
        best_params = {dim.name: result.x[i] for i, dim in enumerate(self.space)}
        print("Best parameters found:", best_params)
        return best_params
    
class TuningExperiment:
    def __init__(self, 
                 start_episodes: int, 
                 end_episodes: int, 
                 num_tuners: int, 
                 random_state: int,
                 nb_agents: int,
                 n_calls: int,
        ):
        self.start_episodes = start_episodes
        self.end_episodes = end_episodes
        self.num_tuners = num_tuners
        self.random_state = random_state
        self.nb_agents = nb_agents
        self.n_calls = n_calls
        self.results = {}

    def distribute_episodes(self) -> List[int]:
        """Distribute episodes linearly across the number of tuners."""
        # np.linspace returns a NumPy array, convert it to a list of integers
        return np.linspace(self.start_episodes, self.end_episodes, num=self.num_tuners, dtype=int).tolist()

    def run_tuners(self):
        """ Run each tuning session with a specific number of episodes. """
        episode_counts = self.distribute_episodes()
        for episodes in episode_counts:
            local_random_state = self.random_state + episodes  # simple variation by episodes
            print(f"Parameter tuning n° {episodes}/{episode_counts}")

            tuner = ParamTuner(
                random_state=local_random_state,
                nb_episodes=episodes,
                nb_agents=self.nb_agents,
                n_calls=self.n_calls
            )
            best_params = tuner.optimize_model()
            self.results[episodes] = best_params

            print(f"Best Params: {best_params}")
            print(self.results)
            self.plot_individual_parameters()
            self.plot_all_parameters()
            print("")

    def get_results(self) -> Dict[int, Dict[str, float]]:
        """Retrieve the collected tuning results."""
        return self.results
    
    def plot_individual_parameters(self):
        tuning_results = self.results
        if not tuning_results:
            print("No tuning results to plot.")
            return

        # Assuming tuning_results is a dictionary structured as {episodes: {param1: value1, param2: value2, ...}}
        episodes = sorted(tuning_results.keys())

        # Find all parameters
        example_result = next(iter(tuning_results.values()))
        parameters = example_result.keys()

        for param in parameters:
            # Extract values for this parameter across all episodes
            values = [tuning_results[ep][param] for ep in episodes]

            # Set up the plot
            plt.figure(figsize=(10, 5))
            plt.plot(episodes, values, marker='o', label=param)
            plt.title(f'Tuning Results for {param}')
            plt.xlabel('Number of Episodes')
            plt.ylabel(f'Value of {param}')
            plt.ylim(min(values), max(values))  # Set y-axis limits to min and max values of the parameter
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{param}_{len(values)}.png")


    def plot_all_parameters(self):
        tuning_results = self.results
        if not tuning_results:
            print("No tuning results to plot.")
            return

        # Prepare plot
        plt.figure(figsize=(10, 5))
        episodes = sorted(tuning_results.keys())
        markers = ['o', '^', 's', 'D', 'x']
        colors = ['b', 'g', 'r', 'c', 'm']

        # Assuming the same set of parameters in all results
        example_result = next(iter(tuning_results.values()))
        parameters = example_result.keys()
        length = 0
        for idx, param in enumerate(parameters):
            values = [tuning_results[ep][param] for ep in episodes]
            length = len(values)
            plt.plot(episodes, values, marker=markers[idx % len(markers)], color=colors[idx % len(colors)], label=param)

        plt.title('Tuning Results for All Parameters')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Parameter Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"all_params_{length}.png")
    

# experiment = TuningExperiment(
#     start_episodes=150, 
#     end_episodes=665, 
#     num_tuners=14, 
#     n_calls=75,
#     nb_agents=10,
#     random_state=0
#     )
# experiment.run_tuners()
# tuner =  ParamTuner(
#                 random_state=0,
#                 nb_episodes=151,
#                 nb_agents=10,
#                 n_calls=75
#             )
# best_params = tuner.optimize_model()