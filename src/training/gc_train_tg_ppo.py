import json
from pathlib import Path
from typing import Callable, Tuple, List

import numpy as np
import torch
from tqdm import tqdm
from trains import Task

# import baseline policy
from src.envs.graph_coloring.gc_baselines.ortools_policy import ORToolsOfflinePolicy
from src.envs.graph_coloring.gc_baselines.simple_policies import random_policy_without_newcolor
# import gc simulation -
from src.envs.graph_coloring.gc_simulation.simulator import Simulator
from src.envs.graph_coloring.gc_wrappers.gc_torch_geometric_wrappers import GraphWithColorsWrapper as TgWrapper
# import problem creator
from src.envs.graph_coloring.gc_experimentation.problems import (create_fixed_static_problem,
                                                                 create_er_random_graph_problem)
# import RL algorithm -
from src.agents.tg_ppo_agent import PPOAgent
from src.models.tg_models import PolicyGNN as PolicyModel


def evaluate_policy_simple(problem: Simulator,
                           seeds: List,
                           policy: Callable[[dict, Simulator], Tuple],
                           samples_per_seed=100):
    """
    For num_seeds times, determine the mean policy reward by running it samples_per_seed times.
        :param problem the `GymWrapper` simulation
        :param seeds the list of seeds (generate problem instances)
        :param policy the policy whose value to estimate
        :param samples_per_seed the number of times to execute the policy for each seed
        :return the empirical mean total rewards of the given policy, a list of length num_seeds

    """
    return {
        seed: evaluate_policy_simple_single_seed(problem=problem,
                                                 policy=policy,
                                                 seed=seed,
                                                 samples_per_seed=samples_per_seed) for seed in tqdm(seeds)
    }


def evaluate_policy_simple_single_seed(problem: Simulator, policy: Callable[[dict, Simulator], Tuple], seed: int,
                                       samples_per_seed: int, reward_function: str = "sequential"):
    total_rewards = []
    for j in range(samples_per_seed):
        problem.seed(seed)
        obs = problem.reset()
        total_reward = 0.0

        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset(obs)
        completed = False
        reward_list = []
        while not completed:
            act = policy(obs, problem)
            obs, reward, completed, _ = problem.step(act)
            total_reward += reward
            reward_list.append(reward)
        if reward_function == "num_colors":
            total_reward = problem.get_number_of_colors_used()
        total_rewards.append(total_reward)
    return np.mean(total_rewards)


def main():
    # Init environment
    use_trains = False
    problem_name = 'gc'
    problem_type = 'er_offline'
    num_new_nodes = 0
    num_initial_nodes = 15
    prob_edge = 0.3
    is_online = False
    random_seed = 0
    if use_trains:
        task = Task.init(
            project_name="train_gc_pytorch",
            task_name=f'train_ppo_agent_{num_initial_nodes}n_{num_new_nodes}new_n_{prob_edge}p'
        )
        logger = Task.current_task().get_logger()
        logger.tensorboard_single_series_per_graph(single_series=True)
    else:
        task = None

    env = create_er_random_graph_problem(num_new_nodes=num_new_nodes, num_initial_nodes=num_initial_nodes,
                                         prob_edge=prob_edge, is_online=is_online, random_seed=random_seed)

    env_tg = TgWrapper(env)
    env_tg.reset()

    model_config = {
        'n_passes': 4,
        'edge_embedding_dim': 128,
        'node_embedding_dim': 128,
        'global_embedding_dim': 128,
        'edge_hidden_dim': 128,
        'edge_target_dim': 128,
        'node_target_dim': 128,
        'node_dim_out': 128,
        'edge_dim_out': 1,
        'node_hidden_dim': 128,
        'global_hidden_dim': 128,
        'global_target_dim': 128,
        'global_dim_out': 128,
        'edge_feature_dim': 1,
        'node_feature_dim': 2,  # indicator, color
        'global_feature_dim': 1,
        'value_embedding_dim': 128,
        'use_value_critic': True,
        'use_batch_norm': False
    }

    agent_config = {
        'lr': 0.0001,
        'discount': 0.95,
        # number of episodes to do altogether
        'number_of_episodes': 50000,
        # a batch is N episodes where N is number_of_episodes_in_batch
        'number_of_episodes_in_batch': 40,  # this must be a division of number of episodes
        'total_num_eval_seeds': 100,
        'num_eval_seeds': 10,
        'evaluate_every': 50,
        'num_train_seeds': 100,
        'reward_average_window_size': 10,
        'entropy_coeff': 0.01,  # consider decreasing this back
        'value_coeff': 0.3,
        'model_config': model_config,
        'save_checkpoint_every': 1000,
        'eps_clip': 0.5,
        'n_ppo_updates': 20,
        'target_kl': 0.005,
        'logit_normalizer': 10,
        'problem_name': problem_name   # used for saving results
    }

    EVAL_BASELINES_RESULTS_FILENAME = (f"experiments/{problem_name}/"
                                       f"{agent_config['total_num_eval_seeds']}n-seeds_{num_initial_nodes}n_"
                                       f"{num_new_nodes}new_n_{prob_edge}p/"
                                       f"baseline_values.json")

    env_config = {'problem_type': problem_type,
                  'num_new_nodes': num_new_nodes,
                  'num_initial_nodes': num_initial_nodes,
                  'prob_edge': prob_edge,
                  'is_online': is_online,
                  'random_seed': random_seed,
                  'eval_baseline_results_filename': EVAL_BASELINES_RESULTS_FILENAME}
    agent_config['run_name'] = f"ep_in_batch_{agent_config['number_of_episodes_in_batch']}_" \
                               f"n_eval_{agent_config['num_eval_seeds']}_lr_{agent_config['lr']}"
    eval_seeds = list(range(agent_config['total_num_eval_seeds']))
    baseline_results_path = Path(EVAL_BASELINES_RESULTS_FILENAME)
    or_tools_policy = ORToolsOfflinePolicy(timeout=1000)
    if not baseline_results_path.exists():
        baseline_values = {
            'random_wo_nc': evaluate_policy_simple(env, eval_seeds, random_policy_without_newcolor, samples_per_seed=1),
            'ORTools': evaluate_policy_simple(env, eval_seeds, or_tools_policy, samples_per_seed=1)
        }
        baseline_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_results_path, 'w') as f:
            json.dump(baseline_values, f, indent=2)
    else:
        print(f"loading: {EVAL_BASELINES_RESULTS_FILENAME}")
        with open(baseline_results_path, 'r') as f:
            baseline_values = json.load(f)
            # JSON saves dictionary keys as strings, so we have to convert them back to ints
            baseline_values = {
                baseline: {int(seed): val for seed, val in baseline_dict.items()
                           } for baseline, baseline_dict in baseline_values.items()
            }

    model = PolicyModel(cfg=model_config, model_name='ppo_policy_model')
    set_seeds()
    if use_trains:
        task.connect(agent_config, name='agent_config')
        task.connect(env_config, name='env_config')
    agent_config['env_config'] = env_config
    ppo_agent = PPOAgent(env_tg,
                         config=agent_config,
                         model=model,
                         eval_seeds=eval_seeds,
                         baseline_eval_values=baseline_values)
    ppo_agent.train()


def set_seeds():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()
