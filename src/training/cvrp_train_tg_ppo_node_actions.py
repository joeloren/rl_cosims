import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from gym import Env
from tqdm import tqdm
from trains import Task

# import baseline policy
from src.envs.cvrp.cvrp_baselines.simple_baseline import distance_proportional_policy
from src.envs.cvrp.cvrp_baselines.or_tools_baseline import ORToolsPolicy
# import cvrp simulation -
from src.envs.cvrp.cvrp_wrappers.cvrp_all_customers_torch_geometric_attention_wrapper import GeometricAttentionWrapper
# import problem creator
from src.envs.cvrp.cvrp_experimentation.problems import (create_uniform_dynamic_problem, create_fixed_static_problem)
# import RL algorithm -
from src.agents.tg_ppo_agent import PPOAgent
from src.models.tg_node_action_models import PolicyFullyConnectedMessagePassing


def evaluate_policy_simple(problem: Env,
                           seeds: np.ndarray,
                           policy: Callable[[dict, Env], np.ndarray],
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


def evaluate_policy_simple_single_seed(problem: Env, policy: Callable[[dict, Env], np.ndarray], seed: int,
                                       samples_per_seed: int):
    total_rewards = []
    for j in range(samples_per_seed):
        problem.seed(seed)
        obs = problem.reset()
        total_reward = 0.0

        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset(obs)
        completed = False
        while not completed:
            action_probs = policy(obs, problem)
            act = np.random.choice(len(obs["action_mask"]), p=action_probs)
            obs, reward, completed, _ = problem.step(act)
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)


def main():
    # Init environment
    use_trains = False
    problem_name = 'cvrp'
    problem_type = 'uniform_offline'
    max_customer_times = 0
    size = 5
    vehicle_velocity = 1
    vehicle_capacity = 100
    random_seed = 0
    max_demand = 10
    start_at_depot = True
    EVAL_BASELINES_RESULTS_FILENAME = (f"experiments/{problem_name}/{size}s_{vehicle_capacity}c_{max_customer_times}t/"
                                       f"baseline_values.json")

    env_config = {'problem_type': problem_type,
                  'max_customer_times': max_customer_times,
                  'size': size,
                  'max_demand': max_demand,
                  'vehicle_velocity': vehicle_velocity,
                  'vehicle_capacity': vehicle_capacity,
                  'start_at_depot': start_at_depot,
                  'random_seed': random_seed,
                  'eval_baseline_results_filename': EVAL_BASELINES_RESULTS_FILENAME}
    if use_trains:
        task = Task.init(
            project_name="train_cvrp_pytorch",
            task_name=f'train_ppo_agent_{size}s_{vehicle_capacity}c_{max_customer_times}t'
        )
        logger = Task.current_task().get_logger()
        logger.tensorboard_single_series_per_graph(single_series=True)
    else:
        logger = None

    env = create_uniform_dynamic_problem(max_customer_times=max_customer_times, size=size, max_demand=max_demand,
                                         vehicle_velocity=vehicle_velocity, vehicle_capacity=vehicle_capacity,
                                         random_seed=random_seed, start_at_depot=start_at_depot)

    # customer_positions = [[0.25, 0.25], [0.5, 0.5], [1, 1]]
    # env = create_fixed_static_problem(customer_positions=customer_positions,
    #                                   depot_position=[0, 0],
    #                                   initial_vehicle_capacity=10,
    #                                   initial_vehicle_position=[0, 0],
    #                                   customer_demands=[1]*len(customer_positions),
    #                                   customer_times=[0]*len(customer_positions),
    #                                   vehicle_velocity=1)
    #
    # env_config = {'problem_type': 'fixed_problem',
    #               'size': 3,
    #               'vehicle_capacity': 10,
    #               'vehicle_position': [0, 0],
    #               'customer_positions': customer_positions,
    #               'start_at_depot': True
    #               }
    # EVAL_BASELINES_RESULTS_FILENAME = (f'experiments/{3}s_{10}c_{0}t/'
    #                                    f'baseline_values.json')

    tg_env = GeometricAttentionWrapper(env)
    tg_env.reset()

    # model_config = {
    #     'use_value_critic': True,
    #     'num_features': 4,
    #     'embedding_dim': 128,
    #     'value_embedding_dim': 128,
    #     'use_batch_norm': False
    # }
    model_config = {
        'n_passes': 4,
        'edge_embedding_dim': 64,
        'node_embedding_dim': 64,
        'global_embedding_dim': 64,
        'edge_hidden_dim': 64,
        'edge_target_dim': 64,
        'node_target_dim': 64,
        'node_dim_out': 1,
        'edge_dim_out': 1,
        'node_hidden_dim': 64,
        'global_hidden_dim': 64,
        'global_target_dim': 64,
        'global_dim_out': 64,
        'edge_feature_dim': 1,
        'node_feature_dim': 5,  # indicator, x, y, demand/capacity, is_visited
        'global_feature_dim': 1,
        'value_embedding_dim': 64,
        'use_value_critic': True,
        'use_batch_norm': False
    }

    agent_config = {
        'lr': 0.0001,
        'discount': 0.90,
        # number of episodes to do altogether
        'number_of_episodes': 50000,
        # a batch is N episodes where N is number_of_episodes_in_batch
        'number_of_episodes_in_batch': 60,  # this must be a division of number of episodes
        'total_num_eval_seeds': 2,
        'num_eval_seeds': 2,
        'evaluate_every': 50,
        'num_train_seeds': 2,
        'reward_average_window_size': 10,
        'entropy_coeff': 0.01,  # consider decreasing this back
        'value_coeff': 0.3,
        'model_config': model_config,
        'save_checkpoint_every': 1000,
        'eps_clip': 0.2,
        'n_ppo_updates': 40,
        'target_kl': 0.005,
        'logit_normalizer': 5,
        'problem_name': problem_name  # used for saving results
    }
    model_config['logit_normalizer'] = agent_config['logit_normalizer']
    agent_config['run_name'] = f"ep_in_batch_{agent_config['number_of_episodes_in_batch']}_" \
                               f"n_eval_{agent_config['num_eval_seeds']}_lr_{agent_config['lr']}"
    eval_seeds = list(range(agent_config['total_num_eval_seeds']))
    baseline_results_path = Path(EVAL_BASELINES_RESULTS_FILENAME)
    or_tools_policy = ORToolsPolicy(timeout=10)
    if not baseline_results_path.exists():
        baseline_values = {
            'distance': evaluate_policy_simple(env, eval_seeds, distance_proportional_policy, samples_per_seed=5),
            'ORTools': evaluate_policy_simple(env, eval_seeds, or_tools_policy, samples_per_seed=5)
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

    # model = PolicyFullyConnectedGAT(cfg=model_config, model_name='ppo_policy_model')
    model = PolicyFullyConnectedMessagePassing(cfg=model_config, model_name='ppo_message_passing_model')
    set_seeds()
    if use_trains:
        parameters_agent = task.connect(agent_config, name='agent_config')
        parameters_env = task.connect(env_config, name='env_config')
    agent_config['env_config'] = env_config
    ppo_agent = PPOAgent(tg_env,
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
