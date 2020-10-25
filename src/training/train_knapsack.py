import json
from pathlib import Path
from typing import Callable, List

from gym import Env
import numpy as np
import torch
from trains import Task
from tqdm import tqdm

from src.utils.logx import setup_logger_kwargs
from src.models.core_models import MLPActorCritic
from src.envs.online_knapsack.oks_simulation.problem_generator import ItemGenerator
from src.envs.online_knapsack.oks_simulation.simulator import Simulator
from src.envs.online_knapsack.oks_wrappers.array_wrapper import KnapsackArrayWrapper
from src.training.torch_utils import get_available_device
from src.agents.ppo_agent import PPOAgent
from src.utils.utils import set_seeds
from src.envs.online_knapsack.oks_baselines.scripted_policies import simple_policy, random_policy


def evaluate_policy_simple(problem: Env,
                           seeds: List[int],
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
            act = policy(obs, problem)
            obs, reward, completed, _ = problem.step(act)
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='knapsack-ppo')
    parser.add_argument('--output', type=str, help='Where to save results')
    parser.add_argument('--items', type=int, default=50, help='Number of items in knapsack instances')
    parser.add_argument('--capacity', type=float, default=10, help='Total initial capacity of the knapsack')
    parser.add_argument('--no_gpu', action="store_true", help='Force CPU usage.')
    parser.add_argument('--trains', action="store_true", help='Use trains to log training progress.')
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    EVAL_BASELINES_RESULTS_FILENAME = (f'experiments/{args.items}n_{args.capacity}c/'
                                       f'baseline_values.json')
    dict_env = Simulator(max_steps=args.items, max_capacity=args.capacity,
                                         problem_generator=ItemGenerator())
    env = KnapsackArrayWrapper(dict_env)

    model_config = dict(observation_space=env.observation_space, action_space=env.action_space,
                        hidden_sizes=[args.hid] * args.l, activation=torch.nn.ReLU)

    agent_config = dict(
        output_dir=args.output,
        lr=0.0001,
        discount=0.95,
        # number of episodes to do altogether
        number_of_episodes=50000,
        # a batch is N episodes where N is number_of_episodes_in_batch
        number_of_episodes_in_batch=20,  # this must be a division of number of episodes
        total_num_eval_seeds=10,
        num_eval_seeds=2,
        evaluate_every=50,
        num_train_seeds=2,
        reward_average_window_size=10,
        entropy_coeff=0.01,  # consider decreasing this back
        value_coeff=0.3,
        minibatch_size=256,
        # model_config=model_config,
        save_checkpoint_every=1000,
        eps_clip=0.5,
        n_ppo_updates=20,
        target_kl=0.005,
        logit_normalizer=10)

    agent_config['run_name'] = f"ep_in_batch_{agent_config['number_of_episodes_in_batch']}_" \
                               f"n_eval_{agent_config['num_eval_seeds']}_lr_{agent_config['lr']}"
    agent_config['policy_model_class'] = 'MLPActorCritic'
    set_seeds()
    if args.trains:
        task = Task.init(
            project_name="train_knapsack_pytorch",
            task_name=f'train_ppo_agent_{args.items}n_{args.capacity}c'
        )
        logger = Task.current_task().get_logger()
        logger.tensorboard_single_series_per_graph(single_series=True)
    else:
        logger = None

    # First compute results baseline algorithms
    baseline_results_path = Path(EVAL_BASELINES_RESULTS_FILENAME)
    eval_seeds = list(range(agent_config['total_num_eval_seeds']))

    if not baseline_results_path.exists():
        baseline_values = {
            'Random': evaluate_policy_simple(dict_env, eval_seeds, random_policy, samples_per_seed=5),
            'Simple': evaluate_policy_simple(dict_env, eval_seeds, simple_policy, samples_per_seed=1)
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

    device = "cpu" if args.no_gpu else get_available_device()
    agent_model = MLPActorCritic(**model_config)
    ppo_agent = PPOAgent(env=env, config=agent_config, model=agent_model, eval_seeds=eval_seeds,
                         baseline_eval_values=baseline_values)
    ppo_agent.train()


if __name__ == "__main__":
    main()
