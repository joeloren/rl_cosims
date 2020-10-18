# basic imports
import argparse
import json
import traceback
from typing import Callable, Dict
import os
import warnings
from tqdm import tqdm
# scientific imports
import numpy as np
from gym import Env
from matplotlib import pyplot as plt
# our imports
from src.envs.online_knapsack.oks_simulation.simulator import Simulator
from src.envs.online_knapsack.oks_baselines.scripted_policies import random_policy, simple_policy
from src.envs.online_knapsack.oks_simulation.problem_generator import ItemGenerator


def evaluate_policy_simple(problems: Dict[int, Env], policy: Callable[[dict, Env], np.ndarray],
                           samples_per_seed=100, policy_name=None):
    """For num_seeds times, determine the mean policy reward by running it samples_per_seed times.
    :param problems: List[Simulator] - list of oks simulations
    :param policy_name: str - name is used for the plotting title
    :param policy: Policy - the policy used to create solution
    :param samples_per_seed: int - the number of times we execute the policy for each seed
    :return the empirical mean total rewards of the given policy. function returns a list the size of num_seeds
    """
    all_rewards = []
    i = 0
    if policy_name is None:
        policy_name = policy.__name__
    print(f"started running policy:{policy_name}")
    for seed, problem in tqdm(problems.items()):
        mean_reward = evaluate_policy_simple_single_seed(problem, policy, seed, samples_per_seed)
        all_rewards.append(mean_reward)
        i += 1
    return all_rewards


def evaluate_policy_simple_single_seed(problem: Simulator, policy: Callable[[dict, Env], np.ndarray], seed: int,
                                       samples_per_seed: int):
    total_rewards = []
    for j in range(samples_per_seed):
        problem.seed(seed)
        obs = problem.reset()
        total_reward = 0.0
        # reset policy if the method is available
        reset_method = getattr(policy, "reset", None)
        if callable(reset_method):
            reset_method(obs)

        completed = False
        while not completed:
            action = policy(obs, problem)
            obs, reward, completed, _ = problem.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)


def main():
    warnings.filterwarnings("ignore")
    POLICIES = ["random", "simple"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", type=str, default=["random", "simple"], nargs="+", choices=POLICIES, help="Policies to be tested")
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--update_results", action="store_true", help="update the existing json files with new results")
    args = parser.parse_args()
    np.random.seed(seed=args.start_seed)

    print("running evaluation code")
    values = {}
    policies = args.policies
    generator = ItemGenerator()
    print(policies)
    envs = {1: Simulator(max_steps=10, max_capacity=10, problem_generator=generator)}
    if "random" in policies:
        values["Uniformly Random"] = evaluate_policy_simple(envs, random_policy, samples_per_seed=5)
    if "simple" in policies:
        values["Simple Threshold"] = evaluate_policy_simple(envs, simple_policy, samples_per_seed=5)

    print(values)


if __name__ == "__main__":
    main()
    print("done")