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
from src.envs.graph_coloring.gc_simulation.simulator import Simulator
from src.envs.graph_coloring.gc_experimentation.problems import (create_fixed_static_problem,
                                                                 create_er_random_graph_problem)
from src.envs.graph_coloring.gc_utils.plot_results import plot_gc_solution
from src.envs.graph_coloring.gc_baselines.simple_policies import random_policy_without_newcolor as random_policy
from src.envs.graph_coloring.gc_baselines.ortools_policy import ORToolsOfflinePolicy


def evaluate_policy_simple(problems: Dict[int, Env], policy: Callable[[dict, Env], np.ndarray], save_solution=True,
                           samples_per_seed=100, policy_name=None, reward_function="num_colors"):
    """For num_seeds times, determine the mean policy reward by running it samples_per_seed times.
    :param problems: List[Simulator] - list of gc simulations
    :param policy_name: str - name is used for the plotting title
    :param save_solution: bool - (True if we want to save the first solution created by policy)
    :param policy: Policy - the policy used to create solution
    :param samples_per_seed: int - the number of times we execute the policy for each seed
    :param reward_function: str - the reward function chosen
    :return the empirical mean total rewards of the given policy. function returns a list the size of num_seeds
    """
    all_rewards = []
    i = 0
    if policy_name is None:
        policy_name = policy.__name__
    print(f"started running policy:{policy_name}")
    for seed, problem in tqdm(problems.items()):
        if i != 0:
            save_solution = False
        mean_reward = evaluate_policy_simple_single_seed(problem, policy, seed, samples_per_seed,
                                                         reward_function=reward_function)
        all_rewards.append(mean_reward)
        if save_solution is not None:
            plot_gc_solution(graph=problem.current_state.graph, nodes_order=problem.current_state.nodes_order)
            plt.title(f"graph for policy:{policy_name}, reward:{-mean_reward:.1f}")
            plt.show()

        i += 1
    return all_rewards


def evaluate_policy_simple_single_seed(problem: Simulator, policy: Callable[[dict, Env], np.ndarray], seed: int,
                                       samples_per_seed: int, reward_function: str):
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
        if reward_function == "num_colors":
            total_reward = problem.get_number_of_colors_used()
        total_rewards.append(total_reward)
    return np.mean(total_rewards)


def main():
    warnings.filterwarnings("ignore")
    POLICIES = ["simple", "ortools"]
    PROBLEMS = ["fixed", "er_random"]
    REWARD_FUNCTIONS = ["sequential", "num_colors"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", type=str, default=[], nargs="+", choices=POLICIES, help="Policies to be tested")
    parser.add_argument("--problem", type=str, default="fixed", choices=PROBLEMS)
    parser.add_argument("--problem_path", type=str, default="gc_experimentation/saved_problems/fixed/fixed.json")
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--output_file", type=str, default="gc_experimentation/saved_problems/fixed/results.json")
    parser.add_argument("--update_results", action="store_true", help="update the existing json files with new results")
    parser.add_argument("--reward_function", type=str, default="num_colors", choices=REWARD_FUNCTIONS,
                        help="the reward to use (either sequential reward used for rl or number of colors used")

    args = parser.parse_args()
    np.random.seed(seed=args.start_seed)
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(args.problem_path, "r") as f:
        problem_params = json.load(f)
    if args.problem == "fixed":
        envs = {
            args.start_seed + seed: create_fixed_static_problem(**problem_params, random_seed=args.start_seed + seed)
            for seed in range(args.num_seeds)
        }
    elif args.problem == "er_random":
        envs = {
            args.start_seed + seed: create_er_random_graph_problem(**problem_params, random_seed=args.start_seed + seed)
            for seed in range(args.num_seeds)
        }
    print("running evaluation code")
    values = {}
    policies = args.policies
    if "simple" in policies:
        values["Uniformly Random"] = evaluate_policy_simple(envs, random_policy, samples_per_seed=5)

    if "ortools" in policies:
        ortools_policy = ORToolsOfflinePolicy(verbose=False, timeout=500)  # timeout is in milli-seconds
        values["OrTools"] = evaluate_policy_simple(envs, ortools_policy, samples_per_seed=5)

    expensive_policies = []
    # Run evaluation of specified policies
    for policy_name, policy, env_list in expensive_policies:
        print("Evaluating: ", policy_name)
        values[policy_name] = evaluate_policy_simple(env_list, policy, samples_per_seed=5,
                                                     policy_name=policy_name)

    if args.update_results:
        with open(args.output_file) as fp:
            res = json.loads(fp.read())
        res.update(values)
        values = res

    with open(args.output_file, "w") as f:
        json.dump(values, f, indent=4)


if __name__ == "__main__":
    main()
    print("done")
