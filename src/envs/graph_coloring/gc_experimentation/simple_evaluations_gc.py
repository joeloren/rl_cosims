# basic imports
import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Callable, Dict, Tuple
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
from src.envs.graph_coloring.gc_utils.plot_results import plot_gc_solution, plot_multiple_result_stats
from src.envs.graph_coloring.gc_baselines.simple_policies import random_policy_without_newcolor, random_policy
from src.envs.graph_coloring.gc_baselines.ortools_policy import ORToolsOfflinePolicy
from src.envs.graph_coloring.gc_policies.ppo_policy import PPOPolicy


def evaluate_policy_simple(problems: Dict[int, Env], policy: Callable[[Dict, Simulator], Tuple], save_solution=True,
                           samples_per_seed=100, policy_name=None):
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
    saved_solution = None
    for seed, problem in tqdm(problems.items()):
        save_solution = True
        if i != 0:
            save_solution = False
        mean_reward = evaluate_policy_simple_single_seed(problem, policy, seed, samples_per_seed)
        all_rewards.append(mean_reward)
        if save_solution:
            saved_solution = deepcopy(problem.current_state.graph)
            if len(problem.current_state.nodes_order) < 200:  # only plot solution if the number of nodes is smaller
                # than 200, to not overload the computer
                plt.figure()
                plot_gc_solution(graph=problem.current_state.graph, nodes_order=problem.current_state.nodes_order)
                plt.title(f"graph for policy:{policy_name}, reward:{-mean_reward:.1f}")
            # plt.show()
        i += 1
    return all_rewards, saved_solution


def evaluate_policy_simple_single_seed(problem: Simulator, policy: Callable[[Dict, Simulator], Tuple], seed: int,
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
    POLICIES = ["simple", "ortools", "ppo"]
    PROBLEMS = ["fixed", "er_random"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", type=str, default=[], nargs="+", choices=POLICIES, help="Policies to be tested")
    parser.add_argument("--problem", type=str, default="er_random", choices=PROBLEMS)
    parser.add_argument("--problem_path", type=str,
                        default="gc_experimentation/saved_problems/er_offline/er_offline.json")
    parser.add_argument("--ppo_model_folder", type=str, help="folder where ppo model is saved",
                        default="gc_experimentation/saved_problems/er_offline/ppo_models/15n_2020-10-25_11_37_07")
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--output_file", type=str, default="gc_experimentation/saved_problems/er_offline/results.json")
    parser.add_argument("--update_results", action="store_true", help="update the existing json files with new results")

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
    else:
        envs = None
    print("running evaluation code")
    values = {}
    saved_solutions = {}
    policies = args.policies
    if "simple" in policies:
        values["Random"], saved_solutions["Random"] = evaluate_policy_simple(envs, random_policy, samples_per_seed=1)
        values["Random W/O new color"], saved_solutions["Random W/O new color"] = (
            evaluate_policy_simple(envs, random_policy_without_newcolor, samples_per_seed=1)
        )

    if "ortools" in policies:
        ortools_policy = ORToolsOfflinePolicy(verbose=False, timeout=2000)  # timeout is in milli-seconds
        values["OrTools"], saved_solutions["OrTools"] = evaluate_policy_simple(envs, ortools_policy, samples_per_seed=1)

    expensive_policies = []
    if "ppo" in policies:
        with open(Path(args.ppo_model_folder) / "agent_params.json") as f:
            agent_config_dict = json.load(f)
        expensive_policies.append(("PPO",
                                   PPOPolicy(agent_config_dict, agent_config_dict["model_config"],
                                             args.ppo_model_folder, envs[args.start_seed]), envs))

    # Run evaluation of specified policies
    for policy_name, policy, env_list in expensive_policies:
        print("Evaluating: ", policy_name)
        values[policy_name], saved_solutions[policy_name] = evaluate_policy_simple(env_list, policy, samples_per_seed=5,
                                                                                   policy_name=policy_name)

    if args.update_results:
        with open(args.output_file) as fp:
            res = json.loads(fp.read())
        res.update(values)
        values = res
    plt.show()
    with open(args.output_file, "w") as f:
        json.dump(values, f, indent=4)
    plot_multiple_result_stats(policy_values=values, relative_to='Random W/O new color')


if __name__ == "__main__":
    main()
    print("done")
