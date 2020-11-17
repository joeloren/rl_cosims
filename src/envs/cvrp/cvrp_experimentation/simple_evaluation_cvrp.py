import argparse
import json
import traceback
from typing import Callable, Dict
from trains import Task
from matplotlib import pyplot as plt
import os
from pathlib import Path
import warnings

import numpy as np
from tqdm import tqdm

from src.envs.cvrp.cvrp_simulation.simulator import CVRPSimulation
from src.envs.cvrp.cvrp_baselines.simple_baseline import (
    random_policy,
    distance_proportional_policy,
)
from src.envs.cvrp.cvrp_baselines.or_tools_baseline import ORToolsPolicy
from src.envs.cvrp.cvrp_utils.plot_results import plot_vehicle_routes
from src.envs.cvrp.cvrp_experimentation.problems import (
    create_fixed_static_problem,
    create_uniform_dynamic_problem,
    create_mixture_guassian_dynamic_problem,
)
from src.envs.cvrp.cvrp_baselines.sweep_baseline import SweepPolicy
from src.envs.cvrp.cvrp_policies.ppo_policy import PPOPolicy


def evaluate_policy_simple(problems: Dict[int, CVRPSimulation],
                           policy: Callable[[dict, CVRPSimulation], np.ndarray],
                           samples_per_seed=100,
                           logger=None,
                           save_routes=False,
                           policy_name=None):
    """For num_seeds times, determine the mean policy reward by running it samples_per_seed times.
    :param problems of type cvrp simulations
    :param logger is the trains logger
    :param policy_name name to use for the plotting title
    :param save_routes (True if we want to save the first route created by policy)
    :param policy the policy whose value to estimate
    :param samples_per_seed the number of times to execute the policy for each seed
    :return the empirical mean total rewards of the given policy, a list of length num_seeds
    """
    all_rewards = []
    i = 0
    if policy_name is None:
        policy_name = policy.__name__
    print(f"started running policy:{policy_name}")
    for seed, problem in tqdm(problems.items()):
        if i != 0:
            save_routes = False
        mean_reward, vehicle_route, all_actions = evaluate_policy_simple_single_seed(
            problem, policy, seed, samples_per_seed, save_routes
        )
        all_rewards.append(mean_reward)
        if logger is not None:
            logger.report_scalar("reward", policy_name, -mean_reward, i)
            Task.current_task().upload_artifact("actions", np.array(all_actions))
        if vehicle_route is not None:
            ax = plot_vehicle_routes(
                depot_position=problem.current_state.depot_position,
                customer_positions=problem.current_state.customer_positions,
                customer_demands=problem.current_state.customer_demands,
                veh_route=vehicle_route,
            )
            ax.set_title(f"route for policy:{policy_name}, reward:{-mean_reward:.1f}")
            plt.show()
            # plt.pause(0.1)
            # plt.close()

        i += 1
    return all_rewards


def evaluate_policy_simple_single_seed(problem: CVRPSimulation,
                                       policy: Callable[[dict, CVRPSimulation], np.ndarray],
                                       seed: int, samples_per_seed: int,
                                       save_routes: bool = False):
    total_rewards = []
    all_actions = []
    for j in range(samples_per_seed):
        problem.seed(seed)
        obs = problem.reset()
        if save_routes:
            if j == 0:
                vehicle_route = {
                    0: {
                        "x": [problem.current_state.current_vehicle_position[0]],
                        "y": [problem.current_state.current_vehicle_position[1]],
                        "total_demand": 0,
                    }
                }
                route_num = 0
        else:
            vehicle_route = None
        total_reward = 0.0
        try:
            # reset policy if the method is available
            reset_method = getattr(policy, "reset", None)
            if callable(reset_method):
                reset_method(obs)
            completed = False
            while not completed:
                if obs["customer_positions"].size > 0:
                    action_probs = policy(obs, problem)
                    act = np.random.choice(len(obs["action_mask"]), p=action_probs)
                    customer_chosen = problem.get_customer_index(act)
                else:
                    # for now if there are no customers choose noop if it is valid, otherwise depot
                    if problem.allow_noop:
                        act = problem.NOOP_INDEX
                        customer_chosen = problem.NOOP_INDEX
                    else:
                        act = problem.DEPOT_INDEX
                        customer_chosen = problem.DEPOT_INDEX
                obs, reward, completed, _ = problem.step(act)
                all_actions.append(act)
                if save_routes and j == 0:
                    vehicle_pos = problem.current_state.current_vehicle_position
                    vehicle_route[route_num]["x"].append(vehicle_pos[0])
                    vehicle_route[route_num]["y"].append(vehicle_pos[1])
                    if customer_chosen == -2:
                        print(
                            f"total demand in route-{route_num} is:{vehicle_route[route_num]['total_demand']}"
                        )
                        # vehicle returning to depot therefore a new route is created
                        route_num += 1
                        vehicle_route[route_num] = {
                            "x": [vehicle_pos[0]],
                            "y": [vehicle_pos[1]],
                            "total_demand": 0,
                        }
                    else:
                        current_customer_demand = problem.current_state.customer_demands[customer_chosen]
                        vehicle_route[route_num]["total_demand"] += current_customer_demand
                total_reward += reward
            total_rewards.append(total_reward)
        except Exception:
            print(traceback.format_exc())
            total_rewards.append(np.nan)
    return np.mean(total_rewards), vehicle_route, all_actions


def main():
    warnings.filterwarnings("ignore")
    POLICIES = ["simple", "ppo"]
    PROBLEMS = ["dynamic_uniform", "static_fixed", "dynamic_mixture_gaussian"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_path", type=str,
                        default="cvrp_experimentation/saved_problems/dynamic/uniform_20/"
                                "dynamic_uniform_20-customers.json")
    parser.add_argument("--policies", type=str, default=["simple"], nargs="+", choices=POLICIES,
                        help="Policies to be tested")
    parser.add_argument("--problem", type=str, default="dynamic_uniform", choices=PROBLEMS, )
    parser.add_argument("--ppo_model_folder", type=str, help="folder where ppo model is saved",
                        default="cvrp_experimentation/saved_problems/static/uniform_5/")
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--output_file", type=str,
                        default="cvrp_experimentation/saved_problems/dynamic/uniform_20/results/results_20.json")
    parser.add_argument("--use_trains", action="store_true", help="if code should run logging with trains")
    parser.add_argument("--trains_task_name", type=str, default="static gaussian", help="name for task in trains")
    parser.add_argument("--save_routes", action="store_true", help="Save full route for each seed and each method")
    parser.add_argument("--timeout", type=int, default=10, help="timeout for search algorithms")
    parser.add_argument("--update_results", action="store_true", help="update the existing json files with new results")
    args = parser.parse_args()
    if args.use_trains:
        task = Task.init(project_name="cvrp_dynamic_baselines", task_name=args.trains_task_name)
        logger = Task.current_task().get_logger()
    else:
        logger = None
    save_routes = args.save_routes
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(args.problem_path, "r") as f:
        problem_params = json.load(f)
        if args.use_trains:
            parameters = task.connect(problem_params)
    envs = None
    if args.problem == "dynamic_uniform":
        envs = {args.start_seed + seed: create_uniform_dynamic_problem(**problem_params,
                                                                       random_seed=args.start_seed + seed)
                for seed in range(args.num_seeds)}
    elif args.problem == "dynamic_mixture_gaussian":
        envs = {args.start_seed + seed: create_mixture_guassian_dynamic_problem(**problem_params,
                                                                                random_seed=args.start_seed + seed)
                for seed in range(args.num_seeds)}
    elif args.problem == "static_fixed":
        envs = {args.start_seed + seed: create_fixed_static_problem(**problem_params)
                for seed in range(args.num_seeds)}

    print("running evaluation code")
    sweep_policy = SweepPolicy()
    or_tools = ORToolsPolicy(timeout=args.timeout)

    values = {}
    policies = args.policies
    if "simple" in policies:
        values["Uniformly Random"] = evaluate_policy_simple(envs, random_policy, samples_per_seed=5, logger=logger,
                                                            save_routes=save_routes)
        values["Inversely Proportional to Distance"] = evaluate_policy_simple(envs, distance_proportional_policy,
                                                                              samples_per_seed=5, logger=logger,
                                                                              save_routes=save_routes)
        values["Sweep"] = evaluate_policy_simple(envs, sweep_policy, samples_per_seed=5, logger=logger,
                                                 save_routes=save_routes)
        values["OR-Tools"] = evaluate_policy_simple(envs, or_tools, samples_per_seed=5, logger=logger,
                                                    save_routes=save_routes)

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
        values[policy_name] = evaluate_policy_simple(env_list, policy, samples_per_seed=5, logger=logger,
                                                     save_routes=save_routes, policy_name=policy_name)

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
