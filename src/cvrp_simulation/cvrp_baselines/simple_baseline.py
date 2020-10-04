import numpy as np
from cvrp_simulation.plot_results import plot_vehicle_routes
from cvrp_simulation.simulation.scenario_generator import (
    FixedSample,
    SampleStaticBenchmark,
)
from cvrp_simulation.simulation.simulator import CVRPSimulation
from matplotlib import pyplot as plt
from scipy import stats


def random_policy(obs, env):
    probs = obs["action_mask"]
    return probs / np.sum(probs)


def action_selector(obs, env, policy):
    action_probs = policy(obs, env)
    act = np.random.choice(len(obs["action_mask"]), p=action_probs)
    return act


def distance_proportional_policy(obs, env):
    """
    this function creates a greedy policy for the vehicle.
    the probability is the inverse distance between the available customers and vehicle
    notes:
    1. if there are no available customers the depot receives prob=1 and all other opened
    customers are prob=0
    2. if there is only one available customer it receives prob=1 and all other receive prob=0
    """
    prob_out = np.zeros_like(obs["action_mask"]).astype(np.float)
    vehicle_position = obs["current_vehicle_position"]
    customer_positions = obs["customer_positions"]
    if np.sum(obs["action_mask"][:-2]) == 0:
        # in this case there are no customers available so returning to depot
        prob_out[-2] = 1
    else:
        distance_matrix = np.linalg.norm(
            vehicle_position - customer_positions, axis=1
        ).reshape(-1)
        distance_matrix[np.logical_not(obs["action_mask"][:-2])] = 0
        if distance_matrix[distance_matrix > 0].size > 1:
            # this is needed so that the customers closest to the vehicle receive the highest
            # probability
            inverse_distance_matrix = np.max(distance_matrix) - distance_matrix
        else:
            # if there is only one customer available there is no need to inverse its distance
            inverse_distance_matrix = distance_matrix
        inverse_distance_matrix[np.logical_not(obs["action_mask"][:-2])] = 0
        prob_out[:-2] = inverse_distance_matrix / np.sum(inverse_distance_matrix)
    return prob_out


def create_benchmark_generator():
    CAPACITIES = {10: 20.0, 20: 30.0, 50: 40.0, 100: 50.0}
    # create random input based on benchmark distributions -
    depot_position_rv = stats.uniform(loc=0, scale=1)
    vehicle_position_rv = stats.uniform(loc=0, scale=1)
    customer_positions_rv = stats.uniform(loc=0, scale=1)
    customer_demands_rv = stats.randint(low=0, high=10)
    vrp_size = 20
    initial_vehicle_capacity = CAPACITIES[vrp_size]
    problem_generator = SampleStaticBenchmark(
        depot_position_rv=depot_position_rv,
        vehicle_position_rv=vehicle_position_rv,
        vehicle_capacity=initial_vehicle_capacity,
        vehicle_velocity=10,
        customer_positions_rv=customer_positions_rv,
        customer_demands_rv=customer_demands_rv,
        vrp_size=vrp_size,
    )
    return problem_generator


def create_fixed_generator():
    # run fixed problem for debugging
    customer_positions = np.array([[1, 0], [1, 1], [0, 1]])
    depot_position = np.array([0, 0])
    initial_vehicle_position = np.array([0, 0])
    initial_vehicle_capacity = 10
    vehicle_velocity = 10
    customer_demands = np.array([5, 5, 5])
    customer_times = np.array([0, 0, 0])
    vrp_size = 3
    problem_generator = FixedSample(
        depot_position,
        initial_vehicle_position,
        initial_vehicle_capacity,
        vehicle_velocity,
        customer_positions,
        customer_demands,
        customer_times,
    )
    return problem_generator


def main():
    fig1, ax1 = plt.subplots(1, 1)
    problem_generator = None
    run_benchmark = True
    plot_results = True
    print_debug = False
    seed = 123
    num_runs = 1
    if run_benchmark:
        problem_generator = create_benchmark_generator()
    else:
        problem_generator = create_fixed_generator()
    vrp_size = problem_generator.vrp_size
    sim = CVRPSimulation(max_customers=vrp_size, problem_generator=problem_generator)
    sim.seed(seed)
    all_random_rewards = np.zeros(num_runs)
    for i_r in range(num_runs):
        obs = sim.reset()
        done = False
        total_reward = 0
        i = 0
        vehicle_route = {
            0: {
                "x": [sim.current_state.current_vehicle_position[0]],
                "y": [sim.current_state.current_vehicle_position[1]],
            }
        }
        depot_position = sim.current_state.depot_position
        customer_positions = sim.current_state.customer_positions
        customer_demands = sim.current_state.customer_demands
        route_num = 0
        while not done:
            action_probs = distance_proportional_policy(obs, sim)
            act = np.argmax(action_probs).astype(np.int)
            customer_chosen = sim.get_customer_index(act)
            obs, reward, done, info = sim.step(act)
            vehicle_pos = sim.current_state.current_vehicle_position
            vehicle_route[route_num]["x"].append(vehicle_pos[0])
            vehicle_route[route_num]["y"].append(vehicle_pos[1])
            if act == action_probs.size - 2:
                # vehicle returning to depot therefore a new route is created
                route_num += 1
                vehicle_route[route_num] = {
                    "x": [vehicle_pos[0]],
                    "y": [vehicle_pos[1]],
                }
            if print_debug:
                print(
                    f"i:{i}, t:{np.round(sim.current_time, 2)}, vehicle capacity:"
                    f"{sim.current_state.current_vehicle_capacity},"
                    f" customer chosen:{customer_chosen}, reward {reward}, done {done}"
                )
            total_reward += reward
            i += 1
        print(total_reward)
        if plot_results:
            plot_vehicle_routes(
                depot_position, customer_positions, customer_demands, vehicle_route, ax1
            )
            plt.show()
        all_random_rewards[i_r] = total_reward
    print(
        f"mean reward is:{np.mean(all_random_rewards)}, std reward is:{np.std(all_random_rewards)}"
    )


if __name__ == "__main__":
    main()
    print("done!")
