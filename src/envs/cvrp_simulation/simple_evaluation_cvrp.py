import numpy as np
from scipy import stats

from src.envs.cvrp_simulation import CVRPSimulation
from src.envs.cvrp_simulation import FixedSample, SampleStaticBenchmark, SampleDynamicBenchmark


def run_simple_test():
    # check cvrp_simulation and state -
    customer_positions = np.array([[1, 0], [1, 1], [0, 1]])
    depot_position = np.array([0, 0])
    initial_vehicle_position = np.array([0, 0])
    initial_vehicle_capacity = 30
    vehicle_velocity = 10
    customer_demands = np.array([5, 5, 5])
    customer_times = np.array([0, 0, 0])
    problem_generator = FixedSample(depot_position, initial_vehicle_position, initial_vehicle_capacity,
                                    vehicle_velocity, customer_positions, customer_demands, customer_times)
    sim = CVRPSimulation(max_customers=3, problem_generator=problem_generator)
    obs = sim.reset()
    # in this case each time we choose action 0 since the available actions change each time
    # the number of available customers changes
    obs, reward, done, _ = sim.step(0)
    print(f"reward {reward}, done {done}")
    opened_customers = sim.get_opened_customers()
    print(f"available customers:{opened_customers}")
    obs, reward, done, _ = sim.step(0)
    print(f"reward {reward}, done {done}")
    opened_customers = sim.get_opened_customers()
    print(f"available customers:{opened_customers}")
    obs, reward, done, _ = sim.step(0)
    print(f"reward {reward}, done {done}")
    opened_customers = sim.get_opened_customers()
    print(f"available customers:{opened_customers}")


def run_static_benchmark():
    depot_position_rv = stats.uniform(loc=0, scale=1)
    vehicle_position_rv = stats.uniform(loc=0, scale=1)
    customer_positions_rv = stats.uniform(loc=0, scale=1)
    customer_demands_rv = stats.randint(low=0, high=10)
    vrp_size = 20
    initial_vehicle_capacity = 10 * vrp_size
    vehicle_velocity = 10
    benchmark_generator = SampleStaticBenchmark(
        depot_position_rv=depot_position_rv,
        vehicle_position_rv=vehicle_position_rv,
        vehicle_capacity=initial_vehicle_capacity,
        vehicle_velocity=vehicle_velocity,
        customer_positions_rv=customer_positions_rv,
        customer_demands_rv=customer_demands_rv,
        vrp_size=vrp_size)
    num_runs = 10
    seed = 50
    rand_reward = np.zeros(num_runs)
    sim = CVRPSimulation(max_customers=vrp_size, problem_generator=benchmark_generator)
    sim.seed(seed)
    for i in range(num_runs):
        obs = sim.reset()
        # in this case each time we choose action 0 since the available actions change each time
        # the number of available customers changes
        tot_reward = 0
        done = False
        while not done:
            available_actions = obs['action_mask']
            action_chosen = np.random.choice(np.flatnonzero(available_actions), 1)[0]
            obs, reward, done, _ = sim.step(action_chosen)
            tot_reward += reward
        print(f"finished random run # {i}, total reward {tot_reward}")
        rand_reward[i] = tot_reward
    print(f"mean random reward is:{np.mean(rand_reward)}")


def run_dynamic_benchmark():
    # check dynamic benchmark generator
    depot_position_rv = stats.uniform(loc=0, scale=1)
    vehicle_position_rv = stats.uniform(loc=0, scale=1)
    customer_positions_rv = stats.uniform(loc=0, scale=1)
    customer_demands_rv = stats.randint(low=0, high=10)
    customer_times_rv = stats.randint(low=0, high=5)
    vrp_size = 20
    initial_vehicle_capacity = 1 * vrp_size
    vehicle_velocity = 10
    dynamic_generator = SampleDynamicBenchmark(
        depot_position_rv=depot_position_rv,
        vehicle_position_rv=vehicle_position_rv,
        vehicle_capacity=initial_vehicle_capacity,
        vehicle_velocity=vehicle_velocity,
        customer_positions_rv=customer_positions_rv,
        customer_demands_rv=customer_demands_rv,
        customer_times_rv=customer_times_rv,
        vrp_size=vrp_size)
    num_runs = 10
    seed = 50
    rand_reward = np.zeros(num_runs)
    sim = CVRPSimulation(max_customers=vrp_size, problem_generator=dynamic_generator)
    sim.seed(seed)
    for i in range(num_runs):
        obs = sim.reset()
        # in this case each time we choose action 0 since the available actions change each time
        # the number of available customers changes
        tot_reward = 0
        done = False
        while not done:
            available_actions = obs['action_mask']
            action_chosen = np.random.choice(np.flatnonzero(available_actions), 1)[0]
            obs, reward, done, _ = sim.step(action_chosen)
            tot_reward += reward
        print(f"finished random run # {i}, total reward {tot_reward}")
        rand_reward[i] = tot_reward
    print(f"mean random reward is:{np.mean(rand_reward)}")


def main():

    print("----------------------------------------------------------------------------")
    print("simulation testing:")
    run_simple_test()

    print("----------------------------------------------------------------------------")
    print("benchmark simulation testing:")
    run_static_benchmark()

    print("----------------------------------------------------------------------------")
    print("dynamic simulation testing:")
    run_dynamic_benchmark()


if __name__ == "__main__":
    main()
    print("done")
