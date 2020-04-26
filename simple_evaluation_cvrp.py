import numpy as np
from scipy import stats

from cvrp_simulation.simulator import CVRPSimulation
from cvrp_simulation.scenario_generator import FixedSample, SampleStaticBenchmark, SampleDynamicBenchmark


def main():
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

    print("--------------------------------------")
    print("simulation testing:")
    obs = sim.reset()
    # in this case each time we choose action 0 since the available actions change each time
    # the number of available customers changes
    obs, reward, done, _ = sim.step(0)
    print(f"reward {reward}, done {done}")
    available_customers = sim.get_available_customers()
    print(f"available customers:{available_customers}")
    obs, reward, done, _ = sim.step(0)
    print(f"reward {reward}, done {done}")
    available_customers = sim.get_available_customers()
    print(f"available customers:{available_customers}")
    obs, reward, done, _ = sim.step(0)
    print(f"reward {reward}, done {done}")
    available_customers = sim.get_available_customers()
    print(f"available customers:{available_customers}")

    # check benchmark generator
    depot_position_rv = stats.uniform(loc=0, scale=1)
    vehicle_position_rv = stats.uniform(loc=0, scale=1)
    customer_positions_rv = stats.uniform(loc=0, scale=1)
    customer_demands_rv = stats.uniform(loc=0, scale=10)
    vrp_size = 20
    initial_vehicle_capacity = 10*vrp_size
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
    print("--------------------------------------")
    print("benchmark simulation testing:")
    sim = CVRPSimulation(max_customers=vrp_size, problem_generator=benchmark_generator)
    sim.seed(seed)
    for i in range(num_runs):
        obs = sim.reset()
        # in this case each time we choose action 0 since the available actions change each time
        # the number of available customers changes
        tot_reward = 0
        done = False
        while not done:
            obs, reward, done, _ = sim.step(0)
            tot_reward += reward
        print(f"finished random run # {i}, total reward {tot_reward}")
        rand_reward[i] = tot_reward
    print(f"mean random reward is:{np.mean(rand_reward)}")

    # check dynamic benchmark generator
    depot_position_rv = stats.uniform(loc=0, scale=1)
    vehicle_position_rv = stats.uniform(loc=0, scale=1)
    customer_positions_rv = stats.uniform(loc=0, scale=1)
    customer_demands_rv = stats.randint(low=0, high=10)
    customer_times_rv = stats.randint(low=0, high=5)
    vrp_size = 20
    initial_vehicle_capacity = 10 * vrp_size
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
    print("--------------------------------------")
    print("benchmark simulation testing:")
    sim = CVRPSimulation(max_customers=vrp_size, problem_generator=dynamic_generator)
    sim.seed(seed)
    for i in range(num_runs):
        obs = sim.reset()
        # in this case each time we choose action 0 since the available actions change each time
        # the number of available customers changes
        tot_reward = 0
        done = False
        while not done:
            obs, reward, done, _ = sim.step(0)
            tot_reward += reward
        print(f"finished random run # {i}, total reward {tot_reward}")
        rand_reward[i] = tot_reward
    print(f"mean random reward is:{np.mean(rand_reward)}")


if __name__ == "__main__":
    main()
    print("done")
