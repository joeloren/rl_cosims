import numpy as np
import pytest

from cvrp_simulation.gym_wrapper import CVRPGymWrapper
from cvrp_simulation.simulator import CVRPSimulation


def main():
    # check cvrp_simulation and state -
    customer_positions = np.array([[1, 0], [1, 1], [0, 1]])
    sim = CVRPSimulation(
        depot_position=np.array([0, 0]),
        initial_vehicle_position=np.array([0, 0]),
        initial_vehicle_capacity=30,
        vehicle_velocity=10,
        customer_positions=customer_positions,
        customer_demands=np.array([5, 5, 5]),
        customer_times=np.array([0, 0, 0]),
        customer_ids=np.arange(0, 3),
        customer_visited=np.zeros([3]).astype(np.bool)
    )
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


if __name__ == "__main__":
    main()
    print("done")
