from gym import Env, spaces
from numpy.random.mtrand import RandomState

from cvrp_simulation.simulator import CVRPSimulation
import numpy as np


class CVRPGymWrapper(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, simulator: CVRPSimulation) -> None:
        """
        Wrap a CVRP cvrp_simulation with a gym compatible interface
        :param simulator: the actual simulator
        """
        super().__init__()
        self.simulator = simulator
        self.max_customers = simulator.initial_state.customer_visited.size
        # actions are: go to one of the customers, or return to the depot
        self.action_space = spaces.Discrete(self.max_customers + 1)
        # observations are:
        # - customer_positions: np.array  # [N, 2] float
        # - customer_demands: np.array  # [N] int
        # - action_mask: np.array  # [N+1] depot is the last index
        # - depot_position: np.array  # x,y
        # - current_vehicle_position: np.array  # x,y
        # - current_vehicle_capacity: int
        obs_spaces = {
            "customer_positions": spaces.Box(
                low=0, high=1,
                shape=(self.max_customers, 2), dtype=np.float32),
            "customer_demands": spaces.Box(
                low=1, high=10,
                shape=(self.max_customers,), dtype=np.int32),
            "action_mask": spaces.MultiBinary(self.max_customers + 1),
            "depot_position": spaces.Box(
                low=0, high=1,
                shape=(2,), dtype=np.float32),
            "current_vehicle_position": spaces.Box(
                low=0, high=1,
                shape=(2,), dtype=np.float32),
            "current_vehicle_capacity": spaces.Discrete(
                n=self.max_customers * 10
            )
        }
        self.observation_space = spaces.Dict(obs_spaces)

    def step(self, action_chosen, machine_index=None) -> (dict, float, bool, object):
        # get the customer chosen based on the action chosen
        customer_index = self.get_customer_index(action_chosen)
        reward, done = self.simulator.step(customer_index)
        return self.current_state_to_observation(), reward, done, {}

    def seed(self, seed=None) -> None:
        self.simulator.set_random_seed(seed)

    def reset(self) -> dict:
        self.simulator.reset()
        return self.current_state_to_observation()

    def render(self, mode="human", close=False) -> None:
        super(CVRPGymWrapper, self).render(mode=mode)
        # TODO : add scatter plot of CVRP problem and create render object

    def current_state_to_observation(self) -> dict:
        available_customers_ids = self.simulator.get_available_customers()
        num_available_customers = available_customers_ids.size
        customer_positions = np.zeros(self.observation_space.spaces["customer_positions"].shape, dtype=np.float32)
        customer_positions[:num_available_customers, :] = \
            self.simulator.current_state.customer_positions[available_customers_ids]
        customer_demands = np.zeros(self.observation_space.spaces["customer_demands"].shape, dtype=np.int8)
        customer_demands[:num_available_customers] = \
            self.simulator.current_state.customer_demands[available_customers_ids]
        action_mask = np.zeros(self.observation_space.spaces["action_mask"].shape, dtype=np.float32)
        # make available actions as the number of available customers + depot
        action_mask[:num_available_customers+1] = 1
        depot_position = np.copy(self.simulator.current_state.depot_position)
        current_vehicle_position = np.copy(self.simulator.current_state.current_vehicle_position)
        current_vehicle_capacity = self.simulator.current_state.current_vehicle_capacity
        return {
            "customer_positions": customer_positions,
            "customer_demands": customer_demands,
            "action_mask": action_mask,
            "depot_position": depot_position,
            "current_vehicle_position": current_vehicle_position,
            "current_vehicle_capacity": current_vehicle_capacity,
        }

    def get_customer_index(self, action_index: int) -> int:
        """
        this function gets the customer index based on chosen index and masked customers
        :param action_index: this is the index chosen by the policy [0, 1,... n_available_customers +1]
        :return: customer index (same as customer id)
        """
        available_customers_ids = self.simulator.get_available_customers()
        num_possible_actions = available_customers_ids.size + 1
        if action_index > num_possible_actions:
            raise ValueError(f"action chosen is: {action_index} and there are only :{num_possible_actions} actions")
        if action_index == num_possible_actions:
            customer_index = None  # depot is chosen
        else:
            customer_index = available_customers_ids[action_index]  # find customer from id (index in real customer matrices)
        return customer_index
