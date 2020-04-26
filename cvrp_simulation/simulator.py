from copy import deepcopy
from dataclasses import dataclass
from gym import Env, spaces

import numpy as np

@dataclass
class State:
    """
    this class defines the simulation state at every time step. this state describes the total environment (including
    customers that are not known to the agent)
    """
    depot_position: np.ndarray  # x,y
    current_vehicle_position: np.ndarray  # x,y
    current_vehicle_capacity: int
    vehicle_velocity: int  # this is needed in order to calculate time
    customer_positions: np.ndarray  # [N, 2] float
    customer_demands: np.ndarray  # [N] int
    customer_times: np.ndarray  # [N] int (in the future might be a [N, 2] array if events will also have a end time
    customer_ids: np.ndarray  # [N]  this is used in order to go from action chosen to customer chosen
    customer_visited: np.ndarray  # [N] bool this vector is used to know if customer has been visited or not
    # (True=vehicle visited the customer)


class CVRPSimulation(Env):
    EPSILON_TIME = 1e-6
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_customers: int, problem_generator) -> None:
        """
        Create a new cvrp_simulation. Note that you need to call reset() before starting cvrp_simulation.
        :param max_customers: maximum number of customers in problem (graph size) [int]
        :param problem_generator: a generator of type ScenarioGenerator which generates one instance of the cvrp problem
        and returns the initial state of the problem

        """
        super().__init__()
        self.initial_state = None  # will be defined in reset() method
        self.current_state = None  # current state of the simulation, this is updated at every step() call
        self.problem_generator = problem_generator  # during reset this will generate a new instance of state
        self.current_time = 0  # a ticker which updates at the end of every step() to the next time step
        self.max_customers = max_customers  # max number of customers in the problem (this is the max size of the graph)
        # create objects for gym environment
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
        # TODO understand if these are needed and if so where they should be used and implemented
        self.jobs_completed_since_last_step = []
        self.current_state_value = 0.0

    def reset(self) -> dict:
        self.initial_state = self.problem_generator.reset()
        self.current_state = deepcopy(self.initial_state)
        self.current_time = 0
        return self.current_state_to_observation()

    def seed(self, seed=None) -> None:
        """
        define seed in problem generator
        :param seed: seed to be used [int]
        :return:
        """
        self.problem_generator.seed(seed)

    def step(self, action_chosen: int) -> (float, bool):
        """
        this function updates the state and observation based on the action chosen by agent.
        the action is first translated to customer index and then state is updated in place
        :param action_chosen: index of action chosen from all options
        :return: the function returns the new observation [dict], reward [float] and if the simulation is done [bool]
        """
        # get the customer chosen based on the action chosen
        customer_index = self.get_customer_index(action_chosen)
        # todo: implement dynamic arrivals
        if customer_index is None:
            # returning to depot
            depot_position = self.current_state.depot_position
            traveled_distance = np.linalg.norm(depot_position - self.current_state.current_vehicle_position)
            self.current_state.current_vehicle_position = depot_position
            self.current_state.current_vehicle_capacity = self.initial_state.current_vehicle_capacity
        else:
            # going to a customer
            if self.current_state.customer_visited[customer_index]:
                raise ValueError("cannot revisit the same customer more than once")
            # updating vehicle position and current capacity
            customer_position = self.current_state.customer_positions[customer_index, :]
            customer_demand = self.current_state.customer_demands[customer_index]
            self.current_state.customer_visited[customer_index] = True
            if customer_demand > self.current_state.current_vehicle_capacity:
                raise ValueError(
                    f"going to customer {customer_index} with demand {customer_demand}"
                    f"exceeds the remaining vehicle capacity ({self.current_state.current_vehicle_capacity})")
            traveled_distance = np.linalg.norm(customer_position - self.current_state.current_vehicle_position)
            self.current_state.current_vehicle_position = customer_position
            self.current_state.current_vehicle_capacity -= customer_demand
        # find if the cvrp_simulation is over
        is_done = self.calculate_is_complete()
        # current cvrp_simulation time is the travel time * vehicle velocity
        self.current_time += traveled_distance * self.current_state.vehicle_velocity
        # in the future might want to make a more sophisticated reward for the dynamic problem
        reward = -traveled_distance
        return self.current_state_to_observation(), reward, is_done, {}

    def render(self, mode="human", close=False) -> None:
        """
        this function is needed for gym environment. for now doesn't do anything. in the future should create a scatter
        plot of the current state (vehicle and customer locations)
        :param mode:
        :param close:
        :return:
        """
        super(CVRPSimulation, self).render(mode=mode)
        # TODO : add scatter plot of CVRP problem and create render object

    def current_state_to_observation(self) -> dict:
        """
        this function creates the observation for the agent based on the current state.
        the agent knows only the opened customers (i.e. customers who's start time >= current time)
        :return: current observation [dict]
        """
        # TODO: figure out if the observation needs to always be the same length. if not no need for zero padding
        available_customers_ids = self.get_available_customers()
        num_available_customers = available_customers_ids.size
        # creating current vectors of vehicles position and capacity
        customer_positions = np.zeros(self.observation_space.spaces["customer_positions"].shape, dtype=np.float32)
        customer_positions[:num_available_customers, :] = \
            self.current_state.customer_positions[available_customers_ids]
        customer_demands = np.zeros(self.observation_space.spaces["customer_demands"].shape, dtype=np.int8)
        customer_demands[:num_available_customers] = \
            self.current_state.customer_demands[available_customers_ids]
        action_mask = np.zeros(self.observation_space.spaces["action_mask"].shape, dtype=np.float32)
        # make available actions as the number of available customers + depot
        action_mask[:num_available_customers+1] = 1
        depot_position = np.copy(self.current_state.depot_position)
        current_vehicle_position = np.copy(self.current_state.current_vehicle_position)
        current_vehicle_capacity = self.current_state.current_vehicle_capacity
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
        available_customers_ids = self.get_available_customers()
        num_possible_actions = available_customers_ids.size + 1
        if action_index > num_possible_actions:
            raise ValueError(f"action chosen is: {action_index} and there are only :{num_possible_actions} actions")
        if action_index == num_possible_actions:
            customer_index = None  # depot is chosen
        else:
            customer_index = available_customers_ids[action_index]  # find customer from id (index in real customer matrices)
        return customer_index

    def get_available_customers(self) -> np.ndarray:
        """
        this function returns the ids of the available customers
        :return: np.ndarray of available customer id's, length of array is the number of available customers
        """
        # returns only customers that are:
        # 1. opened (start_time <= sim_current_time)
        # 2. not visited
        # 3. demand is smaller or equal to the vehicle capacity
        demand_time = np.logical_and(self.current_state.customer_times <= self.current_time,
                                     self.current_state.customer_demands <= self.current_state.current_vehicle_capacity)
        demand_time_visited = np.logical_and(demand_time,
                                             np.logical_not(self.current_state.customer_visited))
        # TODO: figure out what happens when there are no available customers
        available_customer_ids = self.current_state.customer_ids[demand_time_visited]
        return available_customer_ids

    def calculate_is_complete(self) -> bool:
        """
        this function returns True if all customers have been visited and False otherwise
        :return: bool
        """
        return self.current_state.customer_visited.all()
