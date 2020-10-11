from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import numpy as np
from gym import Env, spaces


@dataclass
class State:
    """
    this class defines the simulation state at every time step. this state describes the total
    environment (including
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
    customer_visited: np.ndarray  # [N] bool this vector is used to know if customer has been  visited or not
    # (True=vehicle visited the customer)


class CVRPSimulation(Env):
    EPSILON_TIME = 1e-6
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_customers: int, problem_generator, allow_noop=True) -> None:
        """
        Create a new simulation. Note that you need to call reset() before starting simulation.
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
        self.allow_noop = allow_noop  # if noop action is allowed in simulation
        self.NOOP_INDEX = -1
        self.DEPOT_INDEX = -2
        # create objects for gym environment
        self.action_space = spaces.Discrete(self.max_customers + 2)
        # observations are:
        # - customer_positions: np.array  # [N, 2] float
        # - customer_demands: np.array  # [N] int
        # - customer_ids: np.array  # [N] int this is for being able to go from action to
        # customer index easily
        # - action_mask: np.array  # [N+2] depot is N+1 index, noop is N+2 index
        # action_mask is True/False depending if action is possible (True) or not (False)
        # - depot_position: np.array  # x,y
        # - current_vehicle_position: np.array  # x,y
        # - current_vehicle_capacity: int
        obs_spaces = {
            "customer_positions": spaces.Box(low=0, high=1, shape=(self.max_customers, 2), dtype=np.float32),
            "customer_demands": spaces.Box(low=1, high=10, shape=(self.max_customers,), dtype=np.int32),
            "customer_ids": spaces.Box(low=0, high=max_customers, shape=(max_customers,), dtype=np.int),
            "action_mask": spaces.MultiBinary(self.max_customers + 2),
            "depot_position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "current_vehicle_position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "current_vehicle_capacity": spaces.Discrete(n=self.max_customers * 20),
            "max_vehicle_capacity": spaces.Discrete(n=self.max_customers * 20),
        }
        self.observation_space = spaces.Dict(obs_spaces)

    def reset(self) -> dict:
        self.initial_state = self.problem_generator.reset()
        self.current_state = deepcopy(self.initial_state)
        self.current_time = 0
        obs = self.current_state_to_observation()
        return obs

    def seed(self, seed=None) -> None:
        """
        define seed in problem generator
        :param seed: seed to be used [int]
        :return:
        """
        self.problem_generator.seed(seed)

    def step(self, action_chosen: int) -> (dict, float, bool, dict):
        """
        this function updates the state and observation based on the action chosen by agent.
        the action is first translated to customer index and then state is updated in place
        :param action_chosen: index of action chosen from all options
        :return: the function returns the new observation [dict], reward [float] and if the
        simulation is done [bool]
        """
        # get the customer chosen based on the action chosen
        customer_index = self.get_customer_index(action_chosen)
        # noop is chosen
        if customer_index == self.NOOP_INDEX:
            # the vehicle does not move and the time is moved to the next time a customer is opened
            traveled_distance = 0
            next_time = self.get_next_time()  # based on next customer to be available
            if next_time < self.current_time:
                raise ValueError(f"new time:{next_time} must be larger than the current time:{self.current_time}")
            self.current_time = next_time
        # depot is chosen
        elif customer_index == self.DEPOT_INDEX:
            # the vehicle is moved to the depot and the capacity is updated to initial capacity
            depot_position = self.current_state.depot_position
            traveled_distance = np.linalg.norm(depot_position - self.current_state.current_vehicle_position)
            self.current_state.current_vehicle_position = deepcopy(depot_position)
            self.current_state.current_vehicle_capacity = self.initial_state.current_vehicle_capacity
        # customer is chosen
        else:
            # going to a customer
            if self.current_state.customer_visited[customer_index]:
                raise ValueError(f"cannot revisit the same customer (id:{customer_index}) more than once")
            # getting customer position and demand
            customer_position = self.current_state.customer_positions[customer_index, :]
            customer_demand = self.current_state.customer_demands[customer_index]
            # checking if customer demand exceeds vehicle current capacity
            if customer_demand > self.current_state.current_vehicle_capacity:
                raise ValueError(
                    f"going to customer {customer_index} with demand {customer_demand}"
                    f"exceeds the remaining vehicle capacity ("
                    f"{self.current_state.current_vehicle_capacity})"
                )
            traveled_distance = np.linalg.norm(customer_position - self.current_state.current_vehicle_position)
            # updating current state (vehicle capacity, position and customer visited state)
            self.current_state.customer_visited[customer_index] = True
            self.current_state.current_vehicle_position = customer_position
            self.current_state.current_vehicle_capacity -= customer_demand

        # find if the simulation is over
        is_done = self.calculate_is_complete()
        # if simulation is done, add returning to depot to reward
        if is_done:
            # to do if in the future we want to update the reward to include time, this also needs
            #  to be updated
            traveled_distance += np.linalg.norm(self.current_state.current_vehicle_position -
                                                self.current_state.depot_position)
        # next time is found based on: delta_time = travel_distance * vehicle_velocity  (t = v*x)
        self.current_time += traveled_distance * self.current_state.vehicle_velocity
        # in the future might want to make a more sophisticated reward for the dynamic problem
        reward = -traveled_distance
        return self.current_state_to_observation(), reward, is_done, {}

    def render(self, mode="human", close=False) -> None:
        """
        this function is needed for gym environment. for now doesn't do anything. in the future should create a scatter
        plot of the current state (vehicle and customer locations)
        :param mode:
        :param close:not used for now
        :return:
        """
        super(CVRPSimulation, self).render(mode=mode)
        # to do : add scatter plot of CVRP problem and create render object

    def current_state_to_observation(self) -> dict:
        """
        this function creates the observation for the agent based on the current state.
        the agent knows only the opened customers (i.e. customers who's start time >= current time)
        :return: current observation [dict]
        """
        opened_customers_ids = self.get_opened_customers()
        action_mask = self.get_masked_options(opened_customers_ids)  # return vector of customers , depot and noop masks
        # creating current vectors of vehicles position and capacity
        customer_positions = deepcopy(self.current_state.customer_positions[opened_customers_ids])
        customer_demands = deepcopy(self.current_state.customer_demands[opened_customers_ids])
        customer_ids = deepcopy(opened_customers_ids)
        depot_position = np.copy(self.current_state.depot_position)
        current_vehicle_position = np.copy(self.current_state.current_vehicle_position)
        current_vehicle_capacity = self.current_state.current_vehicle_capacity
        max_vehicle_capacity = self.initial_state.current_vehicle_capacity
        return {
            "customer_positions": customer_positions,
            "customer_demands": customer_demands,
            "customer_ids": customer_ids,
            "action_mask": action_mask,
            "depot_position": depot_position,
            "current_vehicle_position": current_vehicle_position,
            "current_vehicle_capacity": current_vehicle_capacity,
            "max_vehicle_capacity": max_vehicle_capacity,
        }

    @staticmethod
    def observation(obs: Dict) -> Dict:
        # this function returns the observation since the simulator does not change the observation (in future work
        # might change the observation and then this is needed)
        return obs

    def get_customer_index(self, action_index: int) -> int:
        """
        this function gets the customer index based on chosen index and masked customers
        the actions order is: [c_0, c_1,... c_n, depot, noop] where c_i is the customer i that is opened
        noop is an option only if allow_noop is True otherwise vector is [n_available_customers + 1] length
        :param action_index: this is the index chosen by the policy [0, 1, ... n_available_customers +2]
        :return: customer index (same as customer id) or -1 for noop or -2 for depot
        """
        opened_customers_ids = self.get_opened_customers()
        if opened_customers_ids.size > 0:  # there are opened customers waiting for pickup
            num_possible_actions = (opened_customers_ids.size + 2)  # all opened customers + depot + noop
            if action_index > num_possible_actions:
                raise ValueError(f"action chosen is: {action_index} and there are only :{num_possible_actions} actions")
            #  noop chosen
            if action_index == num_possible_actions - 1:
                if not self.allow_noop:
                    raise Exception("noop action chosen even though flag is False")
                customer_index = self.NOOP_INDEX  # noop is chosen
            # depot chosen
            elif action_index == num_possible_actions - 2:
                customer_index = self.DEPOT_INDEX  # depot is chosen
            # customer chosen
            else:
                # find customer index based on masked customers
                masked_options = self.get_masked_options(opened_customers_ids)
                if masked_options[action_index]:
                    customer_index = opened_customers_ids[
                        action_index
                    ]  # find customer from id matrix
                else:
                    # the customer chosen is not available (action should have been masked)
                    raise Exception(f"tried to move to customer that is not available ,  "
                                    f"customer id:{opened_customers_ids[action_index]}")
        else:  # there are no customers opened in the system that have not been picked up
            # to do decide what happens when there are no customers available. for now chooses noop
            if self.allow_noop:
                customer_index = self.NOOP_INDEX  # noop is chosen
            else:
                customer_index = self.DEPOT_INDEX  # depot is chosen
        return customer_index

    def reset_future(
        self, seed: int, horizon_limit: int = None, n_future_customers: int = 3
    ) -> Env:
        """
        this function copies the current environment and in the new environment takes all customers that are not yet
        opened and updates their time, position and demand. this is used for heuristics that preform rollouts and
        need to update the future environment (MCTS for example)
        :param seed: the seed to use for restarting the generators
        :param horizon_limit: this is the limit on how much of the horizon to create,
        if None creates the full horizon
        :param n_future_customers: if not None this is the maximum number of future customers to create
        :return: copy_env: updated simulation with new customers
        """
        # print(f"resetting future for simulation , t:{self.current_time}")
        copy_env = deepcopy(self)
        copy_env.seed(seed)
        n_opened_customers = np.sum(copy_env.current_state.customer_times <= copy_env.current_time)
        n_customers_left = self.max_customers - n_opened_customers
        # print(f"found {len(n_customers_left)} customers that need to be updated")
        if horizon_limit is None and n_future_customers is not None:
            n_customers_to_create = np.min([n_future_customers, n_customers_left])
            new_customer_properties = self.problem_generator.reset_customers(copy_env.current_time,
                                                                             n_customers_to_create)
            future_customer_times = np.sort(new_customer_properties["time"])
        elif horizon_limit is not None and n_future_customers is None:
            new_customer_properties = self.problem_generator.reset_customers(copy_env.current_time, n_customers_left)
            future_customer_times = np.sort(new_customer_properties["time"])
            # in this case create only customers that start time is smaller or equal to the horizon
            n_customers_to_create = np.sum(future_customer_times <= horizon_limit + copy_env.current_time)
        elif horizon_limit is None and n_future_customers is None:
            # in this case create all future customers since the horizon and number of customers
            # in unlimited
            n_customers_to_create = n_customers_left
            new_customer_properties = self.problem_generator.reset_customers(copy_env.current_time, n_customers_left)
            future_customer_times = np.sort(new_customer_properties["time"])
        else:
            raise ValueError("in future_reset horizon_limit and n_future customers are both not equal to None, "
                             "simulation does not know what how many future customers to create")
        future_customer_positions = new_customer_properties["position"]
        future_customer_demands = new_customer_properties["demand"]

        n_total_customers = np.min([n_opened_customers + n_customers_to_create, self.max_customers])
        # initialize all customer parameters for state
        customer_positions = np.zeros(shape=[n_total_customers, 2])
        customer_times = np.zeros(shape=n_total_customers)
        customer_demands = np.zeros_like(customer_times)
        customer_ids = np.arange(n_total_customers).astype(int)
        customer_visited = np.zeros_like(customer_times).astype(np.bool)
        # add opened customers to state
        customer_positions[:n_opened_customers, ...] = np.copy(
            copy_env.current_state.customer_positions[:n_opened_customers, ...]
        )
        customer_times[:n_opened_customers] = np.copy(
            copy_env.current_state.customer_times[:n_opened_customers]
        )
        customer_demands[:n_opened_customers] = np.copy(
            copy_env.current_state.customer_demands[:n_opened_customers]
        )
        customer_visited[:n_opened_customers] = np.copy(
            copy_env.current_state.customer_visited[:n_opened_customers]
        )
        # add new customers to state
        customer_positions[n_opened_customers:, ...] = np.copy(
            future_customer_positions[:n_customers_to_create]
        )
        customer_times[n_opened_customers:, ...] = np.copy(
            future_customer_times[:n_customers_to_create]
        )
        customer_demands[n_opened_customers:, ...] = np.copy(
            future_customer_demands[:n_customers_to_create]
        )
        # create new initial state based on new information for future customers -
        copy_env.initial_state = State(
            depot_position=np.copy(copy_env.initial_state.depot_position),  # [x,y]
            current_vehicle_position=np.copy(
                copy_env.initial_state.current_vehicle_position
            ),
            # [x,y]
            current_vehicle_capacity=int(
                copy_env.initial_state.current_vehicle_capacity
            ),
            vehicle_velocity=copy_env.initial_state.vehicle_velocity,
            customer_positions=customer_positions,  # [x,y] for each customer 0, 1,..N
            customer_demands=customer_demands.astype(int),  # [N]
            customer_times=customer_times,  # [N] this is the start time of each customer
            customer_ids=customer_ids,
            customer_visited=np.zeros(shape=n_total_customers).astype(bool),
        )
        # copy new initial state to current state  -
        copy_env.current_state = State(
            depot_position=np.copy(copy_env.current_state.depot_position),  # [x,y]
            current_vehicle_position=np.copy(
                copy_env.current_state.current_vehicle_position
            ),
            # [x,y]
            current_vehicle_capacity=int(
                copy_env.current_state.current_vehicle_capacity
            ),
            vehicle_velocity=copy_env.current_state.vehicle_velocity,
            customer_positions=customer_positions,  # [x,y] for each customer 0, 1,..N
            customer_demands=customer_demands.astype(int),  # [N]
            customer_times=customer_times,  # [N] this is the start time of each customer
            customer_ids=customer_ids,
            customer_visited=customer_visited,
        )
        return copy_env

    def get_opened_customers(self) -> np.ndarray:
        """
        this function returns the ids of the available customers
        :return: np.ndarray of available customer id's, length of array is the number of
        available customers
        """
        # returns only customers that are:
        # 1. opened (start_time <= sim_current_time)
        # 2. not visited
        time_visited = np.logical_and(
            self.current_state.customer_times <= self.current_time,
            np.logical_not(self.current_state.customer_visited),
        )
        # to do: figure out what happens when there are no available customers
        opened_customer_ids = self.current_state.customer_ids[time_visited]
        return opened_customer_ids

    def get_masked_options(self, opened_customers_ids: np.ndarray) -> np.ndarray:
        """
        this function masks the current action vector where customers exceed the current vehicle
        capacity
        noop is True depending on flag and if there are customers still not opened and depot is
        always True
        :param opened_customers_ids: vector of opened customers that have not been picked up yet
        :return: masked_options - bool vector of masked options (True : option is available ,
        False: not available)
        """
        opened_customer_demands = self.current_state.customer_demands[opened_customers_ids]
        num_opened_customers = opened_customers_ids.size
        masked_options = np.ones(num_opened_customers + 2).astype(np.bool)
        masked_options[:-2] = (opened_customer_demands <= self.current_state.current_vehicle_capacity)
        num_customers_not_opened = np.sum(self.current_state.customer_times > self.current_time)
        if (not self.allow_noop) or (num_customers_not_opened == 0):
            # noop is not allowed and therefore should be masked out
            masked_options[self.NOOP_INDEX] = False
        if np.array_equal(self.current_state.current_vehicle_position, self.current_state.depot_position):
            masked_options[self.DEPOT_INDEX] = False
        return masked_options

    def get_next_time(self) -> int:
        """
        this function finds the next time based on the next customer available (used when noop is chosen)
        :return:
        """
        customer_times = self.current_state.customer_times
        future_customer_times = customer_times[customer_times > self.current_time]
        if future_customer_times.size > 0:
            return np.min(future_customer_times)
        else:
            # to do decide if the simulation should send an exception or something else should
            #  happen in this case
            raise Exception(f"policy chose noop but there are no more customers that need to be opened")

    def calculate_is_complete(self) -> bool:
        """
        this function returns True if all customers have been visited and False otherwise
        :return: bool
        """
        return self.current_state.customer_visited.all()

    def get_simulator(self):
        """
        this function is needed for the fast mcts algorithm
        """
        return self

    @staticmethod
    def get_available_actions(state):
        available_actions = []
        for action_index, action_mask in enumerate(state["action_mask"]):
            if action_mask:
                available_actions.append(action_index)
        return available_actions
