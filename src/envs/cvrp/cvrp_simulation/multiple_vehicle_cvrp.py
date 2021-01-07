from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Tuple
import simpy
from simpy import Event

import numpy as np
from gym import Env, spaces

# global definitions
CUSTOMER_STATUS = {'idle': 0,
                   'opened': 1,
                   'chosen': 2,
                   'visited': 3}
VEHICLE_STATUS = {'available': 0,
                  'busy': 1}
DEPOT_INDEX = -1
NOOP_INDEX = -2
EPSILON_TIME = 1e-6


@dataclass
class State:
    """
    this class defines the simulation state at every time step. this state describes the total
    environment (including
    customers that are not known to the agent)
    """
    depot_position: np.ndarray  # x,y float
    current_vehicle_positions: np.ndarray  # [M, 2] float
    current_vehicle_capacities: np.ndarray  # [M]
    vehicle_velocity: int  # this is needed in order to calculate time
    # available - waiting to go to customer
    # busy - vehicle on it's way to a customer
    vehicle_status: np.ndarray  # [M] int, status of vehicle
    current_vehicle_customer: np.ndarray  # [M] the current customer index the vehicle is at
    # (we assume the depot is index -1)
    customer_positions: np.ndarray  # [N, 2] float
    customer_demands: np.ndarray  # [N] int
    customer_times: np.ndarray  # [N] int (in the future might be a [N, 2] array if events will also have a end time
    customer_ids: np.ndarray  # [N]  this is used in order to go from action chosen to customer chosen
    # idle - not yet opened
    # opened - waiting to be chosen
    # chosen - vehicle is on its way to pick up customer
    # visited - vehicle reached customer
    customer_status: np.ndarray  # [N] int, status of customer
    previous_time: float  # previous decision point time
    vehicle_full_path: Dict  # dictionary where each key is a different vehicle and the value is a
    # list of chosen customers (used mainly for debugging)
    vehicle_full_distance: np.ndarray  # distance each vehicle travelled


class CVRPSimulation(Env, ABC):
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_customers: int, problem_generator, allow_noop=False) -> None:
        """
        Create a new simulation. Note that you need to call reset() before starting simulation.
        :param max_customers: maximum number of customers in problem (graph size) [int]
        :param problem_generator: a generator of type ScenarioGenerator which generates one instance of the cvrp problem
        and returns the initial state of the problem

        """
        super().__init__()
        self.initial_state: State = State(depot_position=np.array([]), current_vehicle_capacities=np.array([]),
                                          vehicle_velocity=0, vehicle_status=np.array([]),
                                          current_vehicle_positions=np.array([]), customer_demands=np.array([]),
                                          customer_times=np.array([]), customer_ids=np.array([]),
                                          customer_status=np.array([]),
                                          current_vehicle_customer=np.array([]), customer_positions=np.array([]),
                                          previous_time=0, vehicle_full_path={}, vehicle_full_distance=np.array([]))
        self.current_state: State = deepcopy(self.initial_state)  # current state of the simulation, updated in reset
        self.problem_generator = problem_generator  # during reset this will generate a new instance of state
        self.env: simpy.Environment = simpy.Environment()
        # max number of customers in the problem (this is the max size of the graph)
        self.max_customers: int = max_customers
        self.num_vehicles: int = problem_generator.num_vehicles
        self.allow_noop: bool = allow_noop  # if noop action is allowed in simulation
        self.NOOP_INDEX: int = NOOP_INDEX
        self.DEPOT_INDEX: int = DEPOT_INDEX
        self.previous_reward = 0
        self.until = 10e4
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
        # - current_vehicle_customer: int,  the index of the vehicle customer chosen (if vehicle is at the depot this
        # will be the number of customers)
        obs_spaces = {
            "customer_positions": spaces.Box(low=0, high=1, shape=(self.max_customers, 2), dtype=np.float32),
            "customer_demands": spaces.Box(low=1, high=10, shape=(self.max_customers,), dtype=np.int32),
            "customer_ids": spaces.Box(low=0, high=max_customers, shape=(max_customers,), dtype=np.int),
            "illegal_actions": spaces.MultiBinary(self.max_customers + 2),
            "depot_position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "current_vehicle_positions": spaces.Box(low=0, high=1, shape=(self.num_vehicles, 2), dtype=np.float32),
            "current_vehicle_capacities": spaces.Box(low=0, high=self.problem_generator.max_vehicle_capacity,
                                                     shape=(self.num_vehicles,), dtype=np.float32),
            "customer_status": spaces.Box(low=0, high=len(CUSTOMER_STATUS),
                                          shape=(self.max_customers,), dtype=np.int),
            "vehicle_status": spaces.Box(low=0, high=len(VEHICLE_STATUS),
                                         shape=(self.num_vehicles,), dtype=np.int),
            "current_vehicle_customer": spaces.Box(low=0, high=max_customers + 1, shape=(1,), dtype=np.int32),
            "max_vehicle_capacity": spaces.Box(low=0, high=self.problem_generator.max_vehicle_capacity, shape=(1,),
                                               dtype=np.float32),
        }
        self.observation_space = spaces.Dict(obs_spaces)
        self.arrivals_suppressed: bool = False

    def reset(self) -> dict:
        self.env = simpy.Environment()
        self.env.timeout(self.until)
        self.previous_reward = 0
        self.initial_state = self.problem_generator.reset()
        self.initial_state.previous_time = self.env.now
        for i_c in range(self.initial_state.customer_positions.shape[0]):
            if self.initial_state.customer_times[i_c] == 0:
                self.initial_state.customer_status[i_c] = CUSTOMER_STATUS['opened']
            else:
                arrival = self.env.timeout(self.initial_state.customer_times[i_c] - self.env.now, value=i_c)
                arrival.callbacks.append(self.notify_customer_opened)
        self.current_state = deepcopy(self.initial_state)
        self.run_to_decision_point()
        obs = self.current_state_to_observation()
        return obs

    def step(self, action: Tuple):
        self.current_state.previous_time = self.env.now
        vehicle_index = action[0]
        customer_index = action[1]
        num_customers = self.current_state.customer_positions.shape[0]
        if customer_index >= num_customers:
            raise ValueError(f"t:{self.env.now}, customer index out of bounds! \n"
                             f"num customers: {num_customers}, vehicle index {vehicle_index},"
                             f" customer index chosen: {customer_index}")
        if customer_index == self.DEPOT_INDEX and np.all(self.current_state.current_vehicle_positions[vehicle_index, :]
                                                         == self.current_state.depot_position):
            raise ValueError(f"t:{self.env.now}, vehicle at depot and depot chosen! \n"
                             f"vehicle index {vehicle_index}")
        # updated current state that the customer has been chosen
        self.notify_customer_chosen(customer_index, vehicle_index)
        travel_info = self.get_travel_info(customer_index, vehicle_index)
        travel_completion = self.env.timeout(travel_info['travel_time'], value={'customer_index': customer_index,
                                                                                'vehicle_index': vehicle_index})
        travel_completion.callbacks.append(self.notify_customer_reached)

        self.run_to_decision_point()
        current_full_reward = -np.sum(self.current_state.vehicle_full_distance)
        reward = current_full_reward - self.previous_reward
        self.previous_reward = current_full_reward
        is_done = self.is_done()
        obs = self.current_state_to_observation()
        return obs, reward, is_done, {}

    def notify_customer_opened(self, event):
        # this function is used to update the status of the customer from idle to opened
        current_status = self.current_state.customer_status[event.value]
        if current_status != CUSTOMER_STATUS['idle']:
            raise ValueError(f"time:{self.env.now}, simulation trying to change non idle customer status to open.\n"
                             f"customer index:{event.value}, customer current status:{current_status}")
        self.current_state.customer_status[event.value] = CUSTOMER_STATUS['opened']

    def notify_customer_chosen(self, customer_index, vehicle_index):
        """
        update vehicle status to "busy" and change capacity (if going to depot capacity is reset otherwise updated based
        on customers demand. in addition update customer status to "chosen"
        :param customer_index: index of customer chosen
        :param vehicle_index: vehicle index chosen for this customer
        :return:
        """
        current_vehicle_customer = self.current_state.current_vehicle_customer[vehicle_index]
        if current_vehicle_customer is not None:
            raise ValueError(f"time:{self.env.now}, trying to chose a customer with vehicle who already has a chosen "
                             f"customer.\n "
                             f"vehicle -> index:{vehicle_index}, current customer:{current_vehicle_customer},"
                             f"status:{self.current_state.vehicle_status[vehicle_index]}\n"
                             f"customer -> index:{customer_index}")
        if self.current_state.vehicle_status[vehicle_index] == VEHICLE_STATUS['busy']:
            raise ValueError(f"time:{self.env.now}, trying to chose a busy vehicle.\n "
                             f"vehicle -> index:{vehicle_index}, current customer:{current_vehicle_customer},"
                             f"status:{self.current_state.vehicle_status[vehicle_index]}\n"
                             f"customer -> index:{customer_index}")
        if customer_index != self.DEPOT_INDEX:
            if self.current_state.customer_status[customer_index] != CUSTOMER_STATUS['opened']:
                raise ValueError(f"time:{self.env.now}, trying to chose a customer which is not opened.\n "
                                 f"vehicle -> index:{vehicle_index}, current customer:{current_vehicle_customer},"
                                 f"status:{self.current_state.vehicle_status[vehicle_index]}\n"
                                 f"customer -> index:{customer_index}, "
                                 f"status:{self.current_state.customer_status[customer_index]}")
            # make sure vehicle capacity is larger than customer demand
            current_capacity = self.current_state.current_vehicle_capacities[vehicle_index]
            customer_demand = self.current_state.customer_demands[customer_index]
            if customer_demand > current_capacity:
                raise ValueError(f"time {self.env.now}, customer chosen has capacity larger than vehicle demand.\n"
                                 f"vehicle info: index {vehicle_index}, capacity {current_capacity}\n"
                                 f"customer info: index {customer_index}, demand {customer_demand}")
            self.current_state.customer_status[customer_index] = CUSTOMER_STATUS['chosen']
        # update vehicle status to busy and chosen customer index for vehicle
        self.current_state.current_vehicle_customer[vehicle_index] = customer_index
        self.current_state.vehicle_full_path[vehicle_index].append(customer_index)
        self.notify_vehicle_busy(vehicle_index)

    def notify_customer_reached(self, event):
        customer_index = event.value['customer_index']
        vehicle_index = event.value['vehicle_index']
        # update vehicle status
        self.notify_vehicle_available(vehicle_index)
        current_vehicle_position = self.current_state.current_vehicle_positions[vehicle_index, :]
        if customer_index == self.DEPOT_INDEX:
            # depot is chosen, reset vehicle capacity
            original_capacity = self.initial_state.current_vehicle_capacities[vehicle_index]
            self.current_state.current_vehicle_capacities[vehicle_index] = original_capacity
            new_position = deepcopy(self.current_state.depot_position)
        else:
            # customer is chosen for vehicle. update vehicle capacity
            current_capacity = self.current_state.current_vehicle_capacities[vehicle_index]
            customer_demand = self.current_state.customer_demands[customer_index]
            self.current_state.current_vehicle_capacities[vehicle_index] = current_capacity - customer_demand
            # update customer status
            self.notify_customer_visited(customer_index)
            # update vehicle position
            new_position = deepcopy(self.current_state.customer_positions[customer_index, :])

        distance_traveled = np.linalg.norm(new_position - current_vehicle_position)
        self.current_state.vehicle_full_distance[vehicle_index] += distance_traveled
        self.current_state.current_vehicle_positions[vehicle_index, :] = new_position
        # move each busy vehicle towards the customer it is trying to reach
        busy_vehicles_indexes = np.where(self.current_state.vehicle_status == VEHICLE_STATUS['busy'])[0]
        delta_t = self.env.now - self.current_state.previous_time
        for b_v in busy_vehicles_indexes:
            vehicle_customer = self.current_state.current_vehicle_customer[b_v]
            if vehicle_customer == self.DEPOT_INDEX:
                # vehicle is moving towards the depot
                target_position = self.current_state.depot_position
            else:
                # vehicle is moving towards a customer
                target_position = self.current_state.customer_positions[vehicle_customer, :]
            current_busy_vehicle_position = self.current_state.current_vehicle_positions[b_v, :]
            movement_angle = np.arctan2(target_position[1] - current_busy_vehicle_position[1],
                                        target_position[0] - current_busy_vehicle_position[0])
            # the delta position is the velocity * time in the direction of the final target movement
            vehicle_velocity = self.current_state.vehicle_velocity
            delta_position_distance = vehicle_velocity * delta_t
            delta_position_x = delta_position_distance * np.cos(movement_angle)
            delta_position_y = delta_position_distance * np.sin(movement_angle)
            delta_position = np.array([delta_position_x, delta_position_y])
            busy_vehicle_new_position = current_busy_vehicle_position + delta_position
            distance_traveled = np.linalg.norm(busy_vehicle_new_position - current_busy_vehicle_position)
            self.current_state.vehicle_full_distance[b_v] += distance_traveled
            self.current_state.current_vehicle_positions[b_v, :] = busy_vehicle_new_position

    def notify_customer_visited(self, event):
        self.current_state.customer_status[event] = CUSTOMER_STATUS['visited']

    def notify_vehicle_busy(self, event):
        self.current_state.vehicle_status[event] = VEHICLE_STATUS['busy']

    def notify_vehicle_available(self, event):
        self.current_state.vehicle_status[event] = VEHICLE_STATUS['available']
        self.current_state.current_vehicle_customer[event] = None

    def get_travel_info(self, customer_index, vehicle_index):
        vehicle_position = self.current_state.current_vehicle_positions[vehicle_index, :]
        if customer_index == self.DEPOT_INDEX:
            target_position = self.current_state.depot_position
        else:
            target_position = self.current_state.customer_positions[customer_index, :]
        travel_distance = np.linalg.norm(target_position - vehicle_position)
        # we assume that the vehicle velocity given in the current state is the velocity in x and y axis, therefore
        # the full velocity in the movement direction is sqrt(2) * velocity
        travel_time = travel_distance / self.current_state.vehicle_velocity
        return {'travel_time': travel_time, 'travel_distance': travel_distance}

    def is_done(self):
        return self.env.now >= self.until

    def run_to_decision_point(self):
        # run until there is a free machine and at least one waiting (and released) job,
        # or the termination time is reached. Adding a small value to self.env.now prevents floating point issues.
        while not self.is_done() and (
                # keep stepping if no vehicles are available
                self.get_num_available_vehicles() == 0
                # keep stepping if no customers are waiting
                or self.get_num_opened_customers() == 0
        ):
            self.env.step()
        # this ensures that any not-yet-processed events that are supposed to happen at the
        # simulation time NOW are processed before control is passed back to the policy
        while self.env.peek() == self.env.now:
            self.env.step()

    def current_state_to_observation(self) -> Dict:
        """
        this function transforms the current state to dictionary for gym env observation
        :return: obs [Dict]
        """
        customer_positions = deepcopy(self.current_state.customer_positions)
        customer_demands = deepcopy(self.current_state.customer_demands)
        customer_ids = deepcopy(self.current_state.customer_ids)
        illegal_actions = self.get_current_illegal_actions()
        depot_position = deepcopy(self.current_state.depot_position)
        current_vehicle_positions = deepcopy(self.current_state.current_vehicle_positions)
        current_vehicle_capacities = deepcopy(self.current_state.current_vehicle_capacities)
        current_vehicle_customer = deepcopy(self.current_state.current_vehicle_customer)
        max_vehicle_capacity = np.max(self.initial_state.current_vehicle_capacities).item()
        customer_status = deepcopy(self.current_state.customer_status)
        vehicle_status = deepcopy(self.current_state.vehicle_status)
        obs = {
            "customer_positions": customer_positions,
            "customer_demands": customer_demands,
            "customer_ids": customer_ids,
            "illegal_actions": illegal_actions,
            "depot_position": depot_position,
            "current_vehicle_positions": current_vehicle_positions,
            "current_vehicle_capacities": current_vehicle_capacities,
            "max_vehicle_capacity": max_vehicle_capacity,
            "current_vehicle_customer": current_vehicle_customer,
            "customer_status": customer_status,
            "vehicle_status": vehicle_status
        }
        return obs

    def get_num_available_vehicles(self) -> int:
        """
        this function returns the number of currently available vehicles
        :return: number of free vehicles [int]
        """
        num_free_vehicles = np.sum(self.current_state.vehicle_status == VEHICLE_STATUS['available']).item()
        return num_free_vehicles

    def get_num_opened_customers(self) -> int:
        """
        this function calculates the number of opened customers
        :return: num_free_customers [int]
        """
        num_opened_customers = np.sum(self.current_state.customer_status == CUSTOMER_STATUS['opened']).item()
        return num_opened_customers

    def get_current_illegal_actions(self) -> np.ndarray:
        """
        get matrix of current illegal actions. illegal_action = True if the action is not possible.
        an action [i, j] is possible if:
        * vehicle i is available
        * customer j is opened
        * capacity of vehicle i is larger than customer j demand
        :return: illegal_actions np.ndarray of size [num_vehicles, num_customers + 2]
        """
        num_customers = self.current_state.customer_positions.shape[0]
        illegal_actions = np.ones(shape=[self.num_vehicles, num_customers + 2]).astype(np.bool)
        available_vehicles = np.where(self.current_state.vehicle_status == VEHICLE_STATUS['available'])[0]
        opened_customers = np.where(self.current_state.customer_status == CUSTOMER_STATUS['opened'])[0]
        for i_v in available_vehicles:
            for i_c in opened_customers:
                if self.current_state.customer_demands[i_c] <= self.current_state.current_vehicle_capacities[i_v]:
                    # if vehicle capacity is larger than customer demand - this is a feasible action
                    illegal_actions[i_v, i_c] = 0
            if np.any(self.current_state.current_vehicle_positions[i_v, :] != self.current_state.depot_position):
                # if vehicle is not at the depot position then depot action is feasible
                illegal_actions[i_v, num_customers] = 0
        return illegal_actions
