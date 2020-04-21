from copy import deepcopy
from dataclasses import dataclass

import numpy as np


@dataclass
class State:
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


class CVRPSimulation:
    EPSILON_TIME = 1e-6

    def __init__(self,
                 depot_position: np.array,
                 initial_vehicle_position: np.array,
                 initial_vehicle_capacity: int,
                 vehicle_velocity: int,
                 customer_positions: np.array,
                 customer_demands: np.array,
                 customer_times: np.ndarray,
                 customer_ids: np.ndarray,
                 customer_visited: np.ndarray) -> None:
        """
        Create a new cvrp_simulation. Note that you need to call reset() before starting cvrp_simulation.
        """
        self.initial_state = State(
            depot_position=depot_position,
            current_vehicle_position=initial_vehicle_position,
            current_vehicle_capacity=initial_vehicle_capacity,
            vehicle_velocity=vehicle_velocity,
            customer_positions=customer_positions,
            customer_demands=customer_demands,
            customer_times=customer_times,
            customer_ids=customer_ids,
            customer_visited=customer_visited
        )

        self.current_state = None
        self.current_time = 0
        # TODO understand if these are needed
        self.jobs_completed_since_last_step = []
        self.current_state_value = 0.0

    def reset(self) -> None:
        self.current_state = deepcopy(self.initial_state)
        self.current_time = 0

    def set_random_seed(self, seed: int) -> None:
        raise NotImplementedError

    # todo: do we need to define a deepcopy function or does deepcopy work as-is?
    def step(self, customer_index) -> (float, bool):
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
        return -traveled_distance, is_done

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
        if self.current_state.customer_visited.all():
            return True
        else:
            return False
