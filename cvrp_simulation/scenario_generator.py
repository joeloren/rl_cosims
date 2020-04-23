from abc import ABC, abstractmethod

import numpy as np

from cvrp_simulation.simulator import State


class ScenarioGenerator(ABC):

    @abstractmethod
    def seed(self, seed: int) -> None:
        """Sets the random seed for the arrival process. """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the arrival process"""
        raise NotImplementedError


class SampleStaticBenchmark(ScenarioGenerator):
    """
    this class creates a generator for the cvrp problem
    """
    def __init__(self, depot_position_rv, vehicle_position_rv, vehicle_capacity, vehicle_velocity,
                 customer_positions_rv, customer_demands_rv,
                 vrp_size: int) -> None:
        """A ScenarioGenerator for the cvrp problem
        :param depot_position_rv: the depot position (scipy) random variable
        :param vehicle_position_rv: the vehicles starting position (scipy) random variable
        :param vehicle_capacity: the vehicles total capacity [int], for now this is constant and pre-defined
        :param vehicle_velocity: the vehicles velocity [int], for now this is constant and pre-defined
        :param customer_positions_rv: customer positions (scipy) random variable,
        :param customer_demands_rv: customer demands (scipy) random variable,
        :param vrp_size: number of customers [int]
        """
        # TODO if customer time becomes a window need to add another customer time variable ??
        super().__init__()
        self.depot_position_rv = depot_position_rv
        self.vehicle_position_rv = vehicle_position_rv
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_velocity = vehicle_velocity
        self.customer_positions_rv = customer_positions_rv
        self.customer_demands_rv = customer_demands_rv
        self.vrp_size = vrp_size

    def reset(self) -> State:
        state = State(
            depot_position=self.depot_position_rv.rvs(2),  # [x,y]
            current_vehicle_position=self.vehicle_position_rv.rvs(2),  # [x,y]
            current_vehicle_capacity=self.vehicle_capacity,
            vehicle_velocity=self.vehicle_velocity,
            customer_positions=self.customer_positions_rv.rvs([self.vrp_size, 2]),  # [x,y] for each customer 0, 1,..N
            customer_demands=self.customer_demands_rv.rvs(self.vrp_size),  # [N]
            customer_times=np.zeros(self.vrp_size).astype(int),
            customer_ids=np.arange(0, self.vrp_size).astype(np.int),
            customer_visited=np.zeros(self.vrp_size).astype(np.bool)  # all customers start as not visited
        )
        return state

    def seed(self, seed: int) -> None:
        np.random.seed(seed)


class SpecificSample(ScenarioGenerator):
    """
    this class creates a generator for the cvrp problem which always produces the same problem
    (used mainly for debugging)
    """
    def __init__(self, depot_position: np.ndarray,
                 vehicle_position: np.ndarray,
                 vehicle_capacity: int,
                 vehicle_velocity: int,
                 customer_positions: np.ndarray,
                 customer_demands: np.ndarray,
                 customer_times: np.ndarray) -> None:
        super().__init__()
        self.depot_position = depot_position
        self.vehicle_position = vehicle_position
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_velocity = vehicle_velocity
        self.customer_positions = customer_positions
        self.customer_demands = customer_demands
        self.customer_times = customer_times
        self.vrp_size = customer_demands.size

    def reset(self) -> State:
        """
        this function creates a state with all the information wanted
        :return:
        """
        state = State(
            depot_position=self.depot_position,
            current_vehicle_position=self.vehicle_position,
            current_vehicle_capacity=self.vehicle_capacity,
            vehicle_velocity=self.vehicle_velocity,
            customer_positions=self.customer_positions,
            customer_demands=self.customer_demands,
            customer_times=self.customer_times,
            customer_ids=np.arange(0, self.vrp_size).astype(np.int),
            customer_visited=np.zeros(self.vrp_size).astype(np.bool)
        )
        return state

    def seed(self, seed: int) -> None:
        # there is nothing random about this class therefore no need to define the seed
        pass
