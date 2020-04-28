from abc import ABC, abstractmethod

import numpy as np
from scipy import stats

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
    this class creates a generator for the cvrp problem based on the benchmark database
    """
    def __init__(self,
                 depot_position_rv: stats._distn_infrastructure.rv_generic,
                 vehicle_position_rv: stats._distn_infrastructure.rv_generic,
                 vehicle_capacity: stats._distn_infrastructure.rv_generic,
                 vehicle_velocity: stats._distn_infrastructure.rv_generic,
                 customer_positions_rv: stats._distn_infrastructure.rv_generic,
                 customer_demands_rv: stats._distn_infrastructure.rv_generic,
                 vrp_size: int) -> None:
        """A ScenarioGenerator for the cvrp problem which generates a random problem each time based on
         distributions wanted for each variable
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
        """
        this function creates a new state based on all random variables and distributions
        :return: state [State] - new problem state
        """
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
        """
        this function initializes the seed of the generator (generator uses scipy random generators which are bases on
        numpy seed)
        :param seed: seed to be used [int]
        :return:
        """
        np.random.seed(seed)


class SampleDynamicBenchmark(ScenarioGenerator):
    """
    this class creates a generator for the cvrp problem based on the benchmark database
    """
    def __init__(self,
                 depot_position_rv: stats._distn_infrastructure.rv_generic,
                 vehicle_position_rv: stats._distn_infrastructure.rv_generic,
                 vehicle_capacity: stats._distn_infrastructure.rv_generic,
                 vehicle_velocity: stats._distn_infrastructure.rv_generic,
                 customer_positions_rv: stats._distn_infrastructure.rv_generic,
                 customer_demands_rv: stats._distn_infrastructure.rv_generic,
                 customer_times_rv: stats._distn_infrastructure.rv_generic,
                 vrp_size: int) -> None:
        """A ScenarioGenerator for the cvrp problem which generates a random problem each time based on
         distributions wanted for each variable
        :param depot_position_rv: the depot position (scipy) random variable
        :param vehicle_position_rv: the vehicles starting position (scipy) random variable
        :param vehicle_capacity: the vehicles total capacity [int], for now this is constant and pre-defined
        :param vehicle_velocity: the vehicles velocity [int], for now this is constant and pre-defined
        :param customer_positions_rv: customer positions (scipy) random variable,
        :param customer_demands_rv: customer demands (scipy) random variable,
        :param customer_times_rv: customer start times (scipy) random variable,
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
        self.customer_times_rv = customer_times_rv
        self.vrp_size = vrp_size

    def reset(self) -> State:
        """
        this function creates a new state based on all random variables and distributions
        :return: state [State] - new problem state
        """
        state = State(
            depot_position=self.depot_position_rv.rvs(2),  # [x,y]
            current_vehicle_position=self.vehicle_position_rv.rvs(2),  # [x,y]
            current_vehicle_capacity=self.vehicle_capacity,
            vehicle_velocity=self.vehicle_velocity,
            customer_positions=self.customer_positions_rv.rvs([self.vrp_size, 2]),  # [x,y] for each customer 0, 1,..N
            customer_demands=self.customer_demands_rv.rvs(self.vrp_size),  # [N]
            customer_times=self.customer_times_rv.rvs(self.vrp_size),  # [N] this is the start time of each customer
            customer_ids=np.arange(0, self.vrp_size).astype(np.int),
            customer_visited=np.zeros(self.vrp_size).astype(np.bool)  # all customers start as not visited
        )
        return state

    def seed(self, seed: int) -> None:
        """
        this function initializes the seed of the generator (generator uses scipy random generators which are bases on
        numpy seed)
        :param seed: seed to be used [int]
        :return:
        """
        np.random.seed(seed)


# fixed sample
class FixedSample(ScenarioGenerator):
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
        """
        initializing each variable with wanted numbers
        :param depot_position: position of depot [x, y]
        :param vehicle_position: vehicle starting position [x, y]
        :param vehicle_capacity: vehicle maximum capacity [int]
        :param vehicle_velocity: vehicle velocity (needed for calculation step time) [int]
        :param customer_positions: customer positions, each row [x, y] and vector is [N, 2]
        :param customer_demands: customer demands [N]
        :param customer_times: customer start time [N] -
        in the future could be a vec with end time and then will be [N, 2]
        """
        super().__init__()
        self.depot_position = depot_position
        self.vehicle_position = vehicle_position
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_velocity = vehicle_velocity
        self.customer_positions = customer_positions
        self.customer_demands = customer_demands
        self.customer_times = customer_times
        self.vrp_size = customer_demands.size  # total number of customers in the problem [int]

    def reset(self) -> State:
        """
        this function creates a state with all the information wanted for the specific problem
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
