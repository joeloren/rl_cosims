# basic imports
from abc import ABC, abstractmethod
from copy import deepcopy
# mathematical imports
import numpy as np
from scipy import stats
# our imports
from src.cvrp_simulation.distributions.mixture_distribution import MixtureModel
from src.cvrp_simulation.simulation.simulator import State


class ScenarioGenerator(ABC):
    @abstractmethod
    def seed(self, seed: int) -> None:
        """Sets the random seed for the arrival process. """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the arrival process"""
        raise NotImplementedError

    @abstractmethod
    def reset_customer(self, t) -> dict:
        """resets a specific customer properties, customer opens after time t"""
        raise NotImplementedError


class SampleStaticBenchmark(ScenarioGenerator):
    """
    this class creates a generator for the cvrp problem based on the benchmark database
    """

    def __init__(
        self,
        depot_position_rv: stats._distn_infrastructure.rv_generic,
        vehicle_position_rv: stats._distn_infrastructure.rv_generic,
        vehicle_capacity: stats._distn_infrastructure.rv_generic,
        vehicle_velocity: stats._distn_infrastructure.rv_generic,
        customer_positions_rv: stats._distn_infrastructure.rv_generic,
        customer_demands_rv: stats._distn_infrastructure.rv_generic,
        vrp_size: int,
        start_at_depot: bool = False,
    ) -> None:
        """A ScenarioGenerator for the cvrp problem which generates a random problem each time based on distributions
        wanted for each variable
        :param depot_position_rv: the depot position (scipy) random variable
        :param vehicle_position_rv: the vehicles starting position (scipy) random variable
        :param vehicle_capacity: the vehicles total capacity [int], for now this is constant and pre-defined
        :param vehicle_velocity: the vehicles velocity [int], for now this is constant and pre-defined
        :param customer_positions_rv: customer positions (scipy) random variable,
        :param customer_demands_rv: customer demands (scipy) random variable,
        :param vrp_size: number of customers [int]
        :param start_at_depot: if `True`, the initial position of the vehicle is the depot, else drawn from
        `vehicle_position_rv`
        """
        super().__init__()
        self.depot_position_rv = depot_position_rv
        self.vehicle_position_rv = vehicle_position_rv
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_velocity = vehicle_velocity
        self.customer_positions_rv = customer_positions_rv
        self.customer_demands_rv = customer_demands_rv
        self.vrp_size = vrp_size
        self.start_at_depot = start_at_depot

    def reset(self) -> State:
        """
        this function creates a new state based on all random variables and distributions
        :return: state [State] - new problem state
        """
        depot_pos = self.depot_position_rv.rvs(2)
        state = State(
            depot_position=deepcopy(depot_pos),  # [x,y]
            current_vehicle_position=deepcopy(depot_pos)
            if self.start_at_depot
            else self.vehicle_position_rv.rvs(2),  # [x,y]
            current_vehicle_capacity=self.vehicle_capacity,
            vehicle_velocity=self.vehicle_velocity,
            customer_positions=self.customer_positions_rv.rvs([self.vrp_size, 2]),   # [x,y] for each customer 0, 1,..N
            customer_demands=self.customer_demands_rv.rvs(self.vrp_size),  # [N]
            customer_times=np.zeros(self.vrp_size).astype(int),
            customer_ids=np.arange(0, self.vrp_size).astype(np.int),
            customer_visited=np.zeros(self.vrp_size).astype(np.bool)  # all customers start as not visited
        )
        return state

    def seed(self, seed: int) -> None:
        """
        this function initializes the seed of the generator
        (generator uses scipy random generators which are bases on numpy seed)
        :param seed: seed to be used [int]
        :return:
        """
        np.random.seed(seed)

    def reset_customer(self, t):
        """
        this function receives the current time and returns one
         future customer based on wanted distribution
        in the static case there are no future customers so there is no need to return anything
        """
        return None


class SampleDynamicBenchmark(ScenarioGenerator):
    """
    this class creates a generator for the cvrp problem based on the benchmark database
    """

    def __init__(
        self,
        depot_position_rv: stats._distn_infrastructure.rv_generic,
        vehicle_position_rv: stats._distn_infrastructure.rv_generic,
        vehicle_capacity: stats._distn_infrastructure.rv_generic,
        vehicle_velocity: stats._distn_infrastructure.rv_generic,
        customer_positions_rv: stats._distn_infrastructure.rv_generic,
        customer_demands_rv: stats._distn_infrastructure.rv_generic,
        customer_times_rv: stats._distn_infrastructure.rv_generic,
        vrp_size: int,
        start_at_depot: bool,
    ) -> None:
        """A ScenarioGenerator for the cvrp problem which generates a random problem
        each time based on distributions wanted for each variable
        :param depot_position_rv: the depot position (scipy) random variable
        :param vehicle_position_rv: the vehicles starting position (scipy) random variable
        :param vehicle_capacity: the vehicles total capacity [int], for now this is constant
         and pre-defined
        :param vehicle_velocity: the vehicles velocity [int], for now this is constant
         and pre-defined
        :param customer_positions_rv: customer positions (scipy) random
         variable for each axis [x, y],
        :param customer_demands_rv: customer demands (scipy) random variable,
        :param customer_times_rv: customer start times (scipy) random variable,
        :param vrp_size: number of customers [int]
        :param start_at_depot: if `True`, the initial position of the vehicle is the depot,
         else drawn from `vehicle_position_rv`
        """
        super().__init__()
        self.depot_position_rv = depot_position_rv
        self.vehicle_position_rv = vehicle_position_rv
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_velocity = vehicle_velocity
        self.customer_positions_rv = customer_positions_rv
        self.customer_demands_rv = customer_demands_rv
        self.customer_times_rv = customer_times_rv
        self.vrp_size = vrp_size
        self.start_at_depot = start_at_depot

    def reset(self) -> State:
        """
        this function creates a new state based on all random variables and distributions
        :return: state [State] - new problem state
        """
        depot_pos = self.depot_position_rv.rvs(2)
        if isinstance(self.customer_positions_rv, MixtureModel):
            # output of rvs is [2, N] therefore need to transpose to get [N, 2]
            customer_positions = np.transpose(self.customer_positions_rv.rvs(self.vrp_size))
        else:
            customer_positions = self.customer_positions_rv.rvs([self.vrp_size, 2])
        customer_times = np.sort(self.customer_times_rv.rvs(self.vrp_size))
        state = State(
            depot_position=deepcopy(depot_pos),  # [x,y]
            current_vehicle_position=deepcopy(depot_pos)
            if self.start_at_depot
            else self.vehicle_position_rv.rvs(2),
            # [x,y]
            current_vehicle_capacity=self.vehicle_capacity,
            vehicle_velocity=self.vehicle_velocity,
            customer_positions=deepcopy(
                customer_positions
            ),  # [x,y] for each customer 0, 1,..N
            customer_demands=self.customer_demands_rv.rvs(self.vrp_size),  # [N]
            customer_times=customer_times,  # [N] this is the start time of each customer
            customer_ids=np.arange(0, self.vrp_size).astype(np.int),
            customer_visited=np.zeros(self.vrp_size).astype(np.bool)
            # all customers start as not visited
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

    def reset_customer(self, t):
        """
        this function receives the current time and returns one future customer properties based on wanted distribution
        :param t - current simulation time
        : return
        """
        customer_properties = {}
        # create a new position from distribution
        if isinstance(self.customer_positions_rv, MixtureModel):
            # MixtureModel for position return [x, y] for each rvs since it is a 2d distribution
            # output of rvs is [2, N] therefore need to transpose to get [N, 2]
            customer_position = np.transpose(self.customer_positions_rv.rvs(size=1))
        else:
            # uniform for position return 1 number for each rvs since it is a 1d distribution
            # therefore we want to
            # choose 2*N random numbers and reshape to [x, y] for each customer
            customer_position = self.customer_positions_rv.rvs(size=[1, 2])
        customer_properties["position"] = customer_position
        # create a new demand from distribution
        customer_properties["demand"] = self.customer_demands_rv.rvs(size=1)
        # customer_properties['time'] = self.customer_times_rv.rvs(1)
        # create a new time from distribution but only accept time that is larger than current
        # simulation time
        # allow 6000 tries
        for i in range(6000):
            customer_time = self.customer_times_rv.rvs(1)
            if customer_time > t:
                customer_properties["time"] = customer_time
                return customer_properties
        customer_properties["time"] = t + 0.001
        print(
            f"ran 6000 times and did not find a time for "
            f"future customer that is larger than current time:{t:.3f},"
            f"using current time as opening time for customer"
        )
        return customer_properties

    def reset_customers(self, t, n_customer_to_create):
        """
        this function receives the current time and number of customers to create
        and returns one future customer properties based on wanted distribution
        :param t - current simulation time
        :param n_customer_to_create - number of customers to create new distributions for
        : return
        """
        customer_properties = {}
        # create a new position from distribution
        if isinstance(self.customer_positions_rv, MixtureModel):
            # MixtureModel for position return [x, y] for each rvs since it is a 2d distribution
            # output of rvs is [2, N] therefore need to transpose to get [N, 2]
            customer_position = np.transpose(
                self.customer_positions_rv.rvs(size=n_customer_to_create)
            )
        else:
            # uniform for position return 1 number for each rvs since it is a 1d distribution
            # therefore we want to
            # choose 2*N random numbers and reshape to [x, y] for each customer
            customer_position = self.customer_positions_rv.rvs(
                size=[n_customer_to_create, 2]
            )
        customer_properties["position"] = customer_position
        # create a new demand from distribution
        customer_properties["demand"] = self.customer_demands_rv.rvs(
            size=n_customer_to_create
        )
        # create a new time from distribution but only accept time that is larger than current simulation time
        # allow 250 tries
        customer_times_used = np.zeros(shape=n_customer_to_create)
        customer_times = self.customer_times_rv.rvs(n_customer_to_create * 200)
        indexs_to_use = np.where(customer_times > t)[0]
        if indexs_to_use.size >= n_customer_to_create:
            customer_times_used = customer_times[indexs_to_use[:n_customer_to_create]]
        else:
            n_missing_customers = n_customer_to_create - indexs_to_use.size
            customer_times_used[: indexs_to_use.size] = customer_times[indexs_to_use]
            customer_times_used[indexs_to_use.size:] = np.random.uniform(t, self.customer_times_rv.b )
            print(
                f"ran 100 times and did not find a time for "
                f"{n_missing_customers} future customers that is larger than current time:"
                f"{t:.3f},"
                f"using uniform between current time and final simulation time - "
                f"{self.customer_times_rv.b}"
            )
        customer_properties["time"] = customer_times_used
        return customer_properties


# fixed sample
class FixedSample(ScenarioGenerator):
    """
    this class creates a generator for the cvrp problem which always produces the same problem
    (used mainly for debugging)
    """

    def __init__(
        self,
        depot_position: np.ndarray,
        vehicle_position: np.ndarray,
        vehicle_capacity: int,
        vehicle_velocity: int,
        customer_positions: np.ndarray,
        customer_demands: np.ndarray,
        customer_times: np.ndarray,
    ) -> None:
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
            customer_visited=np.zeros(self.vrp_size).astype(np.bool),
        )
        return state

    def seed(self, seed: int) -> None:
        # there is nothing random about this class therefore no need to define the seed
        pass

    def reset_customer(self, t):
        """
        this function receives the current time and returns one future customer based on wanted
        distribution
        in the fixed case there are no distributions so there is no need for this function and it
        will return None
        """
        return None
