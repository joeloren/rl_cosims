# basic imports
from functools import partial
# mathematical imports
import numpy as np
from matplotlib import pyplot as plt
from ortools.constraint_solver import pywrapcp
from scipy import stats
# our imports
from src.envs.cvrp.cvrp_baselines import simple_baseline
from src.envs.cvrp.cvrp_utils.plot_results import plot_vehicle_routes
from src.envs.cvrp.cvrp_simulation.scenario_generator import (FixedSample, SampleStaticBenchmark)
from src.envs.cvrp.cvrp_simulation.simulator import CVRPSimulation


def create_data_model(obs, env, precision=1000):
    """Stores the data for the problem"""
    data = {}
    _capacity = obs["max_vehicle_capacity"]
    _locations = [
        obs["depot_position"] * precision,  # depot
    ]
    num_customers = env.get_opened_and_not_visited_customers().size
    _locations.extend(
        [obs["customer_positions"][i, :] * precision for i in range(num_customers)]
    )  # locations to visit
    data["locations"] = _locations
    data["num_locations"] = len(data["locations"])
    data["demands"] = [0]  # depot
    # customers demands
    data["demands"].extend([obs["customer_demands"][i] for i in range(num_customers)])
    data["vehicle_capacity"] = _capacity
    data["depot"] = 0
    data["num_customers"] = num_customers
    data["ids"] = [-1]  # depot
    data["ids"].extend([obs["customer_ids"][i] for i in range(num_customers)])
    return data


#######################
# Problem Constraints #
#######################
def create_distance_evaluator(data):
    """Creates callback to return distance between points."""
    _distances = {}
    # precompute distance between location to have distance callback in O(1)
    for from_node in range(data["num_locations"]):
        _distances[from_node] = {}
        for to_node in range(data["num_locations"]):
            if from_node == to_node:
                _distances[from_node][to_node] = 0
            else:
                _distances[from_node][to_node] = np.linalg.norm(data["locations"][from_node] -
                                                                data["locations"][to_node])

    def distance_evaluator(manager, from_node, to_node):
        return _distances[manager.IndexToNode(from_node)][manager.IndexToNode(to_node)]

    return distance_evaluator


def add_distance_dimension(routing, distance_evaluator_index):
    """Add Global Span constraint"""
    distance = "Distance"
    routing.AddDimension(
        distance_evaluator_index,
        0,  # null slack
        10000,  # maximum distance per vehicle
        True,  # start cumulative variable exactly at zero without slack
        distance,
    )
    distance_dimension = routing.GetDimensionOrDie(distance)
    # Try to minimize the total distance of the vehicle
    distance_dimension.SetGlobalSpanCostCoefficient(100)


def create_demand_evaluator(data):
    """Creates callback to get demands at each location."""
    _demands = data["demands"]

    def demand_evaluator(manager, from_node):
        """Returns the demand of the current node"""
        return _demands[manager.IndexToNode(from_node)]

    return demand_evaluator


def add_capacity_constraints(routing, manager, data, demand_evaluator_index):
    """Adds capacity constraint"""
    vehicle_capacity = data["vehicle_capacity"]
    capacity = "Capacity"
    routing.AddDimension(
        demand_evaluator_index,
        0,  # Null slack
        vehicle_capacity,
        True,  # start cumul to zero
        capacity,
    )


###########
# Printer #
###########
def print_solution(manager, routing, assignment, data, precision):  # pylint:disable=too-many-locals
    """Prints assignment on console"""
    num_customers = data["num_customers"]
    print("Objective: {}".format(assignment.ObjectiveValue()))
    capacity_dimension = routing.GetDimensionOrDie("Capacity")
    dropped = []
    for order in range(1, routing.nodes()):
        index = manager.NodeToIndex(order)
        if assignment.Value(routing.NextVar(index)) == index:
            dropped.append(order - 1)
    print("dropped orders: {}".format(dropped))

    total_route_distance = 0
    for vehicle in range(num_customers):
        index = routing.Start(vehicle)
        plan_output = "Route for vehicle {}:\n".format(vehicle)
        distance = 0
        while not routing.IsEnd(index):
            load_var = capacity_dimension.CumulVar(index)
            node = manager.IndexToNode(index)
            node_string = data["ids"][node] if node > 0 else "depot"
            plan_output += "-Load({1})-> {0}  ".format(
                node_string, assignment.Value(load_var)
            )
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        load_var = capacity_dimension.CumulVar(index)
        node = manager.IndexToNode(index)
        node_string = node - 1 if node > 0 else "depot"
        plan_output += "-Load({1})-> {0}\n".format(
            node_string, assignment.Value(load_var)
        )
        plan_output += "Distance of the route: {}m\n".format(distance / precision)
        plan_output += "Load of the route: {}\n".format(assignment.Value(load_var))
        print(plan_output)
        total_route_distance = total_route_distance + distance
    print("total distance: {}m".format(total_route_distance / precision))


########
# Main #
########


class ORToolsPolicy:
    def __init__(self, precision=1000, verbose=False, timeout=10) -> None:
        super().__init__()
        self.timeout = timeout
        self.verbose = verbose
        self.precision = precision
        self.__name__ = "ortools"
        self.assignment = None
        self.manager = None
        self.routing = None
        self.data = None
        self.next = None

    def reset(self, obs):
        self.assignment = None
        self.manager = None
        self.routing = None
        self.data = None
        self.next = None

    def __call__(self, obs, env):
        if len(obs["action_mask"]) == 2:
            probs = np.zeros_like(obs["action_mask"])
            # there are no customers opened in problem therefore the algorithm chooses the noop
            # option
            # if noop is disabled chooses the depot
            if obs["action_mask"][-1]:
                probs[-1] = 1
            else:
                probs[-2] = 1
            return probs
        elif (
            self.assignment is None
            or (obs["current_vehicle_position"] == obs["depot_position"]).all()
        ):
            # only recompute the route if we are at the depot. Conceptually, we could recompute
            # it at every time step.
            # This would however require the capability to deal with arbitrary starting
            # positions. We could not
            # get this to work properly despite our best efforts. We tried two approaches:
            # * if you model the problem multiple vehicles then you have the problem that OR
            # tools does not deal with
            #   empty routes properly. Routes that do not visit any customers between start and
            #   depot automatically have
            #   distance 0 regardless of actual distance. The distance callback is not even
            #   called. Obviously, this
            #   gives bad results, since the vehicles can "teleport" from the current position to
            #   the depot.
            # * If you model the problem with multiple copies of the depot ("unload depots") as in
            #    """based on https://github.com/google/or-tools/blob/master/ortools
            #    /constraint_solver/samples/cvrp_reload.py"""
            #   then the heuristics (all of them) for constructing initial solutions fail almost
            #   always. Supplying an
            #   initial solution also did not work. Despite obviously feasible initial routes,
            #   the solver did not accept
            #   them as feasible, even after extensive efforts to determine why they fail.

            # Instantiate the data problem.
            self.data = create_data_model(obs, env, self.precision)
            self.assignment, self.manager, self.routing = self.compute_route(
                self.data, env
            )
            vehicle = 0
            self.next = self.assignment.Value(
                self.routing.NextVar(self.routing.Start(vehicle))
            )
            while self.routing.IsEnd(self.next):
                vehicle += 1
                self.next = self.assignment.Value(
                    self.routing.NextVar(self.routing.Start(vehicle))
                )

        # return first step of solution as next action to execute
        result = np.zeros_like(obs["action_mask"])
        next_node = self.manager.IndexToNode(self.next)
        # note: the current OR tools policy never chooses noop. E.g., if all demands of available
        # customers are above the
        # vehicle capacity, then this method will choose going to the depot.
        if next_node == 0:
            # return to depot
            num_customers = len(obs["customer_demands"])
            result[num_customers] = 1
        else:
            id = self.data["ids"][next_node]
            result[np.where(obs["customer_ids"] == id)[0][0]] = 1
            self.next = self.assignment.Value(self.routing.NextVar(self.next))
        return result

    def compute_route(self, data, env):

        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(
            data["num_locations"], data["num_customers"], 0
        )

        # Create Routing Model
        routing = pywrapcp.RoutingModel(manager)

        # Define weight of each edge
        distance_evaluator_index = routing.RegisterTransitCallback(
            partial(create_distance_evaluator(data), manager)
        )
        routing.SetArcCostEvaluatorOfAllVehicles(distance_evaluator_index)

        # Add Capacity constraint
        demand_evaluator_index = routing.RegisterUnaryTransitCallback(
            partial(create_demand_evaluator(data), manager)
        )
        add_capacity_constraints(routing, manager, data, demand_evaluator_index)

        # Setting first solution heuristic (cheapest addition).
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        if self.timeout is not None:
            search_parameters.time_limit.seconds = self.timeout
        # Solve the problem.
        assignment = routing.SolveWithParameters(search_parameters)
        if not assignment:
            print("no solution found! falling back to distance_proportional_policy")
            return simple_baseline.distance_proportional_policy(obs, env)
        if self.verbose:
            print_solution(manager, routing, assignment, data, self.precision)

        return assignment, manager, routing


if __name__ == "__main__":
    fig1, ax1 = plt.subplots(1, 1)
    problem_generator = None
    run_benchmark = True
    plot_results = False
    seed = 456
    num_runs = 1
    if run_benchmark:
        CAPACITIES = {10: 20, 20: 30, 50: 40, 100: 50}
        # create random input based on benchmark distributions -
        depot_position_rv = stats.uniform(loc=0, scale=1)
        vehicle_position_rv = stats.uniform(loc=0, scale=1)
        customer_positions_rv = stats.uniform(loc=0, scale=1)
        customer_demands_rv = stats.randint(low=0, high=10)
        vrp_size = 20
        initial_vehicle_capacity = CAPACITIES[vrp_size]
        problem_generator = SampleStaticBenchmark(
            depot_position_rv=depot_position_rv,
            vehicle_position_rv=vehicle_position_rv,
            vehicle_capacity=initial_vehicle_capacity,
            vehicle_velocity=10,
            customer_positions_rv=customer_positions_rv,
            customer_demands_rv=customer_demands_rv,
            vrp_size=vrp_size,
            start_at_depot=True,
        )
    else:
        # run fixed problem for debugging
        customer_positions = np.array([[1, 0], [1, 1], [0, 1]])
        depot_position = np.array([0, 0])
        initial_vehicle_position = np.array([0, 0])
        initial_vehicle_capacity = 10
        vehicle_velocity = 10
        customer_demands = np.array([5, 5, 5])
        customer_times = np.array([0, 0, 0])
        vrp_size = 3
        problem_generator = FixedSample(
            depot_position,
            initial_vehicle_position,
            initial_vehicle_capacity,
            vehicle_velocity,
            customer_positions,
            customer_demands,
            customer_times,
        )

    sim = CVRPSimulation(max_customers=vrp_size, problem_generator=problem_generator)
    all_rewards = np.zeros(num_runs)
    for i_n in range(num_runs):
        sim.seed(seed)
        obs = sim.reset()
        done = False
        total_reward = 0
        i = 0
        vehicle_route = {
            0: {
                "x": [sim.current_state.current_vehicle_position[0]],
                "y": [sim.current_state.current_vehicle_position[1]],
            }
        }
        depot_position = sim.current_state.depot_position
        customer_positions = sim.current_state.customer_positions
        customer_demands = sim.current_state.customer_demands
        route_num = 0
        pol = ORToolsPolicy(verbose=True)
        while not done:
            action_probs = pol(obs, sim)
            act = np.random.choice(len(obs["action_mask"]), p=action_probs)
            customer_chosen = sim.get_customer_index(act)
            obs, reward, done, info = sim.step(act)
            vehicle_pos = sim.current_state.current_vehicle_position
            vehicle_route[route_num]["x"].append(vehicle_pos[0])
            vehicle_route[route_num]["y"].append(vehicle_pos[1])
            if act == action_probs.size - 2:
                # vehicle returning to depot therefore a new route is created
                route_num += 1
                vehicle_route[route_num] = {
                    "x": [vehicle_pos[0]],
                    "y": [vehicle_pos[1]],
                }
            print(
                f"i:{i}, t:{np.round(sim.current_time, 2)}, vehicle capacity:"
                f"{sim.current_state.current_vehicle_capacity},"
                f" customer chosen:{customer_chosen}, reward {reward}, done {done}"
            )
            total_reward += reward
            i += 1
        print(f"total reward is:{total_reward}")
        all_rewards[i_n] = total_reward
        if plot_results:
            plot_vehicle_routes(depot_position, customer_positions, customer_demands, vehicle_route, ax1)
    if plot_results:
        plt.show()
    print(f"mean reward is:{np.mean(all_rewards)}, std reward is:{np.std(all_rewards)}")
