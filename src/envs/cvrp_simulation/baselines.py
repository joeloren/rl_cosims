import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from matplotlib import pyplot as plt
from scipy import stats

from src.envs.cvrp_simulation import CVRPSimulation
from src.envs.cvrp_simulation import FixedSample, SampleStaticBenchmark
from src.envs.cvrp_simulation.plot_results import plot_vehicle_routes


def ortools_policy(obs, env, precision=1000, timelimit=10, verbose=False):
    # there are always number of customers + 2 nodes: all customer nodes, plus a current position and
    # a depot position. The latter two can correspond to the same position, but do not need to.
    num_customers = env.get_available_customers().size
    num_nodes = int(num_customers + 2)
    # args are (number of nodes, number of vehicles, index of start node, index of depot node)
    manager = pywrapcp.RoutingIndexManager(num_nodes, 1, [1], [0])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node_index = manager.IndexToNode(from_index)
        to_node_index = manager.IndexToNode(to_index)

        if from_node_index == 0:
            from_node = obs["depot_position"]
        elif from_node_index == 1:
            from_node = obs["current_vehicle_position"]
        else:
            from_node = obs["customer_positions"][from_node_index - 2, :]
        if to_node_index == 0:
            to_node = obs["depot_position"]
        if to_node_index == 1:
            to_node = obs["current_vehicle_position"]
        else:
            to_node = obs["customer_positions"][to_node_index - 2, :]
        distance = np.linalg.norm(from_node - to_node)
        return precision * distance

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000 * precision,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return obs['customer_demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [obs['current_vehicle_capacity']],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = timelimit
    solution = routing.SolveWithParameters(search_parameters)

    if verbose:
        index = routing.Start(0)
        plan_output = 'Route for vehicle {}:\n'.format(0)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            immediate_cost = routing.GetArcCostForVehicle(previous_index, index, 0)
            route_distance += immediate_cost
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance / precision)
        print(plan_output)

    num_actions = int(np.sum(obs["action_mask"]))
    result = np.zeros(num_actions)
    next_node = manager.IndexToNode(solution.Value(routing.NextVar(routing.Start(0))))
    if next_node == 0:
        result[num_customers] = 1  # depot is chosen
    else:
        result[next_node - 2] = 1  # customer is chosen
    return result


if __name__ == '__main__':
    fig1, ax1 = plt.subplots(1, 1)
    problem_generator = None
    run_benchmark = True
    if run_benchmark:
        # create random input based on benchmark distributions -
        depot_position_rv = stats.uniform(loc=0, scale=1)
        vehicle_position_rv = stats.uniform(loc=0, scale=1)
        customer_positions_rv = stats.uniform(loc=0, scale=1)
        customer_demands_rv = stats.randint(low=0, high=10)
        vrp_size = 10
        initial_vehicle_capacity = 50
        problem_generator = SampleStaticBenchmark(
            depot_position_rv=depot_position_rv,
            vehicle_position_rv=vehicle_position_rv,
            vehicle_capacity=initial_vehicle_capacity,
            vehicle_velocity=10,
            customer_positions_rv=customer_positions_rv,
            customer_demands_rv=customer_demands_rv,
            vrp_size=vrp_size)
        seed = 50
    else:
        # run fixed problem for debugging
        customer_positions = np.array([[1, 0], [1, 1], [0, 1]])
        depot_position = np.array([0, 0])
        initial_vehicle_position = np.array([0, 0])
        initial_vehicle_capacity = 30
        vehicle_velocity = 10
        customer_demands = np.array([5, 5, 5])
        customer_times = np.array([0, 0, 0])
        vrp_size = 3
        problem_generator = FixedSample(depot_position, initial_vehicle_position, initial_vehicle_capacity,
                                        vehicle_velocity, customer_positions, customer_demands, customer_times)

    sim = CVRPSimulation(max_customers=vrp_size, problem_generator=problem_generator)
    obs = sim.reset()
    done = False
    total_reward = 0
    i = 0
    vehicle_route = {0:
                         {'x': [sim.current_state.current_vehicle_position[0]],
                          'y': [sim.current_state.current_vehicle_position[1]]}
                     }
    depot_position = sim.current_state.depot_position
    customer_positions = sim.current_state.customer_positions
    while not done:
        action_probs = ortools_policy(obs, sim, verbose=True)
        act = np.random.choice(int(np.sum(obs["action_mask"])), p=action_probs)
        customer_chosen = sim.get_customer_index(act)
        obs, reward, done, info = sim.step(act)
        vehicle_pos = sim.current_state.current_vehicle_position
        vehicle_route[0]['x'].append(vehicle_pos[0])
        vehicle_route[0]['y'].append(vehicle_pos[1])
        print(
            f"i:{i}, t:{np.round(sim.current_time, 2)}, vehicle capacity:{sim.current_state.current_vehicle_capacity},"
            f" customer chosen:{customer_chosen}, reward {reward}, done {done}")
        total_reward += reward
        i += 1
    print(total_reward)
    plot_vehicle_routes(depot_position,customer_positions, vehicle_route, ax1)
    plt.show()
