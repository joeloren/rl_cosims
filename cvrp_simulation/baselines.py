import numpy as np
from ortools.constraint_solver import pywrapcp

from cvrp_simulation.simulator import CVRPSimulation
from cvrp_simulation.scenario_generator import FixedSample



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

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

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

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
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
    customer_positions = np.array([[1, 0], [1, 1], [0, 1]])
    depot_position = np.array([0, 0])
    initial_vehicle_position = np.array([0, 0])
    initial_vehicle_capacity = 30
    vehicle_velocity = 10
    customer_demands = np.array([5, 5, 5])
    customer_times = np.array([0, 0, 0])
    problem_generator = FixedSample(depot_position, initial_vehicle_position, initial_vehicle_capacity,
                                    vehicle_velocity, customer_positions, customer_demands, customer_times)
    sim = CVRPSimulation(max_customers=3, problem_generator=problem_generator)
    obs = sim.reset()
    done = False
    total_reward = 0
    i = 0
    while not done:
        action_probs = ortools_policy(obs, sim, verbose=True)
        act = np.random.choice(int(np.sum(obs["action_mask"])), p=action_probs)
        customer_chosen = sim.get_customer_index(act)
        obs, reward, done, info = sim.step(act)
        print(f"i:{i}, t:{np.round(sim.current_time, 2)}, customer chosen:{customer_chosen}, reward {reward}, done {done}")
        total_reward += reward
        i += 1
    print(total_reward)
