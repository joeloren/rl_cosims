import itertools
import math
import time

import numpy as np
from matplotlib import pyplot as plt
# our imports
from src.cvrp_simulation.cvrp_experimentation.problems import (create_uniform_dynamic_problem,
                                                               create_fixed_static_problem)
from src.cvrp_simulation.cvrp_utils.plot_results import plot_vehicle_routes



def angle(depot, p):
    diff = p - depot
    if diff[0] >= 0 and diff[1] >= 0:
        theta = math.atan(diff[1] / diff[0])
    elif diff[0] <= 0 and diff[1] >= 0:
        theta = math.pi + math.atan(diff[1] / diff[0])
    elif diff[0] <= 0 and diff[1] <= 0:
        theta = math.pi + math.atan(diff[1] / diff[0])
    elif diff[0] >= 0 and diff[1] <= 0:
        theta = 2 * math.pi + math.atan(diff[1] / diff[0])
    return theta


def compute_angles(data):
    # enter one line of data
    angle_lst = []
    for i in range(np.shape(data)[0] - 1):
        angle_lst.append([i, (angle(data[-1], data[i]))])
    return np.array(angle_lst)


def sort_data(data, rand):
    # sort the data in terms of the angles
    angle_lst = compute_angles(data[:, :2])
    angle_lst[:, 1] = (angle_lst[:, 1] + rand) % math.pi
    sorted_nodes = angle_lst[angle_lst[:, 1].argsort()][:, 0].astype(int)
    return data[sorted_nodes]


def group_customers(data, capacity, rand):
    # sort the data
    sorted_data = sort_data(data, rand)
    cust_dem = sorted_data[:, 2]
    load = capacity
    # start grouping
    tour_lst = []
    tour_dem_lst = []
    tour_lst.append([])
    tour_dem_lst.append([])
    group = 0
    for i in range(np.shape(sorted_data)[0]):
        if cust_dem[i] <= load:
            tour_lst[group].append(i)
            tour_dem_lst[group].append(cust_dem[i])
            load -= cust_dem[i]
        elif cust_dem[i] > load:
            # go to depot
            tour_lst.append([])
            tour_dem_lst.append([])
            group += 1
            load = capacity
            # new tour
            tour_lst[group].append(i)
            tour_dem_lst[group].append(cust_dem[i])
            load -= cust_dem[i]
    return tour_lst, tour_dem_lst, sorted_data


# solve tsp
# https://gist.github.com/mlalevic/6222750
def length(x, y):
    return np.linalg.norm(np.asarray(x) - np.asarray(y))


def solve_tsp_dynamic(points):
    # calc all lengths
    all_distances = [[length(x, y) for y in points] for x in points]
    # initial value - just distance from 0 to every other point + keep the track of edges
    A = {
        (frozenset([0, idx + 1]), idx + 1): (dist, [0, idx + 1])
        for idx, dist in enumerate(all_distances[0][1:])
    }
    cnt = len(points)
    for m in range(2, cnt):
        B = {}
        for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
            for j in S - {0}:
                # this will use 0th index of tuple for ordering, the same as if key=itemgetter(0)
                # used
                B[(S, j)] = min(
                    [
                        (
                            A[(S - {j}, k)][0] + all_distances[k][j],
                            A[(S - {j}, k)][1] + [j],
                        )
                        for k in S
                        if k != 0 and k != j
                    ]
                )
        A = B
    res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
    return np.asarray(res[0]), np.asarray(res[1])  # 0 for padding


def sweep(data, capacity, rand):
    # sorted_data[tour_lst[0]] gives the first group nodes and demands.
    tour_lst, tour_dem_lst, sorted_data = group_customers(data, capacity, rand)
    depot_loc = np.expand_dims(data[-1, :2], 0)  # [1 x 2]
    cust_loc = sorted_data[:, :2]  # [10 x 2]

    total_tour_len = 0
    total_tour_customer_indexs = []
    for k in range(len(tour_lst)):
        group_loc = np.concatenate([sorted_data[tour_lst[k]][:, :2], depot_loc], 0)
        group_customer_index = np.concatenate(
            [sorted_data[tour_lst[k]][:, 3], np.array([-2])], 0
        )
        (subtour_len, group_seq) = solve_tsp_dynamic(group_loc)
        subtour = group_loc[group_seq]
        subtour_index = group_customer_index[group_seq]
        total_tour_len += subtour_len
        total_tour_customer_indexs += list(subtour_index)

    return total_tour_len, total_tour_customer_indexs


def rand_sweep(data, capacity, NUM_RAND=1):
    # start from a random angle to group
    # compute for all problems
    tour_len_lst = []
    tour_customer_index_lst = []
    start_time = time.time()

    for i in range(np.shape(data)[0]):
        start_time_1 = time.time()
        best_tour_len = 1000000
        best_tour_indexs = []
        # randomize
        for l in range(NUM_RAND):
            rand = np.random.rand() * math.pi
            tour_len, tour_customer_indexs = sweep(data[i], capacity, rand)
            if tour_len < best_tour_len:
                best_tour_len = tour_len
                best_tour_indexs = tour_customer_indexs
        tour_len_lst.append(best_tour_len)
        tour_customer_index_lst.append(best_tour_indexs)
        print("time", time.time() - start_time_1)
    print("average tour len:", np.mean(tour_len_lst))
    print("std tour len:", np.sqrt(np.var(tour_len_lst)))
    print("per data time: {}".format((time.time() - start_time) / np.shape(data)[0]))
    return tour_len_lst, tour_customer_index_lst


class SweepPolicy:
    def __init__(self):
        self.current_solution = None
        self.next_node = None
        self.index_map = None
        self.next_action = None
        self.__name__ = "Sweep"

    def reset(self, obs):
        self.current_solution = None
        self.next_node = None
        self.index_map = None
        self.next_action = None

    def __call__(self, obs, env):
        probs = np.zeros_like(obs["action_mask"])
        if probs.size == 2:
            # there are no customers opened in problem therefore the algorithm chooses the noop
            # option
            # if noop is disabled chooses the depot
            if obs["action_mask"][-1]:
                probs[-1] = 1
            else:
                probs[-2] = 1
            return probs
        else:
            # find next customer from previously calculated route or from new route
            if (
                    self.current_solution is None
                    or (obs["current_vehicle_position"] == obs["depot_position"]).all()
            ):
                self.current_solution = self.compute_route(obs, env)
                self.next_action = 0
            probs = np.zeros_like(obs["action_mask"])
            next_node = self.current_solution[self.next_action]
            if next_node == -2:
                # return to depot
                num_customers = len(obs["customer_demands"])
                probs[num_customers] = 1
            else:
                customer_id = next_node
                action = np.where(obs["customer_ids"] == customer_id)[
                    0
                ]  # get index of current costumer chosen
                probs[action] = 1

            # otherwise there are no customers opened in the problem and
            # therefore the cvrp_simulation will choose the noop option
            self.next_action += 1
            return probs

    def compute_route(self, obs, env):
        depot_position = obs["depot_position"]
        customer_positions = obs["customer_positions"]
        num_customers = customer_positions.shape[0]
        customer_ids = obs["customer_ids"].reshape(num_customers, 1)
        customer_demands = obs["customer_demands"].reshape(num_customers, 1)
        vehicle_capacity = obs["current_vehicle_capacity"]
        customer_data = np.hstack([customer_positions, customer_demands, customer_ids])
        depot_data = np.hstack([depot_position, np.array([0]), np.array([-2])])
        data = np.vstack([customer_data, depot_data])
        current_tour_length, current_solution = sweep(data, vehicle_capacity, rand=10)
        # print(f"solution: {current_solution}")
        # print(f"cost: {current_tour_length}")
        return current_solution


def main():
    try:
        from trains import Task

        task = Task.init(project_name="test_axis", task_name="test sweep baseline")
    except ImportError:
        pass
    fig1, ax1 = plt.subplots(1, 1)
    sim = None
    run_benchmark = True
    plot_results = True
    print_debug = True
    seed = 5
    num_runs = 1
    if run_benchmark:
        sim = create_uniform_dynamic_problem(
            max_customer_times=100,
            size=20,
            vehicle_velocity=10,
            vehicle_capacity=30,
            max_demand=10,
            random_seed=1234,
            start_at_depot=True,
        )
    else:
        customer_positions = [[1, 0], [1, 1], [0, 1]]
        depot_position = [0, 0]
        initial_vehicle_position = [0, 0]
        initial_vehicle_capacity = 10
        vehicle_velocity = 10
        customer_demands = [5, 5, 5]
        customer_times = [0, 0, 0]
        sim = create_fixed_static_problem(
            customer_positions,
            depot_position,
            initial_vehicle_position,
            initial_vehicle_capacity,
            vehicle_velocity,
            customer_demands,
            customer_times,
        )
    sim.seed(seed)
    all_rewards = np.zeros(num_runs)
    for i_n in range(num_runs):
        obs = sim.reset()
        done = False
        total_reward = 0
        i = 0
        vehicle_route = {
            0: {
                "x": [sim.current_state.current_vehicle_position[0]],
                "y": [sim.current_state.current_vehicle_position[1]],
                "total_demand": 0,
            }
        }
        depot_position = sim.current_state.depot_position
        customer_positions = sim.current_state.customer_positions
        customer_demands = sim.current_state.customer_demands
        route_num = 0
        sweep_policy = SweepPolicy()
        while not done:
            action_probs = sweep_policy(obs, sim)
            act = np.random.choice(len(obs["action_mask"]), p=action_probs)
            customer_chosen = sim.get_customer_index(act)
            obs, reward, done, info = sim.step(act)
            vehicle_pos = sim.current_state.current_vehicle_position
            vehicle_route[route_num]["x"].append(vehicle_pos[0])
            vehicle_route[route_num]["y"].append(vehicle_pos[1])
            if act == action_probs.size - 2:
                print(
                    f"total demand in route-{route_num} is:"
                    f"{vehicle_route[route_num]['total_demand']}"
                )
                # vehicle returning to depot therefore a new route is created
                route_num += 1
                vehicle_route[route_num] = {
                    "x": [vehicle_pos[0]],
                    "y": [vehicle_pos[1]],
                    "total_demand": 0,
                }
            else:
                vehicle_route[route_num][
                    "total_demand"
                ] += sim.current_state.customer_demands[customer_chosen]
            # print(
            # f"i:{i}, t:{np.round(sim.current_time, 2)}, vehicle capacity:{
            # sim.current_state.current_vehicle_capacity},"
            # f" customer chosen:{customer_chosen}, reward {reward}, done {done}")
            total_reward += reward
            i += 1
        print(
            f"total demand in route-{route_num} is:{vehicle_route[route_num]['total_demand']}"
        )
        if (
                vehicle_route[route_num]["total_demand"]
                > sim.initial_state.current_vehicle_capacity
        ):
            assert Exception(
                f"route {route_num} : customer demands: {vehicle_route[route_num]['total_demand']},"
                f" this exceeds the vehicle capacity!!!"
            )
        print(f"total reward is:{total_reward}")
        all_rewards[i_n] = total_reward
        if plot_results:
            plot_vehicle_routes(
                depot_position, customer_positions, customer_demands, vehicle_route, ax1
            )
            ax1.set_title("sweep policy results")
    if plot_results:
        plt.pause(0.01)
        plt.close()
        # plt.show()

    print(f"mean reward is:{np.mean(all_rewards)}, std reward is:{np.std(all_rewards)}")


if __name__ == "__main__":
    main()
    print("done!")
