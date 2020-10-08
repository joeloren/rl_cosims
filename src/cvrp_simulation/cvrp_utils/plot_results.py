import numpy as np
from cvrp_simulation.simulation.scenario_generator import SampleStaticBenchmark
from cvrp_simulation.simulation.simulator import CVRPSimulation
from matplotlib import pyplot as plt
from scipy import stats


def discrete_cmap(n_colors, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n_colors))
    cmap_name = base.name + str(n_colors)
    return base.from_list(cmap_name, color_list, n_colors)


def plot_vehicle_routes(
    depot_position,
    customer_positions,
    customer_demands,
    veh_route,
    ax1=None,
    plot_demand=False,
):
    """
    this function plots the route of the vehicle and the customer positions
    :param depot_position: position of depot np.ndarray [x, y]
    :param customer_demands: the demands of each customer
    :param customer_positions: vector of customer positions [N, 2] where each row is customer i's
    [x, y]
    :param veh_route: dict of vehicle routes. keys are vehicle id's and in each value there is [
    x, y] of route
    :param ax1: axis for the plot
    :return:
    """
    if ax1 is None:
        _, ax1 = plt.subplots(1, 1)
    veh_used = [v for v in veh_route if veh_route[v] is not None]
    total_distance = 0
    cmap = discrete_cmap(len(veh_used) + 2, "nipy_spectral")
    ax1.scatter(depot_position[0], depot_position[1], marker="s", c="m")
    ax1.text(depot_position[0], depot_position[1], "depot")
    if plot_demand:
        for i_c, c_pos in enumerate(customer_positions):
            ax1.text(c_pos[0], c_pos[1], f"{i_c}[{customer_demands[i_c]:.2f}]")

    for veh_number in veh_used:
        xs, ys = veh_route[veh_number]["x"], veh_route[veh_number]["y"]
        for j in range(len(xs)):
            if j > 0:
                p0 = np.array([xs[j - 1], ys[j - 1]])
                p1 = np.array([xs[j], ys[j]])
                total_distance += np.linalg.norm(p0 - p1)
        ax1.plot(
            xs,
            ys,
            c=cmap(veh_number),
            marker="o",
            label=f"r:{veh_number}, demand:{veh_route[veh_number]['total_demand']}",
        )
        to_depot = np.array([[xs[-1], ys[-1]], [depot_position[0], depot_position[1]]])
        total_distance += np.linalg.norm(np.array([xs[-1], ys[-1]]) - depot_position)
        ax1.plot(to_depot[:, 0], to_depot[:, 1], c=cmap(veh_number))
        # ax1.legend()
        ax1.text(xs[0], ys[0], "vehicle_start")
    ax1.grid()
    print(f"total distance is:{total_distance}")
    return ax1


def plot_value_stats(values):
    """
    this function plots the values of each policy
    :param values: the values of running cvrp cvrp_simulation for each policy
    """
    _, ax = plt.subplots(1, 1)
    for policy_name in values.keys():
        data = np.array(values[policy_name])
        mean_value = np.mean(-data)
        std_value = np.std(-data)
        median_value = np.median(-data)
        ax.plot(
            range(data.shape[0]),
            -data,
            marker="o",
            label=f"{policy_name} - mean:{mean_value:.2f}, std:{std_value:.2f}, "
            f"median:{median_value:.2f}",
        )
    ax.set_title("Rewards for all policies")
    ax.grid()
    ax.legend()
    ax.set_xlabel("Run num")
    ax.set_ylabel("reward")
    # plt.show()
    plt.pause(0.02)
    plt.close()


def main():
    vehicle_velocity = 10
    # check benchmark generator
    depot_position_rv = stats.uniform(loc=0, scale=1)
    vehicle_position_rv = stats.uniform(loc=0, scale=1)
    customer_positions_rv = stats.uniform(loc=0, scale=1)
    customer_demands_rv = stats.uniform(loc=0, scale=10)
    vrp_size = 10
    initial_vehicle_capacity = 10 * vrp_size
    benchmark_generator = SampleStaticBenchmark(
        depot_position_rv=depot_position_rv,
        vehicle_position_rv=vehicle_position_rv,
        vehicle_capacity=initial_vehicle_capacity,
        vehicle_velocity=vehicle_velocity,
        customer_positions_rv=customer_positions_rv,
        customer_demands_rv=customer_demands_rv,
        vrp_size=vrp_size,
    )
    num_runs = 1
    seed = 50
    rand_reward = np.zeros(num_runs)
    print("--------------------------------------")
    print("benchmark cvrp_simulation testing:")
    sim = CVRPSimulation(max_customers=vrp_size, problem_generator=benchmark_generator)
    sim.seed(seed)
    for i in range(num_runs):
        _ = sim.reset()
        vehicle_route = {
            0: {
                "x": [sim.current_state.current_vehicle_position[0]],
                "y": [sim.current_state.current_vehicle_position[1]],
            }
        }
        # in this case each time we choose action 0 since the available actions change each time
        # the number of available customers changes
        tot_reward = 0
        _, ax1 = plt.subplots(1, 1)
        done = False
        while not done:
            obs, reward, done, _ = sim.step(0)
            vehicle_pos = sim.current_state.current_vehicle_position
            vehicle_route[0]["x"].append(vehicle_pos[0])
            vehicle_route[0]["y"].append(vehicle_pos[1])
            tot_reward += reward
        print(f"finished random run # {i}, total reward {tot_reward}")
        rand_reward[i] = tot_reward
        depot_position = sim.current_state.depot_position
        customer_positions = sim.current_state.customer_positions
        customer_demands = sim.current_state.customer_demands
        plot_vehicle_routes(
            depot_position, customer_positions, customer_demands, vehicle_route, ax1
        )
        plt.show()
    print(f"mean random reward is:{np.mean(rand_reward)}")


if __name__ == "__main__":
    main()
    print("done!")
