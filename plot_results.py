import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from cvrp_simulation.simulator import CVRPSimulation
from cvrp_simulation.scenario_generator import SampleStaticBenchmark


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plot_vehicle_routes(depot_position, customer_positions, customer_demands, veh_route, ax1):
    """
    this function plots the route of the vehicle and the customer positions
    :param depot_position: position of depot np.ndarray [x, y]
    :param customer_positions: vector of customer positions [N, 2] where each row is customer i's [x, y]
    :param veh_route: dict of vehicle routes. keys are vehicle id's and in each value there is [x, y] of route
    :param ax1: axis for the plot
    :return:
    """
    veh_used = [v for v in veh_route if veh_route[v] is not None]

    cmap = discrete_cmap(len(veh_used) + 2, 'nipy_spectral')
    ax1.scatter(depot_position[0], depot_position[1], marker='s', c='m')
    ax1.text(depot_position[0], depot_position[1], 'depot')
    for i_c, c_pos in enumerate(customer_positions):
        ax1.text(c_pos[0], c_pos[1], f"{i_c}[{customer_demands[i_c]:.2f}]")

    for veh_number in veh_used:
        xs, ys = veh_route[veh_number]['x'], veh_route[veh_number]['y']
        ax1.plot(xs, ys, c=cmap(0), marker='o')
        ax1.grid()


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
        vrp_size=vrp_size)
    num_runs = 1
    seed = 50
    rand_reward = np.zeros(num_runs)
    print("--------------------------------------")
    print("benchmark simulation testing:")
    sim = CVRPSimulation(max_customers=vrp_size, problem_generator=benchmark_generator)
    sim.seed(seed)
    for i in range(num_runs):
        obs = sim.reset()
        vehicle_route = {0:
                             {'x': [sim.current_state.current_vehicle_position[0]],
                              'y': [sim.current_state.current_vehicle_position[1]]}
                         }
        # in this case each time we choose action 0 since the available actions change each time
        # the number of available customers changes
        tot_reward = 0
        fig1, ax1 = plt.subplots(1, 1)
        done = False
        while not done:
            obs, reward, done, _ = sim.step(0)
            vehicle_pos = sim.current_state.current_vehicle_position
            vehicle_route[0]['x'].append(vehicle_pos[0])
            vehicle_route[0]['y'].append(vehicle_pos[1])
            tot_reward += reward
        print(f"finished random run # {i}, total reward {tot_reward}")
        rand_reward[i] = tot_reward
        depot_position = sim.current_state.depot_position
        customer_positions = sim.current_state.customer_positions
        customer_demands = sim.current_state.customer_demands
        plot_vehicle_routes(depot_position, customer_positions, customer_demands,  vehicle_route, ax1)
        plt.show()
    print(f"mean random reward is:{np.mean(rand_reward)}")


if __name__ == '__main__':
    main()
    print("done!")
