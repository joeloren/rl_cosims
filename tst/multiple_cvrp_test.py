import numpy as np
import pytest

from src.envs.cvrp.cvrp_experimentation.problems import create_fixed_static_problem
from src.envs.cvrp.cvrp_simulation.multiple_vehicle_cvrp import CUSTOMER_STATUS, VEHICLE_STATUS


def test_fixed_problem_one_vehicle_simple_problem():
    # this test is the most basic test, to make sure the simulation runs as expected with one vehicle, 2 customers
    # when vehicle can handle all customers without returning to the depot
    customer_positions = [[0, 5], [10, 10]]
    customer_demands = [10, 10]
    customer_times = [0, 0]
    vehicle_capacities = [20]
    vehicle_positions = [[0, 0]]
    depot_position = [0, 0]
    sim = create_fixed_static_problem(customer_positions=customer_positions,
                                      customer_demands=customer_demands,
                                      customer_times=customer_times,
                                      initial_vehicle_capacities=vehicle_capacities,
                                      initial_vehicle_positions=vehicle_positions,
                                      depot_position=depot_position, vehicle_velocity=1)
    obs = sim.reset()
    # make sure simulation start with time = 0
    assert sim.env.now == 0
    # make sure all customers are considered opened
    assert np.all(sim.current_state.customer_status == CUSTOMER_STATUS['opened'])
    # make sure vehicle and customers are not in illegal actions
    assert np.all(obs['illegal_actions'][0, 0:2] == 0)
    # make sure depot and noop are not legal actions (since vehicle is at depot and noop is not allowed)
    assert np.all(obs['illegal_actions'][0, 2:] == 1)
    assert np.all(obs['current_vehicle_positions'] == obs['depot_position'])
    assert obs['current_vehicle_capacities'] == obs['max_vehicle_capacity']
    assert sim.current_state.vehicle_status == VEHICLE_STATUS['available']
    obs, reward, is_done, info = sim.step((0, 0))
    assert sim.env.now == 5
    # make sure customers index is saved in vehicles full path
    assert sim.current_state.vehicle_full_path[0][0] == 0
    # next state is when vehicle is available - make sure the vehicle is in the correct state
    assert sim.current_state.vehicle_status[0] == VEHICLE_STATUS['available']
    # make sure customer 0 is in state "reached"
    assert sim.current_state.customer_status[0] == CUSTOMER_STATUS['visited']
    # make sure customer 1 is still in state waiting"
    assert sim.current_state.customer_status[1] == CUSTOMER_STATUS['opened']
    # make sure reward is the distance passed between vehicle position and customer 0
    assert reward == 5
    # make sure choosing customer 0 is not legal any more
    assert obs['illegal_actions'][0, 0] == 1
    # make sure vehicle capacity is now 10
    assert obs['current_vehicle_capacities'] == 10
    # make sure choosing customer 0 again raises a value assert
    with pytest.raises(ValueError):
        _, reward, is_done, info = sim.step((0, 0))
    # choose customer 1
    obs, reward, is_done, info = sim.step((0, 1))
    # make sure is_done is True
    assert is_done is True


def test_fixed_problem_one_vehicle_return_to_depot():
    # this test is the most basic test, to make sure the simulation runs as expected with one vehicle, 2 customers
    # when vehicle must return to depot between customers
    customer_positions = [[0, 5], [10, 10]]
    customer_demands = [10, 10]
    customer_times = [0, 0]
    vehicle_capacities = [15]
    vehicle_positions = [[0, 0]]
    depot_position = [0, 0]
    sim = create_fixed_static_problem(customer_positions=customer_positions,
                                      customer_demands=customer_demands,
                                      customer_times=customer_times,
                                      initial_vehicle_capacities=vehicle_capacities,
                                      initial_vehicle_positions=vehicle_positions,
                                      depot_position=depot_position, vehicle_velocity=1)
    sim.reset()
    # make sure simulation start with time = 0
    assert sim.env.now == 0
    # make sure all customers are considered opened
    assert np.all(sim.current_state.customer_status == CUSTOMER_STATUS['opened'])
    obs, reward, is_done, info = sim.step((0, 0))
    # make sure only depot is possible in illegal actions
    assert obs['illegal_actions'][0, 2] == 0
    assert np.all(obs['illegal_actions'][0, [0, 1, 3]] == 1)
    # make sure choosing customer 1 raises an error since vehicle capacity < customer 1 demand
    with pytest.raises(ValueError):
        _, reward, is_done, info = sim.step((0, 1))
    # chose depot
    obs, reward, is_done, info = sim.step((0, -1))
    # make sure vehicle capacity returned to original value
    assert obs['current_vehicle_capacities'][0] == vehicle_capacities[0]
    # make sure customer 1 is still in state 'opened'
    assert sim.current_state.customer_status[1] == CUSTOMER_STATUS['opened']
    # make sure time is now ~7
    assert sim.env.now == 10
    # make sure reward is the distance between customer 0 and depot
    assert reward == 5
    # make sure total distance vehicle 0 passed is depot -> customer 0 -> depot
    assert sim.current_state.vehicle_full_distance[0] == 10
    assert sim.current_state.vehicle_full_path[0] == [0, -1]
    # make sure choosing customer 0 again raises a value assert
    with pytest.raises(ValueError):
        _, reward, is_done, info = sim.step((0, 0))
    # choose customer 1
    obs, reward, is_done, info = sim.step((0, 1))
    # make sure is_done is True
    assert is_done is True


def test_fixed_problem_two_vehicle():
    num_customers = 3
    customer_positions = [[0., 5.], [10., 10.], [5., 5.]]
    customer_demands = [10, 10, 10]
    customer_times = [0, 0, 0]
    vehicle_capacities = [[20], [20]]
    vehicle_positions = [[0., 0.], [0., 0.]]
    depot_position = [0., 0.]
    sim = create_fixed_static_problem(customer_positions=customer_positions,
                                      customer_demands=customer_demands,
                                      customer_times=customer_times,
                                      initial_vehicle_capacities=vehicle_capacities,
                                      initial_vehicle_positions=vehicle_positions,
                                      depot_position=depot_position, vehicle_velocity=1)
    obs = sim.reset()
    # make sure simulation start with time = 0
    assert sim.env.now == 0
    # make sure all customers are considered opened
    assert np.all(sim.current_state.customer_status == CUSTOMER_STATUS['opened'])
    # make sure vehicle and customers are not in illegal actions
    assert np.all(obs['illegal_actions'][0, 0:num_customers] == 0)
    # make sure depot and noop are not legal actions (since vehicle is at depot and noop is not allowed)
    assert np.all(obs['illegal_actions'][0, num_customers:] == 1)
    assert np.all(obs['current_vehicle_positions'] == obs['depot_position'])
    assert np.all(obs['current_vehicle_capacities'] == obs['max_vehicle_capacity'])
    assert np.all(sim.current_state.vehicle_status == VEHICLE_STATUS['available'])
    obs, reward, is_done, info = sim.step((0, 0))
    # make sure simulation stops at time 0 since there is another vehicle available and customer 1 can be picked up
    assert sim.env.now == 0
    # make sure customers index is saved in vehicles full path
    assert sim.current_state.vehicle_full_path[0][0] == 0
    # next state is when vehicle 1 is available and vehicle 0 is busy - make sure the vehicle is in the correct state
    assert sim.current_state.vehicle_status[0] == VEHICLE_STATUS['busy']
    assert sim.current_state.vehicle_status[1] == VEHICLE_STATUS['available']
    # make sure customer 0 is in state "chosen"
    assert sim.current_state.customer_status[0] == CUSTOMER_STATUS['chosen']
    # make sure customer 1 is still in state waiting"
    assert sim.current_state.customer_status[1] == CUSTOMER_STATUS['opened']
    # make sure reward is 0 since no movement has actually been done
    assert reward == 0
    # make sure choosing customer 0 is not legal any more
    assert obs['illegal_actions'][0, 0] == 1
    # make sure vehicle capacity is still 20 since it has not moved yet
    assert np.all(obs['current_vehicle_capacities'] == 20)
    # choose customer 1 with vehicle 1
    obs, reward, is_done, info = sim.step((1, 1))
    # make sure simulation stops at the time when customer 0 is reached
    assert sim.env.now == 5
    # make sure customer 0 is in state "visited"
    assert sim.current_state.customer_status[0] == CUSTOMER_STATUS["visited"]
    # make sure customer 1 is in state "chosen"
    assert sim.current_state.customer_status[1] == CUSTOMER_STATUS["chosen"]
    # make sure vehicle 0 is available
    assert sim.current_state.vehicle_status[0] == VEHICLE_STATUS['available']
    # make sure vehicle 1 is busy
    assert sim.current_state.vehicle_status[1] == VEHICLE_STATUS['busy']
    # make sure vehicle 0 reached customer 0
    assert np.all(obs['current_vehicle_positions'][0, :] == obs["customer_positions"][0, :])
    # make sure reward is the full distance vehicles passed (both vehicles moved 5 m since both have same velocity)
    assert reward == 10
    # choose customer 2 with vehicle 0
    obs, reward, is_done, info = sim.step((0, 2))
    # make sure simulation returns that it is done
    assert is_done is True
    # make sure all customers are visited
    assert np.all(sim.current_state.customer_status == CUSTOMER_STATUS['visited'])
    # make sure vehicles don't have any customers assigned to them
    assert np.all(obs['current_vehicle_customer'] == None)


def test_fixed_problem_three_vehicle_online_customers():
    """
    this is the most basic test for the online problem.
    here we test the following:
    1. the simulation knows to stop when a new customer appears (2 vehicle are busy and 1 is available)
    2. the simulation knows to calculate the other customers properly
    3. once all customers are chosen the simulation runs until the end since there are no new customers
    :return:
    """
    customer_positions = [[0., 5.], [10., 10.], [5., 5.]]
    customer_demands = [10, 10, 10]
    customer_times = [0, 0, 2]
    vehicle_capacities = [[20], [20], [20]]
    vehicle_positions = [[0., 0.], [0., 0.], [0., 0.]]
    depot_position = [0., 0.]
    sim = create_fixed_static_problem(customer_positions=customer_positions,
                                      customer_demands=customer_demands,
                                      customer_times=customer_times,
                                      initial_vehicle_capacities=vehicle_capacities,
                                      initial_vehicle_positions=vehicle_positions,
                                      depot_position=depot_position, vehicle_velocity=1)
    obs = sim.reset()
    # make sure simulation start with time = 0
    assert sim.env.now == 0
    # make sure customer 2 is idle
    assert sim.current_state.customer_status[2] == CUSTOMER_STATUS['idle']
    # make sure customer 2 is illegal for all vehicles
    assert np.all(obs['illegal_actions'][:, 2] == 1)
    _, reward, is_done, info = sim.step((0, 0))
    # choose customer 1 with vehicle 1
    _, reward, is_done, info = sim.step((1, 1))
    # make sure simulation stops when customer 2 opens
    assert sim.env.now == 2
    # make sure customer 0 and 1 are in state "chosen"
    assert np.all(sim.current_state.customer_status[[0, 1]] == CUSTOMER_STATUS["chosen"])
    # make sure customer 2 is in state "opened"
    assert sim.current_state.customer_status[2] == CUSTOMER_STATUS["opened"]
    # make sure vehicle 2 is available
    assert sim.current_state.vehicle_status[2] == VEHICLE_STATUS['available']
    # make sure vehicle 0 and 1 are busy
    assert np.all(sim.current_state.vehicle_status[[0, 1]] == VEHICLE_STATUS['busy'])
    # choose customer 2 with vehicle 2
    obs, reward, is_done, info = sim.step((2, 2))
    # make sure simulation returns that it is done
    assert is_done is True
    # make sure all customers are visited
    assert np.all(sim.current_state.customer_status == CUSTOMER_STATUS['visited'])
    # make sure vehicles don't have any customers assigned to them
    assert np.all(obs['current_vehicle_customer'] == None)


def test_fixed_problem_two_vehicle_online_customers():
    customer_positions = [[0., 5.], [10., 10.], [5., 5.]]
    customer_demands = [10, 10, 10]
    customer_times = [0, 0, 2]
    vehicle_capacities = [[20], [20]]
    vehicle_positions = [[0., 0.], [0., 0.]]
    depot_position = [0., 0.]
    sim = create_fixed_static_problem(customer_positions=customer_positions,
                                      customer_demands=customer_demands,
                                      customer_times=customer_times,
                                      initial_vehicle_capacities=vehicle_capacities,
                                      initial_vehicle_positions=vehicle_positions,
                                      depot_position=depot_position, vehicle_velocity=1)
    sim.reset()
    # make sure simulation start with time = 0
    assert sim.env.now == 0
    # choose customer 0 for vehicle 0
    _, reward, is_done, info = sim.step((0, 0))
    # choose customer 1 with vehicle 1
    _, reward, is_done, info = sim.step((1, 1))
    # make sure simulation stops when the first vehicle reaches its customer
    # meanwhile the simulation should have opened customer 2
    assert sim.env.now == 5
    # make sure customer 2 is in state "opened"
    assert sim.current_state.customer_status[2] == CUSTOMER_STATUS["opened"]
    # make sure vehicle 0 is available
    assert sim.current_state.vehicle_status[0] == VEHICLE_STATUS['available']
