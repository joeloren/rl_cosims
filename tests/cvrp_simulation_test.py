import numpy as np
import pytest

from cvrp_simulation.simulator import CVRPSimulation
from cvrp_simulation.scenario_generator import SpecificSample, SampleStaticBenchmark


def test_intermediate_states_match_hand_calculated_values():
    customer_positions = np.array([[1, 0], [1, 1], [0, 1]])
    depot_position = np.array([0, 0])
    initial_vehicle_position = np.array([0, 0])
    initial_vehicle_capacity = 30
    vehicle_velocity = 10
    customer_demands = np.array([5, 5, 5])
    customer_times = np.array([0, 0, 0])
    problem_generator = SpecificSample(depot_position, initial_vehicle_position, initial_vehicle_capacity,
                                       vehicle_velocity, customer_positions, customer_demands, customer_times)
    sim = CVRPSimulation(max_customers=3, problem_generator=problem_generator)
    sim.reset()
    reward, done = sim.step(0)
    assert reward == -1
    assert (sim.current_state.current_vehicle_position == np.asarray([1, 0])).all()
    assert sim.current_state.current_vehicle_capacity == 25
    assert len(sim.current_state.customer_positions) == 2
    assert sim.get_available_customers().size == 2
    assert not done
    reward, done = sim.step(0)
    assert reward == -1
    assert (sim.current_state.current_vehicle_position == np.asarray([1, 1])).all()
    assert sim.current_state.current_vehicle_capacity == 20
    assert len(sim.current_state.customer_positions) == 1
    assert sim.get_available_customers().size == 1
    assert not done
    reward, done = sim.step(0)
    assert reward == -1
    assert (sim.current_state.current_vehicle_position == np.asarray([0, 1])).all()
    assert sim.current_state.current_vehicle_capacity == 15
    assert len(sim.current_state.customer_positions) == 0
    assert sim.get_available_customers().size == 0
    assert done
    reward, done = sim.step(None)
    assert reward == -1
    assert (sim.current_state.current_vehicle_position == np.asarray([0, 0])).all()
    assert sim.current_state.current_vehicle_capacity == 30
    assert len(sim.current_state.customer_positions) == 0


def test_bounds_exceeded():
    customer_positions = np.array([[1, 0], [1, 1], [0, 1]])
    depot_position = np.array([0, 0])
    initial_vehicle_position = np.array([0, 0])
    initial_vehicle_capacity = 30
    vehicle_velocity = 10
    customer_demands = np.array([5, 5, 5])
    customer_times = np.array([0, 0, 0])
    problem_generator = SpecificSample(depot_position, initial_vehicle_position, initial_vehicle_capacity,
                                       vehicle_velocity, customer_positions, customer_demands, customer_times)
    sim = CVRPSimulation(max_customers=3, problem_generator=problem_generator)
    sim.reset()
    with pytest.raises(ValueError):
        sim.step(3)



