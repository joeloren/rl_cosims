# basic imports
from typing import List

# mathematical imports
# our imports
from src.graph_coloring_simulation.gc_simulation.problem_generator import FixedGraphGenerator
from src.graph_coloring_simulation.gc_simulation.simulator import Simulator


def create_fixed_static_problem(nodes_ids: List, edge_indexes: List[tuple], random_seed=0) -> Simulator:
    """
    Creates a minimal instance with fixed parameters
    :return: cvrp_simulation with fixed problem generator
    """
    problem_generator = FixedGraphGenerator(nodes_ids=nodes_ids, edge_indexes=edge_indexes)
    sim = Simulator(num_max_nodes=len(nodes_ids), problem_generator=problem_generator)
    sim.seed(random_seed)
    return sim

