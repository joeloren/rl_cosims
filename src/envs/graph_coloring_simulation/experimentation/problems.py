# basic imports
from typing import List

# mathematical imports
# our imports
from src.envs.graph_coloring_simulation import FixedGraphGenerator
from src.envs.graph_coloring_simulation import Simulator


def create_fixed_static_problem(nodes_ids: List,
                                edge_indexes: List[tuple]):
    """
    Creates a minimal instance with fixed parameters
    :return:
    """
    problem_generator = FixedGraphGenerator(nodes_ids=nodes_ids, edge_indexes=edge_indexes)
    sim = Simulator(num_max_nodes=len(nodes_ids), problem_generator=problem_generator)
    return sim

