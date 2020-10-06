# basic imports
from typing import List

# mathematical imports
import numpy as np
# our imports
from src.graph_coloring_simulation.problem_generator import FixedGraphGenerator
from src.graph_coloring_simulation.simulator import Simulator


def create_fixed_static_problem(nodes_ids: List,
                                edge_indexes: List[tuple]):
    """
    Creates a minimal instance with fixed parameters
    :return:
    """
    problem_generator = FixedGraphGenerator(nodes_ids=nodes_ids, edge_indexes=edge_indexes)
    sim = Simulator(num_max_nodes=len(nodes_ids), problem_generator=problem_generator)
    return sim

