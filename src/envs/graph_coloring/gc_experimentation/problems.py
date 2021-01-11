# basic imports
from typing import List
import numpy as np
# mathematical imports
# our imports
from src.envs.graph_coloring.gc_simulation.problem_generator import FixedGraphGenerator, ERGraphGenerator
from src.envs.graph_coloring.gc_simulation.simulator import Simulator


def create_fixed_static_problem(nodes_ids: List, edge_indexes: List[tuple], forbidden_colors: np.ndarray,
                                random_seed=0) -> Simulator:
    """
    Creates a minimal instance with fixed parameters
    :param nodes_ids: the ids of the nodes in the graph
    :param edge_indexes: a list of edges. each edge is a tuple (i, j) from node i to node j
    :param forbidden_colors: a matrix of forbidden colors for each node (only used if the problem is created as a
    sub-problem of a larger problem. m[i, j] = True if color j is forbidden for node i
    :param random_seed: the random seed to be used (not relevant for this type of problems since the problem is static)
    :return: simulation with fixed problem generator
    """
    problem_generator = FixedGraphGenerator(nodes_ids=nodes_ids, edge_indexes=edge_indexes,
                                            forbidden_colors=forbidden_colors)
    sim = Simulator(num_max_nodes=len(nodes_ids), problem_generator=problem_generator,
                    max_time_steps=len(nodes_ids) + 1)
    sim.seed(random_seed)
    return sim


def create_er_random_graph_problem(num_new_nodes: int, num_initial_nodes: int, prob_edge: float,
                                   is_online: bool, random_seed: int = 0) -> Simulator:
    """
    this function creates a new problem with ER graph (can create online and offline problems, depending on the
    is_online variable. total run_time = num_initial_nodes + num_new_nodes
    :param num_new_nodes: number of nodes to add to the problem
    :param num_initial_nodes: number of initial nodes in the starting graph
    :param prob_edge: probability for having an edge between two nodes
    :param is_online: if online problem is_online=True otherwise it's False
    :param random_seed: random seed to use in the problem generator
    :return:
    """
    problem_generator = ERGraphGenerator(num_nodes=num_initial_nodes, prob_edge=prob_edge,
                                         is_online=is_online, seed=random_seed)
    if is_online:
        max_run_time = num_new_nodes
    else:
        max_run_time = num_initial_nodes
    sim = Simulator(num_max_nodes=num_initial_nodes + num_new_nodes, problem_generator=problem_generator,
                    max_time_steps=max_run_time)
    sim.seed(random_seed)
    return sim
