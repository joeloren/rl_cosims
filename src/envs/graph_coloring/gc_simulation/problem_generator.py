# basic imports
from abc import ABC, abstractmethod
from typing import List
from copy import deepcopy
# mathematical imports
import networkx as nx
# our imports
from src.envs.graph_coloring.gc_simulation.simulator import State


class ScenarioGenerator(ABC):

    @abstractmethod
    def seed(self, seed: int) -> None:
        """Sets the random seed for the arrival process. """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the arrival process"""
        raise NotImplementedError


# Generate random graph - in our case we care about adjacency list
def generate_random_graph(number_of_nodes, probability_of_edge):
    """
    Generates a random graph that has on average probability_of_edge * (number_of_nodes choose 2) edges
    :param number_of_nodes: number of graph nodes
    :type number_of_nodes: int
    :param probability_of_edge: Probability of an edge given any two different nodes
    :type probability_of_edge: float
    :return: (number of edges, adjacency list)
    :rtype: (int, list of lists)
    """

    g = nx.fast_gnp_random_graph(number_of_nodes, probability_of_edge, seed=None, directed=False)
    edges = []
    for i in range(number_of_nodes):
        temp1 = g.adj[i]
        edges.append(list(g.adj[i].keys()))
    return g.number_of_edges(), edges


class FixedGraphGenerator(ScenarioGenerator):
    """
    this class creates a problem generator that always returns the same initial graph
    """
    def __init__(self, nodes_ids: List, edge_indexes: List):
        """
        initialize the class with the variables that create the graph
        :param nodes_ids: nodes iid (this is used as the identifier for the node, each node has a unique id)
        :param edge_indexes: tuple (u, v) indicates an edge between node u and node v
        """
        self.nodes_id = nodes_ids
        self.edge_indexes = edge_indexes
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes_id, color=-1, start_time=0)
        self.graph.add_edges_from(self.edge_indexes)
        # add nodes to graph with the features:
        #   - color : the color of the node (default is -1)
        #   - open_time: time when node starts to be visible (in offline problem all start_time is 0)

    def seed(self, seed: int) -> None:
        pass

    def next(self, current_state: State) -> State:
        """
        this function returns the current state with the updated graph
        :param current_state: the current cvrp state with the current graph and other important details
        :return: updated current state with new graph (added nodes and edges)
        """
        # in the fixed case, no new nodes are created,
        # in the online problem - when calling next, more nodes are added to the graph
        pass

    def reset(self) -> State:
        state = State(unique_colors=set(),
                      graph=deepcopy(self.graph),
                      num_colored_nodes=0,
                      nodes_order=[])
        return state

