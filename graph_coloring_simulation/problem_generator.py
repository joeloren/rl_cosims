from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
import networkx as nx


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

    G = nx.fast_gnp_random_graph(number_of_nodes, probability_of_edge, seed=None, directed=False)
    edges = []
    for i in range(number_of_nodes):
        temp1 = G.adj[i]
        edges.append(list(G.adj[i].keys()))
    return G.number_of_edges(), edges



