# basic imports
from abc import ABC, abstractmethod
from typing import List
from copy import deepcopy
import random
# mathematical imports
import numpy as np
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


class FixedGraphGenerator(ScenarioGenerator):
    """
    this class creates a problem generator that always returns the same initial graph
    """
    def __init__(self, nodes_ids: List, edge_indexes: List, *kwargs):
        """
        initialize the class with the variables that create the graph
        :param nodes_ids: nodes iid (this is used as the identifier for the node, each node has a unique id)
        :param edge_indexes: tuple (u, v) indicates an edge between node u and node v
        """
        self.nodes_id = nodes_ids
        self.edge_indexes = edge_indexes
        self.edge_indexes.append((j, i) for i, j in self.edge_indexes)
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes_id, color=-1, start_time=0)
        self.graph.add_edges_from(self.edge_indexes)
        pos = nx.spring_layout(self.graph)
        att = {i: {'pos': p} for i, p in pos.items()}
        nx.set_node_attributes(self.graph, att)
        # add nodes to graph with the features:
        #   - color : the color of the node (default is -1)
        #   - open_time: time when node starts to be visible (in offline problem all start_time is 0)
        #   - pos : position of nodes, used for drawing graph

    def seed(self, seed: int) -> None:
        pass

    def next(self, current_state: State) -> State:
        """
        this function returns the current state with the updated graph
        :param current_state: the current simulation state with the current graph and other important details
        :return: updated current state with new graph (added nodes and edges)
        """
        # in the fixed case, no new nodes are created,
        # in the online problem - when calling next, more nodes are added to the graph
        return current_state

    def reset(self) -> State:
        state = State(unique_colors=set(),
                      graph=deepcopy(self.graph),
                      num_colored_nodes=0,
                      nodes_order=[],
                      current_time=0)
        return state


class ERGraphGenerator(ScenarioGenerator):
    """
    this class creates a problem generator that returns Erdos-Renyi graph (there are N nodes and probability P for
    having an edge E(i, j) between N_i and N_j)
    """
    def __init__(self, num_nodes: int, prob_edge: float, is_online: bool = False, seed: int = 0):
        """
        initialize the class with the variables that create the graph
        :param num_nodes: int , number of initial nodes in the graph
        :param prob_edge: float, probability if there exists an edge E(i, j) between two nodes
        :param is_online: bool, define if problem is the online (True) or offline problem (False)
        """
        # save input to class -
        self.num_initial_nodes = num_nodes
        self.prob_edge = prob_edge
        self.is_online = is_online
        self.seed(seed)

    def create_new_graph(self):
        # create random graph -
        graph = nx.fast_gnp_random_graph(self.num_initial_nodes, self.prob_edge, directed=True)
        undirected_edges = [(j, i) for i, j in graph.edges()]
        graph.add_edges_from(undirected_edges)
        # add features to nodes in graph -
        #   - color : the color of the node (default is -1)
        #   - open_time: time when node starts to be visible (in offline problem all start_time is 0)
        #   - pos: position of node, used for visualization
        pos = nx.spring_layout(graph)
        att = {i: {'color': -1, 'start_time': 0, 'pos': p} for i, p in pos.items()}
        nx.set_node_attributes(graph, att)
        return graph

    def seed(self, seed: int) -> None:
        np.random.seed(seed)
        random.seed(seed)

    def next(self, current_state: State):
        """
        this function returns the current state with the updated graph
        :param current_state: the current cvrp state with the current graph and other important details
        :return: updated current state with new graph (added nodes and edges)
        """
        if self.is_online:
            num_node_to_add = len(current_state.graph.nodes())
            # add new node to graph with time = current_time and no color
            current_state.graph.add_node(num_node_to_add, color=-1, start_time=current_state.current_time,
                                         pos=np.array([np.random.uniform(-1, 1, 1), np.random.uniform(-1, 1, 1)]))
            # go over all other nodes and graph and see if edge needs to be added
            for n in current_state.graph.nodes():
                if np.random.random() > self.prob_edge:
                    current_state.graph.add_edge(num_node_to_add, n)
                    current_state.graph.add_edge(n, num_node_to_add)  # add edge in other direction
        return current_state

    def reset(self) -> State:
        state = State(unique_colors=set(),
                      graph=self.create_new_graph(),
                      num_colored_nodes=0,
                      nodes_order=[],
                      current_time=0)
        return state
