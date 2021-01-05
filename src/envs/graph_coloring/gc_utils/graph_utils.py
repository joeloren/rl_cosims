from collections import defaultdict
from copy import deepcopy
from typing import Dict, List

import numpy as np
import networkx as nx


def create_graph_from_observation(obs: Dict, with_attributes: bool = True) -> nx.Graph:
    """
    Create networkx graph from observation dictionary
    this creates a graph similar to the graph in the Env State
    :param obs: Dict of observation
    :param with_attributes: if True add attributes to graph
    :return: graph_nx: graph as networkx
    """
    # get relevant data from observation
    graph_nodes = obs['nodes_id'].tolist()
    graph_node_positions = obs["node_positions"]
    graph_colors = obs['node_colors']
    graph_edges = obs['edge_indexes']
    # create graph and add nodes and edges
    graph_nx = nx.DiGraph()
    graph_nx.add_nodes_from(graph_nodes)
    graph_nx.add_edges_from(graph_edges)
    # add attributes to graph
    if with_attributes:
        att_nodes = {i: {'color': c, 'start_time': 0, 'pos': p, 'indicator': 0} for i, c, p in
               zip(graph_nodes, graph_colors, graph_node_positions)}
        att_edges = {e: {'indicator': 0} for e in graph_nx.edges()}
        nx.set_node_attributes(graph_nx, att_nodes)
        nx.set_edge_attributes(graph_nx, att_edges)
    return graph_nx


def add_color_nodes_to_graph(obs: Dict, with_attributes: bool = True) -> nx.Graph:
    """
    Create graph from observation and add color nodes to graph. the new graph will include n+m+1 nodes
    where n - number of original nodes, m - number of used colors.
    there will be edges between all the un-colored nodes and colored nodes only if color is feasible on that node
    :param obs: Dict of simulation observation
    :param with_attributes: if True add attributes to graph
    :return: graph_nx: new graph of current state and color nodes
    """
    graph_nx = create_graph_from_observation(obs, with_attributes)
    num_original_nodes = len(obs['nodes_id'].tolist())
    extra_nodes = [i for i in range(num_original_nodes, num_original_nodes + len(obs["used_colors"]) + 1)]
    # create node position
    delta_pos = 2 / (len(obs["used_colors"]) + 1)
    extra_nodes_positions = [np.array([2, -1 + delta_pos * (1 + i)]) for i in range(len(obs["used_colors"]) + 1)]
    # add color feature
    extra_nodes_colors = deepcopy(obs["used_colors"])
    if len(obs["used_colors"]):
        new_color_index = 1 + max(obs["used_colors"])
    else:
        new_color_index = 0
    extra_nodes_colors.add(new_color_index)
    # add edges based on color constraints
    extra_edge_indexes = []
    extra_edge_att = {}
    for n in graph_nx.nodes():
        if graph_nx.nodes('color')[n] == -1:
            allowed_colors = deepcopy(extra_nodes_colors)
            for n_neighbor in graph_nx.neighbors(n):
                neighbor_color = graph_nx.nodes('color')[n_neighbor]
                if neighbor_color != -1 and neighbor_color in allowed_colors:
                    allowed_colors.remove(neighbor_color)
            for c in allowed_colors:
                extra_edge_indexes.append((n, c + num_original_nodes))
                extra_edge_att[(n, c + num_original_nodes)] = {'indicator': 1}
    graph_nx.add_nodes_from(extra_nodes)
    graph_nx.add_edges_from(extra_edge_indexes)
    if with_attributes:
        att = {i: {'color': c, 'start_time': 0, 'pos': p, 'indicator': 1} for i, c, p in
               zip(extra_nodes, extra_nodes_colors, extra_nodes_positions)}
        nx.set_node_attributes(graph_nx, att)
        nx.set_edge_attributes(graph_nx, extra_edge_att)
    return graph_nx


def destroy_graph_solution(nodes_to_destroy: List[int], graph: nx.Graph) -> nx.Graph:
    """
    this method destroys part of a full solution of a graph based on the list of nodes given
    """
    for n in nodes_to_destroy:
        graph.nodes[n]['color'] = -1
    return graph


def create_subproblem_from_partial_solution(graph: nx.Graph):
    unique_colors = set([c for i, c in graph.nodes('color') if c != -1])
    max_color_index = np.max(unique_colors)
    uncolored_nodes = [i_n for i_n, c in graph.nodes('color') if c == -1]
    node_constraints = defaultdict(set)
    for n in uncolored_nodes:
        for neighbor_node in graph.neighbors(n):
            neighbor_color = graph.nodes[neighbor_node]['color']
            if neighbor_color != -1:
                node_constraints[n].add(neighbor_color)
    node_att = {k: {'forbidden_colors': v} for k, v in node_constraints.items()}
    # extract sub graph
    nx.set_node_attributes(graph, node_att)
    subgraph = graph.subgraph(nodes=uncolored_nodes).copy()
    return subgraph

