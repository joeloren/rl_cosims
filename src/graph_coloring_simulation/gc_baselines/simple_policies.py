# basic imports
from typing import Tuple
import numpy as np
import networkx as nx
# our imports
from src.graph_coloring_simulation.gc_simulation.simulator import Simulator


def random_policy(obs, env: Simulator) -> Tuple:
    """
    this function chooses the node and color randomaly from the colors and nodes allowed
    :param obs: Dict - the current observation of the current state in the env
    :param env: Simulator - the environment of graph coloring problem
    :return: action: tuple - the action chosen by policy in format (node_chosen, action_chosen)
    """
    # create a networkx graph of the current observation (this is used in order to quickly find the neighboring nodes)
    graph = nx.Graph()
    graph.add_nodes_from(obs["nodes_id"])
    graph.add_edges_from(obs["edge_indexes"])
    # filter only un-colored nodes from list of all nodes - nodes that are not colored have color=-1
    available_node_indexes = np.where(obs["node_colors"] == -1)[0]
    # chose node randomly using np.random choice
    node_chosen = np.random.choice(available_node_indexes, 1)[0]
    unavailable_colors = set()
    for nn in graph.neighbors(node_chosen):
        if obs["node_colors"][nn] != -1:
            unavailable_colors.add(obs["node_colors"][nn])
    # filter out of the used colors the colors that are not available
    available_colors = obs["used_colors"] - unavailable_colors
    # add new color option (new color is either the next number after the maximum color id already in graph or 0)
    new_color = np.max(list(obs["used_colors"])) + 1 if len(obs["used_colors"]) > 0 else 0
    available_colors.add(new_color)
    # choose color from list of available colors
    color_chosen = np.random.choice(list(available_colors), 1)[0]
    return node_chosen, color_chosen


def random_policy_without_newcolor(obs, env: Simulator) -> Tuple:
    """
    this function chooses the node and color randomaly from the colors and nodes allowed
    :param obs: Dict - the current observation of the current state in the env
    :param env: Simulator - the environment of graph coloring problem
    :return: action: tuple - the action chosen by policy in format (node_chosen, action_chosen)
    """
    # create a networkx graph of the current observation (this is used in order to quickly find the neighboring nodes)
    graph = nx.Graph()
    graph.add_nodes_from(obs["nodes_id"])
    graph.add_edges_from(obs["edge_indexes"])
    # filter only un-colored nodes from list of all nodes - nodes that are not colored have color=-1
    available_node_indexes = np.where(obs["node_colors"] == -1)[0]
    # chose node randomly using np.random choice
    node_chosen = np.random.choice(available_node_indexes, 1)[0]
    unavailable_colors = set()
    for nn in graph.neighbors(node_chosen):
        if obs["node_colors"][nn] != -1:
            unavailable_colors.add(obs["node_colors"][nn])
    # filter out of the used colors the colors that are not available
    available_colors = obs["used_colors"] - unavailable_colors
    # add new color option (new color is either the next number after the maximum color id already in graph or 0)
    new_color = np.max(list(obs["used_colors"])) + 1 if len(obs["used_colors"]) > 0 else 0
    if len(available_colors) == 0:
        available_colors.add(new_color)
    # choose color from list of available colors
    color_chosen = np.random.choice(list(available_colors), 1)[0]
    return node_chosen, color_chosen



