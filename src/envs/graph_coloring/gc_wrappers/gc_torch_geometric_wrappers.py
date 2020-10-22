# basic imports
from typing import Dict
# mathematical imports
import networkx as nx
import numpy as np
# nn imports
import torch
import torch_geometric as tg
from gym import Wrapper
# our imports
from src.envs.graph_coloring.gc_simulation.simulator import Simulator
from src.envs.graph_coloring.gc_utils.graph_utils import add_color_nodes_to_graph


class GraphWithColorsWrapper(Wrapper):
    """
    This class is used as a wrapper for running the gc_simulator with torch geometric.
    We assume that the state representation is the current graph with additional nodes.
    we add a new node for each color already used + a new color. We add edges between each not yet colored node in the
    original graph and the colors that are feasible (not used by neighboring nodes)
    in the end the new graph includes n+m+1 nodes (n original nodes + m colors used + 1 new color)
    and E + n*(m+1) edges at most (there can be less edges if there are already colored nodes that created constraints)
    """
    def __init__(self, env: Simulator):
        super().__init__(env)
        # dictionary between action id and (num edge) and nodes the edge connects
        self.action_to_simulation_action_dict = {}
        # dictionary between node id and node color (used for translating action to simulation action)
        self.node_to_color_dict = {}

    def obs_to_graph_dict(self, obs: Dict) -> tg.data.Data:
        """
        this function takes the observation and creates a graph including the following
        features: [indicator, color]
        indicator: 0: real node, 1: color node
        color: if no color is used color = -1, otherwise color = id of the color used, for color nodes color = color id
        of the node
        the tg graph has the following attributes:
        x : node features [n_nodes, n_features]
        u : global feature [0] for now
        edge_index: [2, n_edges] matrix where edge_index[i, j] is an edge between node i and node j
        edge_attribute: edge features [n_edges, n_features]
        illegal_actions: [n_edges, 1] , boolean vector where True: action is illegal , False: action is feasible
        """
        # create networkx graph of original nodes and color nodes
        graph_nx = add_color_nodes_to_graph(obs, with_attributes=True)
        node_features = []
        for n, f in graph_nx.nodes(data=True):
            node_features.append(np.array([f['indicator'], f['color']]))
        edge_features = []
        for u, v, f in graph_nx.edges(data=True):
            edge_features.append(f['indicator'])
        node_features_array = np.vstack(node_features)
        edge_feature_array = np.vstack(edge_features)
        # create directed graph from original graph
        graph_tg = tg.utils.from_networkx(graph_nx)
        # save node features as x tensor
        graph_tg.x = torch.tensor(node_features_array, dtype=torch.float32)
        # save edge features as tensor
        graph_tg.edge_attr = torch.tensor(edge_feature_array, dtype=torch.float32, device=graph_tg.x.device)
        # save illegal actions tensor
        # an edge that is part of the real graph is considered illegal, therefore if indicator = 0 the edge is illegal
        # (so we take the logical_not if the indicators
        graph_tg.illegal_actions = torch.logical_not(graph_tg.edge_attr.view(-1))
        graph_tg.u = torch.tensor([[0]], dtype=torch.float32, device=graph_tg.x.device)
        self.action_to_simulation_action_dict = {i: (u, v) for i, (u, v) in enumerate(graph_nx.edges())}
        self.node_to_color_dict = nx.get_node_attributes(graph_nx, 'color')
        return graph_tg

    def reset(self):
        """
        Reset the environment and return tg observation
        :return:
        """
        # reset env -
        obs = self.env.reset()
        # create tg observation from obs dictionary -
        obs_tg = self.observation(obs)
        return obs_tg

    def observation(self, obs: Dict) -> tg.data.Data:
        """
        Translate current observation into new tg Data tensor
        :param obs:
        :return:
        """
        # first run any other wrapper in env
        obs = self.env.observation(obs)
        # convert observation to tg graph
        obs_tg = self.obs_to_graph_dict(obs)
        return obs_tg

    def step(self, reinforce_action: int):
        """
        this function first translates the action chosen by the agent (in our case the action is an edge in the graph)
        from ppo action to env action, then steps through the env and translates the observation from dictionary to
        tg observation (tg graph)
        the reward is the negative distance the vehicle travelled between its current position and the chosen position
        :param reinforce_action: int - edge chosen by agent
        :return: tg_obs: tg.Data, reward: double, done: bool. info: Dict
        """
        (node_chosen, color_node_chosen) = self.action_to_simulation_action_dict[reinforce_action]
        color_chosen = self.node_to_color_dict[color_node_chosen]
        action = (node_chosen, color_chosen)
        next_state, reward, done, _ = self.env.step(action)
        obs_tg = self.obs_to_graph_dict(next_state)
        return obs_tg, reward, done, {}
