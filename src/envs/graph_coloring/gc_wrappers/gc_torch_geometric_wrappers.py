# basic imports
from typing import Dict
# mathematical imports
import numpy as np
# nn imports
import torch_geometric as tg
from gym import Wrapper
# our imports
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
    @staticmethod
    def obs_to_graph_dict(obs) -> tg.data.Data:
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
        for e, f in graph_nx.edges(data=True):
            edge_features.append(f['indicator'])
        node_features_array = np.vstack(node_features)
        edge_feature_array = np.vstack(edge_features)
        graph_tg = tg.utils.from_networkx(graph_nx)
        graph_tg.x = node_features_array
        graph_tg.edge_attr = edge_feature_array
        # illegal actions are all edges that have an indicator 0 which means they are real edges in the graph and not
        # action edges (for action edges indicator = 1)
        illegal_actions = [e for e, f in graph_nx.edges(data=True) if f['indicator'] == 0]
        graph_tg.illegal_actions = illegal_actions
        return graph_tg

    def __init__(self, env):
        super().__init__(env)

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
        if reinforce_action == 0:
            # this means that the depot was chosen
            action = self.num_nodes + self.env.DEPOT_INDEX
        else:
            action = reinforce_action - 1
        next_state, reward, done, _ = self.env.step(action)
        obs_tg = self.obs_to_graph_dict(next_state)
        return obs_tg, reward, done, {}
