# basic imports
from copy import deepcopy
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
from src.envs.graph_coloring.gc_utils.graph_utils import create_graph_from_observation, add_color_nodes_to_graph


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
        # # create networkx graph of original nodes and color nodes
        # graph_nx = add_color_nodes_to_graph(obs, with_attributes=True)
        # node_features = []
        # for n, f in graph_nx.nodes(data=True):
        #     node_features.append(np.array([f['indicator'], f['color']]))
        # edge_features = []
        # for u, v, f in graph_nx.edges(data=True):
        #     edge_features.append(f['indicator'])
        # node_features_array = np.vstack(node_features)
        # edge_feature_array = np.vstack(edge_features)
        # # create directed graph from original graph
        # graph_tg = tg.utils.from_networkx(graph_nx)
        # # save node features as x tensor
        # graph_tg.x = torch.tensor(node_features_array, dtype=torch.float32)
        # # save edge features as tensor
        # graph_tg.edge_attr = torch.tensor(edge_feature_array, dtype=torch.float32, device=graph_tg.x.device)

        # get information about the real graph
        real_nodes_id = obs['nodes_id'].tolist()
        real_nodes_colors = obs["node_colors"]
        num_original_nodes = len(real_nodes_id)
        num_color_nodes = len(obs["used_colors"]) + 1
        color_nodes_id = [i for i in range(num_original_nodes, num_original_nodes + num_color_nodes)]
        # add color feature
        color_node_colors = deepcopy(obs["used_colors"])
        if len(obs["used_colors"]):
            new_color_index = 1 + max(obs["used_colors"])
        else:
            new_color_index = 0
        color_node_colors.add(new_color_index)
        color_edge_indexes = []
        # get allowed edges between color nodes and real graph nodes
        for i_n, n in enumerate(real_nodes_id):
            if real_nodes_colors[n] == -1:
                allowed_colors = deepcopy(color_node_colors)
                neighbor_colors = obs["color_adjacency_matrix"][n, obs["color_adjacency_matrix"][n, :] != -9999]
                allowed_colors = allowed_colors - set(neighbor_colors)
                for c in allowed_colors:
                    color_edge_indexes.append((n, int(c) + num_original_nodes))
        num_real_edges = len(obs["edge_indexes"])
        num_color_edges = len(color_edge_indexes)
        graph_edges = obs["edge_indexes"] + color_edge_indexes
        # create node features for real and color nodes f[indicator, color]
        real_node_features = np.zeros(shape=(num_original_nodes, 2))
        real_node_features[:, 1] = real_nodes_colors
        color_node_features = np.zeros(shape=(num_color_nodes, 2))
        color_node_features[:, 0] = 1  # indicator of color nodes is 1
        color_node_features[:, 1] = list(color_node_colors)
        node_features_array = np.concatenate([real_node_features, color_node_features])
        # create edge features for real graph edges and constraint edges f[indicator]
        real_edge_features = np.zeros(shape=(num_real_edges, 1))
        color_edge_features = np.ones(shape=(num_color_edges, 1))
        edge_features_array = np.concatenate([real_edge_features, color_edge_features])
        # convert all arrays to tensors
        node_features_tensor = torch.tensor(node_features_array, dtype=torch.float32)
        edge_features_tensor = torch.tensor(edge_features_array, dtype=torch.float32, device=node_features_tensor.device)
        edge_indexes_tensor = torch.tensor(graph_edges, dtype=torch.long,
                                           device=node_features_tensor.device).transpose(1, 0)
        # create tg graph from data
        graph_tg = tg.data.Data(x=node_features_tensor, edge_index=edge_indexes_tensor,
                                edge_attr=edge_features_tensor)
        # save illegal actions tensor
        # an edge that is part of the real graph is considered illegal, therefore if indicator = 0 the edge is illegal
        # (so we take the logical_not if the indicators
        graph_tg.illegal_actions = torch.logical_not(graph_tg.edge_attr.view(-1))
        graph_tg.u = torch.tensor([[0]], dtype=torch.float32, device=graph_tg.x.device)
        self.action_to_simulation_action_dict = {i: (u, v) for i, (u, v) in enumerate(graph_edges)}
        self.node_to_color_dict = {i: int(c) for i, c in enumerate(node_features_array[:, 1])}
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


class GraphOnlyColorsWrapper(Wrapper):
    """
    This class is used as a wrapper for running the gc_simulator with torch geometric.
    We assume that the state representation is the current graph with additional nodes.
    we add a new node for each color already used + a new color. We add edges between each not yet colored node
    in the
    original graph and the colors that are feasible (not used by neighboring nodes)
    in the end the new graph includes n+m+1 nodes (n original nodes + m colors used + 1 new color)
    and E + n*(m+1) edges at most (there can be less edges if there are already colored nodes that created
    constraints)
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
        color: if no color is used color = -1, otherwise color = id of the color used, for color nodes color =
        color id of the node.
        this graph is a bipartite graph and only includes the nodes that are not yet colored from the original graph
        (this is in order to reduce complexity of the problem)
        the tg graph has the following attributes:
        x : node features [n_nodes, n_features]
        u : global feature [0] for now
        edge_index: [2, n_edges] matrix where edge_index[i, j] is an edge between node i and node j
        edge_attribute: edge features [n_edges, n_features]
        illegal_actions: [n_edges, 1] , boolean vector where True: action is illegal , False: action is feasible
        """
        if sum(obs["node_colors"] != -1) == len(obs["nodes_id"]):
            graph_tg = tg.data.Data(x=torch.tensor([]), edge_index=torch.tensor([], dtype=torch.long),
                                    u=torch.tensor([]), edge_attr=torch.tensor([]))
        else:
            nodes_not_colored_id = np.where(obs['node_colors'] < 0)[0].tolist()
            num_nodes_not_colored = len(nodes_not_colored_id)
            num_color_nodes = len(obs["used_colors"]) + 1  # number of colors used + extra color
            # add color feature
            color_node_features = np.zeros(shape=(num_color_nodes, 2))  # we have 2 features , color and indicator
            color_node_features[:, 0] = 1  # indicators are 1 for colors
            real_node_features = np.zeros(shape=(num_nodes_not_colored, 2))
            real_node_features[:, 1] = -1  # color of all uncolored nodes in graph is -1
            colors_for_color_node = deepcopy(obs["used_colors"])
            if len(obs["used_colors"]):
                new_color_index = 1 + max(obs["used_colors"])
            else:
                new_color_index = 0
            colors_for_color_node.add(new_color_index)
            color_node_features[:, 1] = np.array(list(colors_for_color_node))  # first feature is the node color
            node_features_array = np.concatenate([real_node_features, color_node_features])  # full feature matrix
            edge_indexes = []
            for i_n, n in enumerate(nodes_not_colored_id):
                allowed_colors = deepcopy(colors_for_color_node)
                neighbor_colors = obs["color_adjacency_matrix"][n, obs["color_adjacency_matrix"][n, :] != -9999]
                allowed_colors = allowed_colors - set(neighbor_colors)
                for c in allowed_colors:
                    edge_indexes.append((i_n, int(c) + num_nodes_not_colored))
            # assert set(edge_indexes) == set(new_edge_indexes)
            undirected_edge_indexes = [(j, i) for i, j in edge_indexes]
            num_directed_edges = len(edge_indexes)
            edge_indexes = edge_indexes + undirected_edge_indexes
            # create directed graph from original graph
            node_features_tensor = torch.tensor(node_features_array, dtype=torch.float32)
            edge_index_tensor = torch.tensor(edge_indexes, dtype=torch.long, device=node_features_tensor.device)
            edge_index_tensor = edge_index_tensor.transpose(0, 1)
            graph_tg = tg.data.Data(x=node_features_tensor, edge_index=edge_index_tensor)
            edge_features = np.zeros(shape=(graph_tg.edge_index.shape[1], 1))
            # this is an indicator that these edges are illegal (no need to chose the same edge twice)
            edge_features[num_directed_edges:] = 1
            graph_tg.edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=graph_tg.x.device)
            # save illegal actions tensor
            # in this case all edges are legal so we remove only the ones that are in the other direction
            graph_tg.illegal_actions = graph_tg.edge_attr.view(-1).to(dtype=torch.bool)
            graph_tg.u = torch.tensor([[0]], dtype=torch.float32, device=graph_tg.x.device)
            self.action_to_simulation_action_dict = {i: (nodes_not_colored_id[u], v) for i, (u, v) in
                                                     enumerate(edge_indexes) if u < num_nodes_not_colored}
            self.node_to_color_dict = {i: c for i, c in enumerate(node_features_array[:, 1])}
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
        this function first translates the action chosen by the agent (in our case the action is an edge in the
        graph)
        from ppo action to env action, then steps through the env and translates the observation from dictionary to
        tg observation (tg graph)
        the reward is the negative distance the vehicle travelled between its current position and the chosen
        position
        :param reinforce_action: int - edge chosen by agent
        :return: tg_obs: tg.Data, reward: double, done: bool. info: Dict
        """
        (node_chosen, color_node_chosen) = self.action_to_simulation_action_dict[reinforce_action]
        color_chosen = self.node_to_color_dict[color_node_chosen]
        action = (node_chosen, color_chosen)
        next_state, reward, done, _ = self.env.step(action)
        obs_tg = self.obs_to_graph_dict(next_state)
        return obs_tg, reward, done, {}
