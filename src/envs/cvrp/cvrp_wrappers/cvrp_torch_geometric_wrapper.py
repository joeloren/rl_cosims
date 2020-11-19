import itertools
import scipy.stats as sps

import networkx as nx
import numpy as np
import torch
import torch_geometric as tg
from gym import Wrapper, ObservationWrapper


class GeometricWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.num_nodes = 0

    def reset(self):
        """
        this function resets the environment and the wrapper
        :return: tg_obs: tg.Data - graph with all features as a torch geometric graph
        """
        # reset env -
        obs = self.env.reset()
        # create tg observation from obs dictionary -
        tg_obs = self.observation(obs)
        # # add illegal actions to graph -
        # tg_obs.illegal_actions = torch.zeros(tg_obs.edge_index.shape[1], dtype=torch.bool,
        #                                      device=tg_obs.x.device)
        return tg_obs

    def step(self, reinforce_action):
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
        tg_obs = self.obs_to_graph(next_state)
        return tg_obs, reward, done, {}

    def observation(self, obs):
        obs = self.env.observation(obs)
        return self.obs_to_graph_dict(obs)

    def obs_to_graph_dict(self, obs) -> tg.data.Data:
        """
        this function takes the observation and creates a graph including the following
        features:
        (indicator, x, y, node_demand)
        indicator: 0: vehicles, 1: depot, 2: customers
        x, y: position of the node in grid (double, double)
        node_demand: the customer demand or current vehicle capacity depending on the type of node (the vehicle
        capacity is negative)
        """
        # # if creating bipartite graph with only vehicles and customers use the following code:
        # there are 4 features all together :
        customer_positions = obs['customer_positions']
        customer_demands = obs['customer_demands']
        vehicle_capacity = obs['current_vehicle_capacity']
        num_customers = customer_positions.shape[0]
        num_vehicles = 1
        num_depots = 1
        num_nodes = num_customers + num_vehicles + num_depots
        node_pos = np.vstack([obs['current_vehicle_position'],
                              obs['depot_position'],
                              customer_positions])
        # indicator is : 0: vehicles, 1: depot, 2: customers
        node_ind = np.vstack([np.ones(shape=(num_vehicles, 1)) * 0,
                              np.ones(shape=(num_depots, 1)) * 1,
                              np.ones(shape=(num_customers, 1)) * 2])
        node_demand = np.vstack([-vehicle_capacity,
                                 np.zeros(shape=(num_depots, 1)),
                                 customer_demands.reshape(-1, 1)])
        # features are : indicator, pos_x, pos_y, demand/capacity
        node_features = np.hstack([node_ind, node_pos, node_demand])
        # assume that all visited customers are masked and also that vehicle nodes are masked
        g = nx.DiGraph()
        g.add_nodes_from(range(num_nodes))
        # edged go from vehicles to all depots and customers (star kind of graph)
        g.add_edges_from([(i, j) for i, j in itertools.product(range(num_vehicles),
                                                               range(num_vehicles,
                                                                     num_depots + num_vehicles + num_customers))])
        # create graph as torch_geometric tensor
        g_tensor = tg.utils.from_networkx(G=g)
        g_tensor.x = torch.tensor(node_features, dtype=torch.float32)
        g_tensor.u = torch.tensor([[0]], dtype=torch.float32)
        g_tensor.edge_attr = torch.zeros(size=[g_tensor.edge_index.shape[1], 1], dtype=torch.float32)
        # action mask is True if action is possible, therefore we need to take the not of this list
        actions_mask = np.zeros(shape=[g_tensor.edge_index.shape[1]])
        actions_mask[0] = obs['action_mask'][self.env.DEPOT_INDEX]
        actions_mask[1:] = obs['action_mask'][:-2]
        g_tensor.illegal_actions = torch.tensor(np.logical_not(actions_mask), dtype=torch.bool,
                                                device=g_tensor.x.device)
        self.num_nodes = g_tensor.x.shape[0]
        return g_tensor


class GeometricBidirectionalWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.num_nodes = 0

    def reset(self):
        """
               this function resets the environment and the wrapper
               :return: tg_obs: tg.Data - graph with all features as a torch geometric graph
               """
        # reset env -
        obs = self.env.reset()
        # create tg observation from obs dictionary -
        tg_obs = self.observation(obs)
        return tg_obs

    def observation(self, obs):
        obs = self.env.observation(obs)
        return self.obs_to_graph_dict(obs)

    def obs_to_graph_dict(self, obs) -> tg.data.Data:
        """
        this function takes the observation and creates a graph including the following
        features:
        (indicator, x, y, node_demand)
        indicator: 0: vehicles, 1: depot, 2: customers
        x, y: position of the node in grid (double, double)
        node_demand: the customer demand or current vehicle capacity depending on the type of node (the vehicle
        capacity is negative)
        the vehicle node is connected to all other nodes (like a start) in both directions resulting in a bi-directional
        graph.
        """
        # # if creating bipartite graph with only vehicles and customers use the following code:
        # there are 4 features all together :
        customer_positions = obs['customer_positions']
        customer_demands = obs['customer_demands']
        vehicle_capacity = obs['current_vehicle_capacity']
        num_customers = customer_positions.shape[0]
        num_vehicles = 1
        num_depots = 1
        num_nodes = num_customers + num_vehicles + num_depots
        node_pos = np.vstack([obs['current_vehicle_position'],
                              obs['depot_position'],
                              customer_positions])
        # indicator is : 0: vehicles, 1: depot, 2: customers
        node_ind = np.vstack([np.ones(shape=(num_vehicles, 1)) * 0,
                              np.ones(shape=(num_depots, 1)) * 1,
                              np.ones(shape=(num_customers, 1)) * 2])
        node_demand = np.vstack([-vehicle_capacity,
                                 np.zeros(shape=(num_depots, 1)),
                                 customer_demands.reshape(-1, 1)])
        # features are : indicator, pos_x, pos_y, demand/capacity
        node_features = np.hstack([node_ind, node_pos, node_demand])
        # assume that all visited customers are masked and also that vehicle nodes are masked
        g = nx.DiGraph()
        g.add_nodes_from(range(num_nodes))
        # edged go from vehicles to all depots and customers (star kind of graph)
        edge_list = [(i, j) for i, j in itertools.product(range(num_vehicles),
                                                          range(num_vehicles,
                                                                num_depots + num_vehicles + num_customers))]
        # add edges in opposite direction in order to create a bi-directional graph
        edge_list.extend([(j, i) for i, j in edge_list])
        g.add_edges_from(edge_list)

        # create graph as torch_geometric tensor
        g_tensor = tg.utils.from_networkx(G=g)
        g_tensor.x = torch.tensor(node_features, dtype=torch.float32)
        g_tensor.u = torch.tensor([[0]], dtype=torch.float32)
        g_tensor.edge_attr = torch.ones(size=[g_tensor.edge_index.shape[1], 1], dtype=torch.float32)
        # get illegal actions from action_mask (action_mask is False if customers demand is beyond current vehicle
        # capacity or if customer is already visited)
        actions_mask = np.zeros(shape=[g_tensor.edge_index.shape[1]])
        actions_mask[0] = obs['action_mask'][self.env.DEPOT_INDEX]
        # all edges that are from customer/depot to vehicle are not valid
        actions_mask[1:num_customers + num_depots] = obs['action_mask'][:-2]
        # illegal actions are the not of the actions_mask (since actions_mask is True if action is valid and
        # illegal actions is True if action is not valid)
        illegal_actions = torch.logical_not(torch.tensor(actions_mask, device=g_tensor.x.device))
        g_tensor.illegal_actions = illegal_actions
        # update number of nodes in observation (used in step to translate action from agent to simulation)
        self.num_nodes = g_tensor.x.shape[0]
        return g_tensor

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
        tg_obs = self.obs_to_graph_dict(next_state)
        return tg_obs, reward, done, {}


class ObservationNormalizationWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.data_to_normalize = [
            "customer_positions",
            "customer_demands",
            "depot_position",
            "current_vehicle_position",
            "current_vehicle_capacity"
        ]

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def observation(self, obs):
        for k, v in obs.items():
            # Normalize all Box space observations
            if k in self.data_to_normalize:
                low_mean = np.mean(self.observation_space[k].low).item()
                high_mean = np.mean(self.observation_space[k].high).item()
                obs[k] = self.mean_std_transform(
                    v, low_mean, high_mean)

        return obs

    @staticmethod
    def mean_std_transform(value, lower, upper):
        mean, var = sps.uniform.stats(loc=lower, scale=upper - lower, moments="mv")
        std = np.sqrt(var) + 1e-8

        value = value - mean
        value = value / std

        return value

    @staticmethod
    def mean_std_inverse(value, lower, upper):
        mean, var = sps.uniform.stats(loc=lower, scale=upper - lower, moments="mv")
        std = np.sqrt(var) + 1e-8

        value = value * std
        value = value + mean

        return value


class GeometricFullyConnectedWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.num_nodes = 0
        self.num_customers = 0
        self.action_to_simulation_action_dict = {}

    def reset(self):
        """
        this function resets the environment and the wrapper
        :return: tg_obs: tg.Data - graph with all features as a torch geometric graph
        """
        # reset env -
        obs = self.env.reset()
        self.action_to_simulation_action_dict = {}
        self.num_customers = 0
        # create tg observation from obs dictionary -
        tg_obs = self.observation(obs)
        return tg_obs

    def step(self, reinforce_action):
        """
        this function first translates the action chosen by the agent (in our case the action is an edge in the graph)
        from ppo action to env action, then steps through the env and translates the observation from dictionary to
        tg observation (tg graph)
        the reward is the negative distance the vehicle travelled between its current position and the chosen position
        :param reinforce_action: int - edge chosen by agent
        :return: tg_obs: tg.Data, reward: double, done: bool. info: Dict
        """
        # edge chosen goes from vehicle to node (which is either the depot or the customer chosen)
        _, node_chosen = self.action_to_simulation_action_dict[reinforce_action]
        if node_chosen == self.num_customers:
            action = self.num_nodes + self.env.DEPOT_INDEX
        else:
            action = node_chosen
        next_state, reward, done, _ = self.env.step(action)
        tg_obs = self.observation(next_state)
        return tg_obs, reward, done, {}

    def observation(self, obs):
        obs = self.env.observation(obs)
        return self.obs_to_graph_dict(obs)

    def obs_to_graph_dict(self, obs) -> tg.data.Data:
        """
        this function takes the observation and creates a graph including the following
        features:
        (indicator, x, y, node_demand)
        indicator: 0: customers, 1: depot, 2: vehicles
        x, y: position of the node in grid (double, double)
        node_demand: the customer demand or current vehicle capacity depending on the type of node (the vehicle
        capacity is negative)
        """
        # # if creating bipartite graph with only vehicles and customers use the following code:
        # there are 4 features all together :
        customer_positions = obs['customer_positions']
        customer_demands = obs['customer_demands']
        vehicle_capacity = obs['current_vehicle_capacity']
        vehicle_position = obs['current_vehicle_position']
        num_customers = customer_positions.shape[0]
        num_vehicles = 1
        num_depots = 1
        num_nodes = num_customers + num_vehicles + num_depots
        node_pos = np.vstack([customer_positions,
                              obs['depot_position'],
                              vehicle_position])
        # indicator is : 0: vehicles, 1: depot, 2: customers
        node_ind = np.vstack([np.ones(shape=(num_customers, 1)) * 0,
                              np.ones(shape=(num_depots, 1)) * 1,
                              np.ones(shape=(num_vehicles, 1)) * 2])
        node_demand = np.vstack([customer_demands.reshape(-1, 1),
                                 np.zeros(shape=(num_depots, 1)),
                                 -vehicle_capacity])
        # features are : indicator, pos_x, pos_y, demand/capacity
        node_features = np.hstack([node_ind, node_pos, node_demand])
        # customer edge indexes include all customers and depot
        customer_edge_indexes = [(i, j) for i, j in itertools.product(range(num_customers+1), range(num_customers+1)) if
                                 i != j]
        # add constraint edges
        customer_constraint_edge_indexes = [(i + num_customers + num_depots, j) for i in range(num_vehicles)
                                            for j in range(num_customers) if obs['action_mask'][j]]
        depot_constraint_edge_indexes = [(i + num_customers + num_depots, j + num_customers) for i in
                                         range(num_vehicles)
                                         for j in range(num_depots) if obs['action_mask'][self.env.DEPOT_INDEX]]
        edge_indexes = customer_edge_indexes + customer_constraint_edge_indexes + depot_constraint_edge_indexes
        edge_attributes = np.zeros(shape=(len(edge_indexes), 1))
        edge_attributes[len(customer_edge_indexes):] = 1
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        edge_indexes_tensor = torch.tensor(edge_indexes, dtype=torch.long,
                                           device=node_features_tensor.device).transpose(1, 0)
        edge_attributes_tensor = torch.tensor(edge_attributes, device=node_features_tensor.device, dtype=torch.float32)
        illegal_actions = torch.logical_not(edge_attributes_tensor.view(-1))
        graph_tg = tg.data.Data(x=node_features_tensor, edge_attr=edge_attributes_tensor,
                                edge_index=edge_indexes_tensor)
        graph_tg.illegal_actions = illegal_actions
        graph_tg.u = torch.tensor([[1]], device=node_features_tensor.device, dtype=torch.float32)
        self.num_nodes = graph_tg.x.shape[0]
        self.num_customers = num_customers
        self.action_to_simulation_action_dict = {i: (u, v) for i, (u, v) in enumerate(edge_indexes)}
        return graph_tg
