import itertools

import numpy as np
import torch
import torch_geometric as tg
from gym import Wrapper


class GeometricAttentionWrapper(Wrapper):
    """
    this class is a wrapper for the cvrp simulation. It takes the observation given from the simulator and translates
    it to a Fully connected torch geometric graph where the nodes are [depot, customer 0, ..., customer n].
    all nodes are connected and they are then the actions.
    we mask out the actions that are not currently feasible (where demand exceeds the current capacity or if the vehicle
    is currently at the depot)
    """

    def __init__(self, env):
        super().__init__(env)
        self.num_nodes = 0
        self.num_customers = 0

    def reset(self):
        """
        this function resets the environment and the wrapper
        :return: tg_obs: tg.Data - graph with all features as a torch geometric graph
        """
        # reset env -
        obs = self.env.reset()
        self.num_customers = obs['customer_positions'].shape[0]
        # create tg observation from obs dictionary -
        tg_obs = self.observation(obs)
        return tg_obs

    def seed(self, seed=None):
        self.env.seed(seed)

    def step(self, reinforce_action):
        """
        this function first translates the action chosen by the agent (in our case the action is a node in the graph)
        from ppo action to env action, then steps through the env and translates the observation from dictionary to
        tg observation (tg graph)
        the reward is the negative distance the vehicle travelled between its current position and the chosen position
        :param reinforce_action: int - node chosen by agent
        :return: tg_obs: tg.Data, reward: double, done: bool. info: Dict
        """
        if reinforce_action == self.num_customers:
            # the depot was chosen
            action = self.num_customers + 2 + self.env.DEPOT_INDEX
        else:
            # customer chosen
            action = reinforce_action
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
        (indicator, x, y, node_demand, node_visited)
        indicator: 0: depot, 1: customers
        x, y: position of the node in grid (double, double)
        node_demand: the customer demand or current vehicle capacity depending on the type of node (the vehicle
        capacity is negative)
        """
        customer_positions = obs['customer_positions']
        customer_visited = obs['customer_visited']
        vehicle_position = obs["current_vehicle_position"]
        customer_demands = obs['customer_demands'] / obs['max_vehicle_capacity']
        vehicle_capacity = obs['current_vehicle_capacity'] / obs['max_vehicle_capacity']
        num_customers = customer_positions.shape[0]
        num_depots = 1
        num_vehicles = 1
        num_nodes = num_customers + num_depots + num_vehicles
        node_pos = np.vstack([customer_positions,
                              obs['depot_position'],
                              vehicle_position])
        # if vehicle is currently at the depot position than depot is treated like a visited node
        if np.array_equal(vehicle_position, obs['depot_position']):
            depot_visited = np.zeros(shape=(num_depots, 1))
        else:
            depot_visited = np.ones(shape=(num_depots, 1))
        # node visited is True if the node can be chosen as action (customer that is not yet visited or depot if the
        # vehicle is not currently at the depot). Otherwise the node visited value is False (this is always true for
        # vehicle node)
        node_visited = np.vstack([np.logical_not(customer_visited).reshape(-1, 1),
                                  depot_visited,
                                  np.zeros(shape=(num_vehicles, 1))])
        # indicator is : 0: customers, 1: depot, 2: vehicle
        node_ind = np.vstack([np.ones(shape=(num_customers, 1)) * 0,
                              np.ones(shape=(num_depots, 1)) * 1,
                              np.ones(shape=(num_vehicles, 1)) * 2])
        node_demand = np.vstack([customer_demands.reshape(-1, 1),
                                 np.zeros(shape=(num_depots, 1)),
                                 -vehicle_capacity])
        customer_nodes = np.where(node_ind == 0)[0]
        depot_nodes = np.where(node_ind == 1)[0]
        vehicle_nodes = np.where(node_ind == 2)[0]
        # features are : pos_x, pos_y, demand/capacity
        node_features = np.hstack([node_ind, node_pos, node_demand, node_visited])
        # customer edge indexes include all customers and depot
        # edge_indexes = [(i, j) for i, j in itertools.product(range(num_customers + 1), range(num_customers + 1)) if
        #                 i != j]
        customer_and_depot_nodes = np.concatenate([customer_nodes, depot_nodes])
        vehicle_edge_indexes = [(i.item(), j.item()) for i in vehicle_nodes for j in customer_and_depot_nodes]
        vehicle_edge_indexes = vehicle_edge_indexes + [(j, i) for i, j in vehicle_edge_indexes]
        edge_indexes_directed = vehicle_edge_indexes
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        edge_indexes_tensor = torch.tensor(edge_indexes_directed, dtype=torch.long,
                                           device=node_features_tensor.device).transpose(1, 0)
        edge_attributes_tensor = torch.ones(size=(len(edge_indexes_directed), 1), device=node_features_tensor.device,
                                            dtype=torch.float32)
        illegal_actions = np.zeros(shape=(num_nodes,))
        if not obs['action_mask'][self.env.DEPOT_INDEX]:
            # depot option is not available, and therefore this action should be masked
            illegal_actions[depot_nodes] = True
        # mask out all customers that there demand exceeds the vehicle current capacity
        illegal_actions[customer_nodes] = np.logical_or(customer_demands > vehicle_capacity,
                                                        customer_visited)
        # mask out the vehicle nodes since they can never be chosen
        illegal_actions[vehicle_nodes] = True
        illegal_actions_tensor = torch.tensor(illegal_actions, device=node_features_tensor.device,
                                              dtype=torch.bool)
        graph_tg = tg.data.Data(x=node_features_tensor, edge_attr=edge_attributes_tensor,
                                edge_index=edge_indexes_tensor)
        graph_tg.illegal_actions = illegal_actions_tensor
        graph_tg.u = torch.tensor([[1]], device=node_features_tensor.device, dtype=torch.float32)
        self.num_customers = num_customers
        return graph_tg
