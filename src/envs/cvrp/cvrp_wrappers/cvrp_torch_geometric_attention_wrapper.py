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
        self.num_customers = 0
        # create tg observation from obs dictionary -
        tg_obs = self.observation(obs)
        return tg_obs

    def step(self, reinforce_action):
        """
        this function first translates the action chosen by the agent (in our case the action is a node in the graph)
        from ppo action to env action, then steps through the env and translates the observation from dictionary to
        tg observation (tg graph)
        the reward is the negative distance the vehicle travelled between its current position and the chosen position
        :param reinforce_action: int - node chosen by agent
        :return: tg_obs: tg.Data, reward: double, done: bool. info: Dict
        """
        if reinforce_action == 0:
            # the depot was chosen
            action = self.num_customers + 2 - self.env.DEPOT_INDEX
        else:
            # customer chosen
            action = reinforce_action - 1
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
        indicator: 0: depot, 1: customers
        x, y: position of the node in grid (double, double)
        node_demand: the customer demand or current vehicle capacity depending on the type of node (the vehicle
        capacity is negative)
        """
        customer_positions = obs['customer_positions']
        customer_demands = obs['customer_demands']
        vehicle_capacity = obs['current_vehicle_capacity']
        num_customers = customer_positions.shape[0]
        num_depots = 1
        num_nodes = num_customers + num_depots
        node_pos = np.vstack([obs['depot_position'],
                              customer_positions])
        # indicator is : 0: customers, 1: depot,
        node_ind = np.vstack([np.ones(shape=(num_depots, 1)) * 0,
                              np.ones(shape=(num_customers, 1)) * 1])
        node_demand = np.vstack([np.zeros(shape=(num_depots, 1)),
                                 customer_demands.reshape(-1, 1)])
        # features are : indicator, pos_x, pos_y, demand/capacity
        node_features = np.hstack([node_ind, node_pos, node_demand])
        # customer edge indexes include all customers and depot
        edge_indexes = [(i, j) for i, j in itertools.product(range(num_customers + 1), range(num_customers + 1)) if
                        i != j]
        [edge_indexes.append((j, i)) for i, j in edge_indexes]
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        edge_indexes_tensor = torch.tensor(edge_indexes, dtype=torch.long,
                                           device=node_features_tensor.device).transpose(1, 0)
        edge_attributes_tensor = torch.ones(size=(len(edge_indexes), 1), device=node_features_tensor.device,
                                            dtype=torch.float32)
        illegal_actions = torch.zeros(size=(num_nodes, 1), device=node_features_tensor.device, dtype=torch.bool)
        if not obs['action_mask'][self.env.DEPOT_INDEX]:
            # depot option is not available, and therefore this action should be masked
            illegal_actions[0] = True
        # mask out all customers that there demand exceeds the vehicle current capacity
        illegal_actions[1:] = customer_demands > vehicle_capacity
        vehicle_current_customer_index = torch.tensor(obs["current_vehicle_customer"],device=node_features_tensor.device,
                                                dtype=torch.int32)
        graph_tg = tg.data.Data(x=node_features_tensor, edge_attr=edge_attributes_tensor,
                                edge_index=edge_indexes_tensor)
        graph_tg.illegal_actions = illegal_actions
        graph_tg.vehicle_current_customer_index = vehicle_current_customer_index
        graph_tg.u = torch.tensor([[1]], device=node_features_tensor.device, dtype=torch.float32)
        self.num_customers = num_customers
        return graph_tg
