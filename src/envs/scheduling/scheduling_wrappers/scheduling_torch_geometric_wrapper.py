import itertools

from typing import Dict
import networkx as nx
import numpy as np
import torch
import torch_geometric as tg
from gym import Wrapper


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

    def observation(self, obs):
        obs_env = self.env.observation(obs)  # List of List
        return self.obs_to_graph_dict(obs_env)

    def obs_to_graph_dict(self, obs) -> tg.data.Data:

        num_machines = len(obs)
        num_jobs = np.sum([len(o) for o in obs])
        all_jobs_list = list(itertools.chain.from_iterable(obs))

        # create graph and add nodes and edges
        graph_nx = nx.DiGraph()
        graph_nodes = list(range(num_machines + num_jobs))
        graph_edges = list()
        for m in range(num_machines):
            for j in range(num_machines, num_machines + num_jobs):
                graph_edges.append((m, j))
                graph_edges.append((j, m))

        graph_nx.add_nodes_from(graph_nodes)
        graph_nx.add_edges_from(graph_edges)

        # add attributes to graph
        # Feature 0 - for machines - total time on machine, for jobs the length of the job
        # Feature 1 indicator of machine=0/job=1.
        node_features = np.zeros(shape=(num_machines + num_jobs, 2))
        for j in range(num_machines + num_jobs):
            if j < num_machines:
                feature = [np.sum(obs[j]), 0]
            else:
                feature = [np.sum(obs[j]), 0]
            node_features[j] = feature

        # create directed graph from original graph
        graph_tg = tg.utils.from_networkx(graph_nx)

        # save node features as x tensor
        graph_tg.x = torch.tensor(node_features, dtype=torch.float32)
        # save edge features as tensor

        return graph_tg

