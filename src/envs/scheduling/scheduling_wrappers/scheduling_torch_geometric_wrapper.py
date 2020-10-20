import itertools

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
