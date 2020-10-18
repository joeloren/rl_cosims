import numpy as np
from typing import Tuple
from src.envs.online_knapsack.oks_simulation.simulator import Simulator


def random_policy(obs, env: Simulator) -> int:
    """
    this function chooses if to take the current item or not
    :param obs: Dict - the current observation of the current state in the env
    :param env: Simulator - the environment of online knapsack problem
    :return: action: int - a binary action if the item is taken or not
    """
    num = np.random.rand()
    if num > 0.5:
        return 1
    else:
        return 0

def simple_policy(obs, env: Simulator) -> int:
    """
    this function chooses if to take the current item or not
    :param obs: Dict - the current observation of the current state in the env
    :param env: Simulator - the environment of online knapsack problem
    :return: action: int - a binary action if the item is taken or not
    """
    num = np.random.rand()
    if obs['item_obs'][env.observation_indices['value']] > obs['item_obs'][env.observation_indices['cost']]:
        return 1
    else:
        return 0

