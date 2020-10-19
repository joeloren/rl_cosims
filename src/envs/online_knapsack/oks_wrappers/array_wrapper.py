import gym
import numpy as np


class KnapsackArrayWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self):
        # reset env -
        obs = self.env.reset()
        obs = obs['item_obs']
        return obs

    def step(self, action_chosen: int) -> (np.array, int, bool, np.array):
        new_obs, reward, is_done, info = self.env.step(action_chosen)
        new_obs = new_obs['item_obs']
        return new_obs, reward, is_done, info

    def seed(self, seed=None):
        self.env.seed(seed)

