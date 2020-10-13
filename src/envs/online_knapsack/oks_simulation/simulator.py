from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from gym import Env, spaces
from typing import Set, Dict, List


@dataclass
class State:
    """
    this class is the full state of the env and has the full information about the past
    """
    current_item: List  # features of the current item, composed of [value, cost]
    num_steps_taken: int  # number of steps taken in the episode
    max_steps: int  # max length of episode
    current_capacity: float  # current used capacity in the knapsack
    max_capacity: float  # max capacity available in the knapsack0
    item_history: List  # list of previously viewed items, composed of [value, cost, is_taken]


class Simulator(Env):
    EPSILON_TIME = 1e-6
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_steps: int, max_capacity: float, problem_generator) -> None:
        """
        Create a new online_knapsack. Note that you need to call reset() before starting the simulation.
        :param max_steps: max length of episode [int]
        :param problem_generator: a generator of type ScenarioGenerator which generates one instance of the online knapsack
        and returns the initial state of the problem

        """
        super().__init__()
        # initial state is empty data variables if self.reset() is not called
        self.initial_state: State = State(current_item=[], num_steps_taken=0, max_steps=max_steps,
                                          current_capacity=0.0, max_capacity=max_capacity, item_history=[])
        # current state of the simulation, this is updated at every step() call
        self.current_state: State = deepcopy(self.initial_state)
        self.problem_generator = problem_generator  # during reset this will generate a new distribution over items
        self.current_time = 0  # a ticker which updates at the end of every step() to the next time step
        obs_spaces = {
            "item_obs": spaces.Box(
                low=0, high=1,
                shape=(4,), dtype=np.float32)
        }
        self.observation_space = spaces.Dict(obs_spaces)

    def render(self, mode="human", close=False) -> None:
        """
        this function is needed for gym environment. for now doesn't do anything. in the future should create a graph
        of the current state
        :param mode:
        :param close:
        :return:
        """
        super(Simulator, self).render(mode=mode)

    def reset(self) -> Dict:
        self.initial_state.current_item = self.problem_generator.reset()
        self.current_state = deepcopy(self.initial_state)
        self.current_time = 0
        return self.current_state_to_observation()

    def seed(self, seed=None) -> None:
        """
        define seed in problem generator
        :param seed: seed to be used [int]
        :return:
        """
        self.problem_generator.seed(seed)

    def step(self, action_chosen: int) -> (np.array, int, bool, Dict):
        """
        add current item to the history list, if the current item is chosen, update the capacity
        """
        self.current_state.item_history.append(self.current_state.current_item)
        if action_chosen == 1:
            self.current_state.current_capacity += self.current_state.current_item[1]
            reward = self.current_state.current_item[0]
        else:
            reward = 0.0
        self.current_state.current_item = self.problem_generator.sample_item()
        is_done = self.calc_is_done()
        self.current_time += 1
        return self.current_state_to_observation(), reward, is_done, {}

    def calc_is_done(self):
        """
        calculate if the simulation is done
        simulation is done if num_steps_taken >= max_steps, or current_capacity >= max_capacity
        """
        if (self.current_state.num_steps_taken >= self.current_state.max_steps) or (self.current_state.current_capacity >= self.current_state.max_capacity):
            return True
        else:
            return False

    def current_state_to_observation(self):
        """
        this function returns the dictionary observation of the current state
        """
        time_ratio = float(self.current_state.num_steps_taken) / float(self.current_state.max_steps)
        capacity_ratio = float(self.current_state.current_capacity) / float(self.current_state.max_capacity)
        value = self.current_state.current_item[0]
        cost = self.current_state.current_item[1]
        item_obs = np.array([time_ratio, capacity_ratio, value, cost])
        obs = {'item_obs': item_obs}
        return obs

    # def change_visibility_of_nodes(self):
    #     for node in self.current_state.graph.nodes:

