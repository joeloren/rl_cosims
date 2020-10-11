from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from gym import Env, spaces
import networkx as nx
from typing import Set, Dict, List


@dataclass
class State:
    """
    this class is the state of the env and has the full information about the graph
    """
    unique_colors: Set[int]  # this is a set of unique colors already used in the graph
    graph: nx.Graph  # this represents in the graph using network x representation
    num_colored_nodes: int  # number of colored nodes so far
    node_order: List  # list of the order the nodes were chosen
    # each node in the graph has :
    #   - color : the color of the node (default is -1)
    #   - visible: if True node is visible at the current time, otherwise False
    #   - open_time: time when node starts to be visible (in offline problem all start_time is 0)
    # each edge has a weight (assuming for now all edges have the same weight)


class Simulator(Env):
    EPSILON_TIME = 1e-6
    metadata = {"render.modes": ["human"]}

    def __init__(self, num_max_nodes: int, problem_generator) -> None:
        """
        Create a new graph_coloring_simulation. Note that you need to call reset() before starting the simulation.
        :param num_nodes: number of nodes in the graph [int]
        :param problem_generator: a generator of type ScenarioGenerator which generates one instance of the cvrp problem
        and returns the initial state of the problem

        """
        super().__init__()
        # initial state is empty data variables if self.reset() is not called
        self.initial_state: State = State(unique_colors=set(), graph=nx.Graph(), num_colored_nodes=0, node_order=[])
        # current state of the simulation, this is updated at every step() call
        self.current_state: State = deepcopy(self.initial_state)
        self.problem_generator = problem_generator  # during reset this will generate a new instance of state
        self.current_time = 0  # a ticker which updates at the end of every step() to the next time step
        self.num_max_nodes = num_max_nodes
        # nodes_id - the id of each node in the graph
        # TODO : add edges to observation dictionary
        # nodes_colors - for each node this is the color it is drawn with (-1 means the node has not been colored yet)
        # used_color_ids - this is a bool vector if color is used or not
        # (0 - not used, 1 - used at least once in the graph)
        obs_spaces = {
            "nodes_id": spaces.Box(
                low=0, high=self.num_max_nodes,
                shape=(self.num_max_nodes,), dtype=np.int32),
            "nodes_color": spaces.Box(
                low=-1, high=self.num_max_nodes,
                shape=(self.num_max_nodes,), dtype=np.int32),
            "used_color_ids": spaces.Box(
                low=0, high=self.num_max_nodes,
                shape=(self.num_max_nodes,), dtype=np.bool)
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
        self.initial_state = self.problem_generator.reset()
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

    def step(self, action_chosen: (int, int)) -> (float, int, bool, Dict):
        node_chosen = action_chosen[0]
        color_chosen = action_chosen[1]
        # make sure  the color chosen is valid:
        for neighbor_node in self.current_state.graph.neighbors(node_chosen):
            if neighbor_node['color'] == color_chosen:
                raise ValueError(f"chose invalid color for node id: {node_chosen}, "
                                 f"problem is with neighbor node id:{neighbor_node}")
        # assuming we didn't fail in the previous check, update the color of the node chosen to be the new color
        self.current_state.graph.nodes[node_chosen]['color'] = color_chosen
        if color_chosen not in self.current_state.unique_colors:
            self.current_state.unique_colors.add(color_chosen)
        self.current_state.num_colored_nodes += 1
        self.current_state.node_order.append(node_chosen)
        reward = len(self.current_state.unique_colors)
        is_done = self.calc_is_done()
        self.current_time += 1
        return self.current_state_to_observation(), reward, is_done, {}

    def calc_is_done(self):
        """
        calculate if the simulation is done
        simulation is done if the number of nodes in the graph is equal to the number of nodes we colored
        """
        if self.current_state.graph.number_of_nodes() == self.current_state.num_colored_nodes:
            return True
        else:
            return False

    def current_state_to_observation(self):
        """
        this function returns the dictionary observation of the current state
        """
        nodes_id = np.array(self.current_state.graph.nodes)
        colors = nx.get_node_attributes(self.current_state.graph, 'color')
        nodes_color = np.array([colors[i] for i in nodes_id])
        used_colors = np.zeros_like(self.num_max_nodes)
        for color_id in self.current_state.unique_colors:
            used_colors[color_id] = 1
        obs = {'nodes_color': nodes_color,
               'used_colors': used_colors,
               'nodes_id': nodes_id}
        return obs

    # def change_visibility_of_nodes(self):
    #     for node in self.current_state.graph.nodes:

