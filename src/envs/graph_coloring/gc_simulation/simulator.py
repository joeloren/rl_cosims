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
    # each node in the graph has :
    #   - color : the color of the node (default is -1)
    #   - open_time: time when node starts to be visible (in offline problem all start_time is 0)
    # each edge has a weight (assuming for now all edges have the same weight)
    num_colored_nodes: int  # number of colored nodes so far
    nodes_order: List  # list of the order the nodes were chosen
    current_time: int  # the current simulation time
    # this is an adjacency matrix where A[i, j] = -9999 if nodes (i, j) are not adjacent,
    # otherwise A[i, j] = color of node j (we will use this to check all of i's neighbors and see what their colors are)
    colors_adjacency_matrix: np.array


class Simulator(Env):
    EPSILON_TIME = 1e-6
    metadata = {"render.modes": ["human"]}

    @staticmethod
    def observation(obs):
        # the simulator returns obs without any changes (used for wrappers)
        return obs

    def __init__(self, num_max_nodes: int, max_time_steps: int, problem_generator) -> None:
        """
        Create a new graph_coloring. Note that you need to call reset() before starting the simulation.
        :param num_max_nodes: maximum number of nodes in the graph [int]
        :param problem_generator: a generator of type ScenarioGenerator which generates one instance of the problem
        and returns the initial state of the problem

        """
        super().__init__()
        # initial state is empty data variables if self.reset() is not called
        self.initial_state: State = State(unique_colors=set(), graph=nx.Graph(), num_colored_nodes=0,
                                          nodes_order=[], current_time=0, colors_adjacency_matrix=np.array([[]]))
        # current state of the simulation, this is updated at every step() call
        self.current_state: State = deepcopy(self.initial_state)
        self.problem_generator = problem_generator  # during reset this will generate a new instance of state
        self.current_time = 0  # a ticker which updates at the end of every step() to the next time step
        self.num_max_nodes = num_max_nodes
        self.max_time_steps = max_time_steps
        self.current_reward = 0.0  # this is needed so that we can calculate the current difference in the reward
        # nodes_id - the id of each node in the graph
        # TODO : add edges to observation dictionary
        # edge_indexes: List[Tuple] - edge index tuples where each tuple (i, j) is an edge between nodes i and j
        # current_time: int - current simulation time
        # nodes_colors: np.array [n_nodes, 1] -  for each node this is the color it is drawn with
        # (-1 means the node has not been colored yet)
        # used_color_ids: set - this is a bool vector if color is used or not
        # (0 - not used, 1 - used at least once in the graph)
        # nodes_start_time: np.array [n_nodes, 1] - start time for each node
        # forbidden_colors: a matrix of size [n_colors, n_max_nodes] where the Matrix[n, c] is 1 if color c is
        # forbidden for node n
        # color_adjacency_matrix: np.array [n_nodes, n_nodes] - this is an array where A[i, j] = -inf if there is no
        # edge between nodes i and j, and A[i, j] = node j color otherwise
        obs_spaces = {
            "nodes_id": spaces.Box(low=0, high=self.num_max_nodes, shape=(self.num_max_nodes,), dtype=np.int32),
            "forbidden_colors": spaces.MultiBinary(n=[self.num_max_nodes, self.num_max_nodes]),
            "edge_indexes": spaces.Box(low=0, high=self.num_max_nodes, shape=(self.num_max_nodes, 2), dtype=np.int32),
            "current_time": spaces.Box(low=0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32),
            "nodes_color": spaces.Box(low=-1, high=self.num_max_nodes, shape=(self.num_max_nodes,), dtype=np.int32),
            "used_color_ids": spaces.Box(low=0, high=self.num_max_nodes, shape=(self.num_max_nodes,), dtype=np.bool),
            "nodes_start_time": spaces.Box(low=0, high=self.num_max_nodes, shape=(self.num_max_nodes,), dtype=np.int32),
            "color_adjacency_matrix": spaces.Box(low=-np.inf, high=self.num_max_nodes,
                                                 shape=(self.num_max_nodes, self.num_max_nodes), dtype=np.int32)
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
        self.current_reward = 0.0
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
        # make sure node chosen has no color yet (color = -1)
        if self.current_state.graph.nodes()[node_chosen]['color'] != -1:
            return ValueError(f"node chosen has already been colored in previous steps, node id:{node_chosen}")
        # make sure  the color chosen is valid:
        for neighbor_node in self.current_state.graph.neighbors(node_chosen):
            if self.current_state.graph.nodes[neighbor_node]['color'] == color_chosen:
                raise ValueError(f"chose invalid color for node id: {node_chosen}, "
                                 f"problem is with neighbor node id:{neighbor_node}")
        # assuming we didn't fail in the previous checks, update the color of the node chosen to be the new color
        self.current_state.graph.nodes[node_chosen]['color'] = color_chosen
        if color_chosen not in self.current_state.unique_colors:
            self.current_state.unique_colors.add(color_chosen)
        self.current_state.num_colored_nodes += 1
        self.current_state.nodes_order.append(node_chosen)
        colors_adj_matrix_update = np.where(self.current_state.colors_adjacency_matrix[:, node_chosen] == -1)[0]
        self.current_state.colors_adjacency_matrix[colors_adj_matrix_update, node_chosen] = color_chosen
        reward = -len(self.current_state.unique_colors)
        # calculate the added reward in the current step
        reward_diff = reward - self.current_reward
        # save the current reward, to be used in the next calculation
        self.current_reward = reward
        is_done = self.calc_is_done()
        self.current_time += 1
        self.current_state.current_time = self.current_time
        if not is_done:
            # add new nodes to problem if online
            self.current_state = self.problem_generator.next(self.current_state)
        return self.current_state_to_observation(), reward_diff, is_done, {}

    def calc_is_done(self) -> bool:
        """
        calculate if the simulation is done
        simulation is done if the number of nodes in the graph are equal to the number of nodes we colored or we reached
        the maximum number of time steps
        """
        if (self.current_state.graph.number_of_nodes() == self.current_state.num_colored_nodes or
                self.current_time == self.max_time_steps):
            return True
        else:
            return False

    def current_state_to_observation(self):
        """
        this function returns the dictionary observation of the current state
        """
        nodes_id = np.array(self.current_state.graph.nodes)
        colors = nx.get_node_attributes(self.current_state.graph, 'color')
        positions = self.current_state.graph.nodes('pos')
        node_positions = [positions[i] for i in nodes_id]
        start_times = nx.get_node_attributes(self.current_state.graph, 'start_time')
        node_colors = np.array([colors[i] for i in nodes_id])
        node_start_times = np.array([start_times[i] for i in nodes_id])
        forbidden_colors = np.zeros(shape=(np.max(nodes_id)+1, np.max(nodes_id)+1), dtype=np.bool)
        for n in nodes_id:
            if 'forbidden_colors' in self.current_state.graph.nodes()[n].keys():
                for c in self.current_state.graph.nodes()[n]['forbidden_colors']:
                    forbidden_colors[n, c] = True
        obs = {
            'node_colors': node_colors,
            'used_colors': deepcopy(self.current_state.unique_colors),
            'nodes_id': nodes_id,
            'forbidden_colors': forbidden_colors,
            'edge_indexes': list(self.current_state.graph.edges),
            'current_time': self.current_time,
            'nodes_start_time': node_start_times,
            'node_positions': node_positions,
            'color_adjacency_matrix': self.current_state.colors_adjacency_matrix
        }
        return obs

    def get_number_of_colors_used(self):
        """
        this function returns the number of colors used so far in the graph.
        this is used to calculate the final reward of the simulation
        :return: num_colors_used: int - number of colors used in the graph
        """
        num_colors_used = len(self.current_state.unique_colors)
        return num_colors_used
