from typing import Dict, List, Tuple

import numpy as np
import random
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph as er_graph
from matplotlib import pyplot as plt
from ortools.linear_solver import pywraplp

from src.envs.graph_coloring.gc_utils.plot_results import plot_gc_solution
from src.envs.graph_coloring.gc_simulation.simulator import Simulator
from src.envs.graph_coloring.gc_experimentation.problems import create_fixed_static_problem


def solve_or_tools(nodes: List[int], edges: List[Tuple], max_num_colors: int, timeout: float = 40, verbose=False):
    """
    Given an undirected graph with no loops G = (V, E), where V is a set of
      nodes, E <= V x V is a set of arcs, the Graph Coloring Problem is to
      find a mapping (coloring) F: V -> C, where C = {1, 2, ... } is a set
      of colors whose cardinality is as small as possible, such that
      F(i) != F(j) for every arc (i,j) in E, that is adjacent nodes must
      be assigned different colors.
      '''
      This model was created by Hakan (hakank@gmail.com)
      Also see my other Google CP Solver models:
      http://www.hakank.org/google_or_tools/
    :param nodes: nx graph of problem
    :param edges:
    :param verbose: if True should print logs
    :param max_num_colors: maximum number of colors allowed in graph
    :param timeout: run time given to algorithm
    :return:
    """
    solver = pywraplp.Solver.CreateSolver('graph_coloring_solver', "CBC")
    solver.set_time_limit(timeout)
    num_nodes = len(nodes)
    # x[i,c] = 1 means that node i is assigned color c
    x = {}
    for n in nodes:
        for c in range(max_num_colors):
            x[n, c] = solver.IntVar(0, 1, 'v[%i,%i]' % (n, c))
    # u[c] = 1 means that color c is used, i.e. assigned to some node
    u = [solver.IntVar(0, 1, 'u[%i]' % i) for i in range(max_num_colors)]
    # the objective function is the number of colors used
    obj = solver.Sum(u)
    # constraints are:
    # each node must be assigned exactly one color
    for n in nodes:
        solver.Add(solver.Sum([x[n, c] for c in range(max_num_colors)]) == 1)
    # adjacent nodes cannot be assigned the same color
    # (and adjust to 0-based)
    for e in edges:
        for c in range(max_num_colors):
            solver.Add(x[e[0], c] + x[e[1], c] <= u[c])
    objective = solver.Minimize(obj)
    if verbose:
        print(f"objective is:{objective}")
    # run solver -
    results_status = solver.Solve()
    node_color = {}
    graph = nx.Graph()
    graph.add_nodes_from(nodes, color=-1)
    graph.add_edges_from(edges)
    found_solution = False
    if results_status == solver.OPTIMAL:
        found_solution = True
        num_colors_used = int(solver.Objective().Value())
        if verbose:
            print(f'number of colors: {num_colors_used}')
        colors_used = [int(u[i].SolutionValue()) for i in range(max_num_colors)]
        if verbose:
            print(f'colors used: {colors_used}')
        node_color = {n: {} for n in nodes}
        color_node_mat = np.zeros(shape=(num_nodes, max_num_colors))
        for i in range(num_nodes):
            for j in range(max_num_colors):
                color_node_mat[i, j] = x[i, j].SolutionValue()
            color = np.where(color_node_mat[i, :] == 1)[0].item()
            node_color[i]['color'] = color
        nx.set_node_attributes(graph, node_color)
    elif results_status == solver.INFEASIBLE:
        if verbose:
            print('No solution found.')
    else:
        if verbose:
            print("solver could not find optimal solution")
    return node_color, graph, found_solution


class ORToolsOfflinePolicy:
    def __init__(self, verbose=False, timeout=10):
        super().__init__()
        self.timeout = timeout
        self.verbose = verbose
        self.__name__ = 'or_tools'
        self.graph = None
        self.node_colors = {}

    def reset(self, obs):
        """
        :param obs: observation (not used currently)
        this function resets the solution
        :return:
        """
        self.graph = None
        self.node_colors = {}

    def __call__(self, obs: Dict, env: Simulator):
        """
        this function returns the next node to color and its new color
        :param obs: observation dictionary from environment
        :param env: simulation (should not be used here)
        :return: (node_chosen, color_chosen)
        """
        # if current time is 0, run problem and save results -
        if obs['current_time'] == 0:
            nodes = list(obs["nodes_id"])
            edges = obs["edge_indexes"]
            found_solution = False
            num_iters = 0
            for i in range(4, len(nodes)):
                max_num_colors = i
                if self.verbose:
                    print(f"trying to solve or-tools with maximum colors:{i} , num nodes:{len(nodes)}")
                node_colors, graph, found_solution = solve_or_tools(nodes, edges, max_num_colors,
                                                                    timeout=self.timeout, verbose=self.verbose)
                if found_solution:
                    self.graph = graph
                    self.node_colors = node_colors
                    break
                num_iters += 1
            if found_solution is False:
                raise RuntimeError(f"no solution was found by or tools for the current graph, "
                                   f"num_iters:{num_iters}, runtime for each try:{self.timeout}")

        available_node_indexes = np.where(obs["node_colors"] == -1)[0]
        node_chosen = np.random.choice(available_node_indexes, 1)[0]
        color_chosen = self.node_colors[node_chosen]['color']
        if color_chosen == -1:
            raise ValueError(f"current time:{obs['current_time']}, "
                             f"or tools solution did not work, color in solution for chosen node is -1")
        if obs["node_colors"][node_chosen] != -1:
            raise ValueError(f"current time:{obs['current_time']}, "
                             f"node_chosen :{node_chosen}, already has a color:{obs['node_colors'][node_chosen]}")
        return node_chosen, color_chosen


def main():
    #  -----------------------------
    # simple graph generator
    #  -----------------------------
    # graph = nx.Graph()
    # nodes = list(range(10))
    # edges = [(u, v) for u in range(5) for v in range(5, 10)]
    # graph.add_nodes_from(nodes)
    # graph.add_edges_from(edges)
    #  -----------------------------
    # erdos renyi graph generator -
    #  -----------------------------
    np.random.seed(0)
    random.seed(0)
    num_nodes = 20
    prob_edges = 0.3
    max_num_colors = 10
    graph = er_graph(n=num_nodes, p=prob_edges)
    node_att_color = {i: {'color': -1} for i in graph.nodes()}
    nx.set_node_attributes(graph, node_att_color)
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    node_colors, graph, found_solution = solve_or_tools(nodes=nodes, edges=edges,
                                                        max_num_colors=max_num_colors, timeout=2000)
    plot_gc_solution(graph, [])
    or_tools_policy = ORToolsOfflinePolicy(verbose=True, timeout=1000)
    env = create_fixed_static_problem(nodes, edges, random_seed=0)
    obs = env.reset()
    done = False
    while not done:
        node_chosen, color_chosen = or_tools_policy(obs, env)
        action = (node_chosen, color_chosen)
        obs, reward, done, _ = env.step(action)
    plt.figure()
    plot_gc_solution(env.current_state.graph, [])
    plt.show()


if __name__ == '__main__':
    main()
    print("done!")
