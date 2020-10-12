import numpy as np
import random
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph as er_graph
from matplotlib import pyplot as plt
from ortools.linear_solver import pywraplp

from src.envs.graph_coloring.gc_utils.plot_results import plot_gc_solution


def solve_or_tools(graph: nx.Graph, max_num_colors: int, timeout: float=40):
    """
    Given an undirected loopless graph G = (V, E), where V is a set of
      nodes, E <= V x V is a set of arcs, the Graph Coloring Problem is to
      find a mapping (coloring) F: V -> C, where C = {1, 2, ... } is a set
      of colors whose cardinality is as small as possible, such that
      F(i) != F(j) for every arc (i,j) in E, that is adjacent nodes must
      be assigned different colors.
      '''
      This model was created by Hakan Kjellerstrand (hakank@gmail.com)
      Also see my other Google CP Solver models:
      http://www.hakank.org/google_or_tools/
    :param graph:
    :param max_num_colors:
    :return:
    """
    solver = pywraplp.Solver.CreateSolver('graph_coloring_solver', "CBC")
    solver.set_time_limit(timeout)
    nodes = list(graph.nodes())
    edges = list(graph.edges())
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
    # run solver -
    results_status = solver.Solve()
    node_color = {}
    if results_status == solver.OPTIMAL:
        num_colors_used = int(solver.Objective().Value())
        print(f'number of colors: {num_colors_used}')
        colors_used = [int(u[i].SolutionValue()) for i in range(max_num_colors)]
        print(f'colors used: {colors_used}')
        node_color = {n:{} for n in nodes}
        color_node_mat = np.zeros(shape=(num_nodes, max_num_colors))
        for i in range(num_nodes):
            for j in range(max_num_colors):
                color_node_mat[i, j] = x[i, j].SolutionValue()
            color = np.where(color_node_mat[i, :] == 1)[0].item()
            node_color[i]['color'] = color
        nx.set_node_attributes(graph, node_color)
    elif results_status == solver.INFEASIBLE:
        print('No solution found.')
    else:
        print("solver could not find optimal solution")
    return node_color, graph


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
    node_colors, graph = solve_or_tools(graph, max_num_colors=max_num_colors, timeout=2000)
    plot_gc_solution(graph, [])
    plt.show()


if __name__ == '__main__':
    main()
    print("done!")
