from typing import List
from itertools import count

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def plot_gc_solution(graph: nx.Graph, nodes_order: List):
    # get unique groups
    colors = list(nx.get_node_attributes(graph, 'color').values())
    nodes = graph.nodes()
    # this function should print the full graph solution
    ec = nx.draw_networkx_edges(graph, graph.nodes('pos'), alpha=0.2)
    nc = nx.draw_networkx_nodes(graph, graph.nodes('pos'), nodelist=nodes, node_color=colors,
                                with_labels=True, node_size=100, cmap=plt.cm.jet)
    nx.draw_networkx_labels(graph, graph.nodes('pos'), labels={i: i for i in graph.nodes()})
    plt.colorbar(nc)
    plt.axis('off')

