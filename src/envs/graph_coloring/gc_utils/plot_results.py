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
    pos = nx.spring_layout(graph)
    ec = nx.draw_networkx_edges(graph, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=colors,
                                with_labels=False, node_size=100, cmap=plt.cm.jet)
    plt.colorbar(nc)
    plt.axis('off')

