import numpy as np
import networkx as nx
from typing import List


def randomly_destroy_solution(graph: nx.Graph, percentage: float) -> List[int]:
    """
    each node in the graph has :
      - color : the color of the node (default is -1)
      - open_time: time when node starts to be visible (in offline problem all start_time is 0)
    """
    num_nodes = graph.number_of_nodes()
    nodes_to_destroy = np.random.choice(range(num_nodes), size=np.floor(num_nodes*percentage), replace=False)
    return nodes_to_destroy



