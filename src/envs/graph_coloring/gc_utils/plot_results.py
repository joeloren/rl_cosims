# basic imports
from typing import List
# mathematical imports
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns


def plot_gc_solution(graph: nx.Graph, nodes_order: List, plot_title: str = ""):
    # get unique groups
    colors = list(nx.get_node_attributes(graph, 'color').values())
    nodes = graph.nodes()
    # this function should print the full graph solution
    ec = nx.draw_networkx_edges(graph, graph.nodes('pos'), alpha=0.2)
    nc = nx.draw_networkx_nodes(graph, graph.nodes('pos'), nodelist=nodes, node_color=colors,
                                with_labels=True, node_size=100, cmap=plt.cm.jet)
    nx.draw_networkx_labels(graph, graph.nodes('pos'), labels={i: i for i in graph.nodes()})
    plt.title(plot_title)
    plt.colorbar(nc)
    plt.axis('off')


def plot_neighbor_node_colors(graph: nx.Graph):
    node_colors = nx.get_node_attributes(graph, 'color')
    edge_colors = np.zeros(shape=(len(graph.edges()), 2))
    for i, u, v in enumerate(graph.edges()):
        edge_colors[i, :] = np.array([node_colors[u], node_colors[v]])
    plt.plot(edge_colors)


def plot_multiple_result_stats(policy_values: dict, relative_to: str, output_file: str = None, show_mean: bool = True, ) -> None:
    relative_to_values = np.array(policy_values[relative_to])
    data_dict = {"Policy": [], "Values": []}
    data = {}
    for i, lab in enumerate(policy_values.keys()):
        data[lab] = (np.asarray(policy_values[lab]) - relative_to_values) / relative_to_values
        data_dict["Policy"] += [lab] * data[lab].size
        data_dict["Values"] += list(data[lab])

    # data_dict =  {k: v for k,v in zip(labels, data)}
    plt.figure()
    df = pd.DataFrame(data_dict, columns=list(data_dict.keys()))
    ax = sns.violinplot(x="Policy", y="Values", data=df, palette="pastel", linewidth=1,
                        scale="width", width=0.6, inner=None)
    for j, pol in enumerate(data.keys()):
        mean_val = np.mean(data[pol])
        ax.plot([j - 0.22, j + 0.22], [mean_val, mean_val], color="grey", linestyle="-")
        if show_mean:
            ax.text(j - 0.23, mean_val + 0.005, f"{mean_val:.2f}", color="dimgrey", fontweight="bold")
    ax.set_ylabel("relative difference to " + relative_to, fontsize=15)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=13)
    plt.tight_layout()
    ax.grid(alpha=1, linestyle="dotted")
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)
