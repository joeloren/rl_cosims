import numpy as np

from src.envs.graph_coloring.gc_experimentation.problems import (create_fixed_static_problem,
                                                                 create_er_random_graph_problem)
from src.envs.graph_coloring.gc_baselines.simple_policies import random_policy, random_policy_without_newcolor
from src.envs.graph_coloring.gc_baselines.ortools_policy import ORToolsOfflinePolicy
from src.envs.graph_coloring.gc_utils.graph_utils import (create_graph_from_observation, add_color_nodes_to_graph,
                                                          create_subproblem_from_partial_solution)
from src.envs.graph_coloring.gc_utils.plot_results import plot_gc_solution

from matplotlib import pyplot as plt


def test_fixed_problem():
    # test if reset works
    nodes = [i for i in range(10)]
    edges = [(u, v) for u in range(5) for v in range(5, 10) if np.random.random() < 0.5]
    env = create_fixed_static_problem(nodes_ids=nodes, edge_indexes=edges)
    # make sure that before reset there are no nodes
    assert len(env.initial_state.graph.nodes) == 0
    # make sure that before reset there are no edges
    assert len(env.initial_state.graph.edges) == 0
    obs = env.reset()
    # make sure that the right number of nodes was created
    assert len(obs["nodes_id"]) == len(nodes)
    # make sure the right number of edges was created
    assert len(obs["edge_indexes"]) == len(edges)
    # make sure no colors have been used (all colors should be -1
    assert np.all(obs["node_colors"] == -1)


def test_random_offline_er_problem():
    # test if reset works
    num_initial_nodes = 10
    num_new_nodes = 0
    prob_edge = 0.3
    env = create_er_random_graph_problem(num_new_nodes=num_new_nodes, num_initial_nodes=num_initial_nodes,
                                         prob_edge=prob_edge, is_online=False, random_seed=0)
    # make sure that before reset there are no nodes
    assert len(env.initial_state.graph.nodes) == 0
    # make sure that before reset there are no edges
    assert len(env.initial_state.graph.edges) == 0
    obs = env.reset()
    # make sure that the right number of nodes was created
    assert len(obs["nodes_id"]) == num_initial_nodes
    # make sure no colors have been used (all colors should be -1
    assert np.all(obs["node_colors"] == -1)
    # make sure node has color and time attribute
    assert np.all(obs['nodes_start_time'] == 0)


def test_random_online_er_problem():
    # test if reset works
    num_initial_nodes = 2
    num_new_nodes = 5
    prob_edge = 0.3
    env = create_er_random_graph_problem(num_new_nodes=num_new_nodes, num_initial_nodes=num_initial_nodes,
                                         prob_edge=prob_edge, is_online=True, random_seed=0)
    # make sure that before reset there are no nodes
    assert len(env.initial_state.graph.nodes) == 0
    # make sure that before reset there are no edges
    assert len(env.initial_state.graph.edges) == 0
    obs = env.reset()
    # make sure that the right number of nodes was created
    assert len(obs["nodes_id"]) == num_initial_nodes
    # make sure no colors have been used (all colors should be -1
    assert np.all(obs["node_colors"] == -1)
    # make sure node has color and time attribute
    assert np.all(obs['nodes_start_time'] == 0)
    obs, reward, is_done, _ = env.step((0, 0))  # color node id 0 with color 0
    # make sure 1 node was added
    assert len(env.current_state.graph.nodes()) == num_initial_nodes + 1
    # make sure the first node was colored
    assert obs['node_colors'][0] == 0
    is_done = False
    i = 1
    while not is_done:
        # color each node with a color matching it's id
        obs, reward, is_done, _ = env.step((i, i))
        print(f"current time is:{obs['current_time']}, num_nodes:{len(obs['node_colors'])}")
        i += 1
    # make sure env added the right amount of nodes
    assert len(env.current_state.graph.nodes()) == num_initial_nodes + num_new_nodes


def test_illegal_action():
    # test if reset works
    nodes = [i for i in range(10)]
    edges = [(u, v) for u in range(5) for v in range(5, 10) if np.random.random() < 0.5]
    env = create_fixed_static_problem(nodes_ids=nodes, edge_indexes=edges)
    obs = env.reset()
    done = False
    action = (0, 0)  # chose the first node and colored it 0
    next_obs, reward, is_done, _ = env.step(action)
    # make sure now one color is used
    assert reward == 1
    # make sure simulation is not done yet
    assert is_done is False
    # check if color 0 was added to observation
    assert next_obs["node_colors"][0] == 0
    assert 0 in next_obs["used_colors"]


def test_random_policy_with_fixed_problem():
    # test if reset works
    nodes = [i for i in range(10)]
    edges = [(u, v) for u in range(5) for v in range(5, 10) if np.random.random() < 0.5]
    env = create_fixed_static_problem(nodes_ids=nodes, edge_indexes=edges)
    obs = env.reset()
    done = False
    action = random_policy(obs, env)
    color_chosen = action[1]
    # make sure that the first color chosen is 0
    assert color_chosen == 0
    while not done:
        action = random_policy(obs, env)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs


def test_random_policy_with_er_online_problem():
    num_initial_nodes = 2
    num_new_nodes = 10
    prob_edge = 0.3
    env = create_er_random_graph_problem(num_new_nodes=num_new_nodes, num_initial_nodes=num_initial_nodes,
                                         prob_edge=prob_edge, is_online=True, random_seed=0)
    obs = env.reset()
    done = False
    action = random_policy(obs, env)
    color_chosen = action[1]
    # make sure that the first color chosen is 0
    assert color_chosen == 0
    while not done:
        action = random_policy(obs, env)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
    print("")
    print(f"number of colors used by random policy:{len(obs['used_colors'])}")
    print(f"initial number of nodes in problem:{len(env.initial_state.graph.nodes())},"
          f" final number of nodes:{len(env.current_state.graph.nodes())}")


def test_ortools_policy_with_fixed_problem():
    # test if policy works
    nodes = [i for i in range(10)]
    edges = [(u, v) for u in range(5) for v in range(5, 10) if np.random.random() < 0.5]
    env = create_fixed_static_problem(nodes_ids=nodes, edge_indexes=edges)
    obs = env.reset()
    or_tools_policy = ORToolsOfflinePolicy(verbose=True, timeout=100)
    action = or_tools_policy(obs, env)
    assert or_tools_policy.graph.nodes()[action[0]]['color'] != -1
    assert or_tools_policy.graph.nodes()[action[0]]['color'] == action[1]


def test_ortools_policy_with_er_offline_problem():
    num_initial_nodes = 15
    num_new_nodes = 0
    prob_edge = 0.3
    env = create_er_random_graph_problem(num_new_nodes=num_new_nodes, num_initial_nodes=num_initial_nodes,
                                         prob_edge=prob_edge, is_online=False, random_seed=0)
    obs = env.reset()
    or_tools_policy = ORToolsOfflinePolicy(verbose=True, timeout=1000)
    action = or_tools_policy(obs, env)
    assert or_tools_policy.graph.nodes()[action[0]]['color'] != -1
    assert or_tools_policy.graph.nodes()[action[0]]['color'] == action[1]


def test_create_graph_from_observation():
    sim = create_er_random_graph_problem(num_new_nodes=0, num_initial_nodes=8, prob_edge=0.4,
                                         is_online=False, random_seed=0)
    sim.seed(10)
    obs = sim.reset()
    is_done = False
    while not is_done:
        obs, reward, is_done, _ = sim.step(random_policy_without_newcolor(obs, sim))
        graph_nx = create_graph_from_observation(obs)
        # make sure all edges and nodes are the same in both graphs
        assert np.all(np.array(graph_nx.nodes()) == np.array(sim.current_state.graph.nodes()))
        assert np.all(np.array(graph_nx.edges()) == np.array(sim.current_state.graph.edges()))
        # make sure the attributes in both graphs are the same, except for the indicator attribute which we have in
        # graph_nx and not in the simulation graph
        assert dict(graph_nx.nodes(data='color')) == dict(sim.current_state.graph.nodes(data='color'))
        assert dict(graph_nx.nodes(data='start_time')) == dict(sim.current_state.graph.nodes(data='start_time'))
        graph_nx_pos = [v for v in dict(graph_nx.nodes(data='pos')).values()]
        graph_sim_pos = [v for v in dict(sim.current_state.graph.nodes(data='pos')).values()]
        assert np.all(graph_nx_pos) == np.all(graph_sim_pos)


def test_add_colors_graph():
    sim = create_er_random_graph_problem(num_new_nodes=0, num_initial_nodes=8, prob_edge=0.4,
                                         is_online=False, random_seed=0)
    sim.seed(10)
    obs = sim.reset()
    is_done = False
    while not is_done:
        obs, reward, is_done, _ = sim.step(random_policy_without_newcolor(obs, sim))
        graph_nx = create_graph_from_observation(obs)
        colors_graph_nx = add_color_nodes_to_graph(obs)
        # make sure color graph has the correct number of nodes
        assert len(graph_nx.nodes()) + len(obs["used_colors"]) + 1 == len(colors_graph_nx.nodes())
        num_colors_used = len(obs["used_colors"])
        num_colors_color_graph = len(colors_graph_nx.nodes()) - len(graph_nx.nodes())
        # make sure colored graph has the correct number of nodes added (should be number of used colors + 1)
        assert num_colors_color_graph == num_colors_used + 1


def test_color_adjacency_matrix():
    sim = create_er_random_graph_problem(num_new_nodes=0, num_initial_nodes=8, prob_edge=0.4,
                                         is_online=False, random_seed=0)
    sim.seed(10)
    obs = sim.reset()
    action = random_policy_without_newcolor(obs, sim)
    num_edges_to_node_zero = np.sum(sim.current_state.colors_adjacency_matrix[:, 0] == -1)
    obs, reward, is_done, _ = sim.step(action)
    num_colored_nodes = np.sum(sim.current_state.colors_adjacency_matrix[:, action[0]] == 0)
    # make sure the correct amount of nodes was changed from -1 to 0
    assert num_edges_to_node_zero == num_colored_nodes
    # make sure column was not changed (since the column indicates all the colors of the nodes that this node is
    # connected to)
    assert num_edges_to_node_zero == np.sum(sim.current_state.colors_adjacency_matrix[action[0], :] == -1)
    while not is_done:
        action = random_policy_without_newcolor(obs, sim)
        obs, reward, is_done, _ = sim.step(action)
        graph_nx = create_graph_from_observation(obs)
        colors_graph_nx = add_color_nodes_to_graph(obs)
        # make sure color graph has the correct number of nodes
        assert len(graph_nx.nodes()) + len(obs["used_colors"]) + 1 == len(colors_graph_nx.nodes())
        num_colors_used = len(obs["used_colors"])
        num_colors_color_graph = len(colors_graph_nx.nodes()) - len(graph_nx.nodes())
        # make sure colored graph has the correct number of nodes added (should be number of used colors + 1)
        assert num_colors_color_graph == num_colors_used + 1
    # make sure the matrix was completely changed from -1 to a color
    adj_matrix = sim.current_state.colors_adjacency_matrix
    num_nodes_not_colored = np.sum(adj_matrix == -1)
    assert num_nodes_not_colored == 0
    # make sure no wrong color was added by accident
    num_nodes_wrong_color = np.sum((adj_matrix > len(obs["used_colors"])-1))
    assert num_nodes_wrong_color == 0
    # make sure each node (column) has only one color that is not in the set of neighbor colors
    for r in range(adj_matrix.shape[0]):
        if np.sum(adj_matrix[:, r] == -9999) != adj_matrix.shape[0]:
            node_color = adj_matrix[adj_matrix[:, r] != -9999, r]
            assert np.all(node_color == node_color[0])
            neighbor_colors = adj_matrix[r, adj_matrix[r, :] != -9999]
            assert set(node_color) not in set(neighbor_colors)


def test_repair_preperation():
    # test if reset works
    np.random.seed(0)
    nodes = [i for i in range(10)]
    edges = [(u, v) for u in range(5) for v in range(5, 10) if np.random.random() < 0.5]
    env = create_fixed_static_problem(nodes_ids=nodes, edge_indexes=edges)

    obs = env.reset()
    is_done = False
    i = 0
    while not is_done:
        # color each node with a color matching it's id
        obs, reward, is_done, _ = env.step((i, i))
        print(f"current time is:{obs['current_time']}, num_nodes:{len(obs['node_colors'])}")
        i += 1
    new_graph = env.current_state.graph.copy()
    new_graph.nodes[1]['color'] = -1
    new_graph.nodes[9]['color'] = -1
    subproblem_graph = create_subproblem_from_partial_solution(new_graph)

    print('hi')
