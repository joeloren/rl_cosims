import numpy as np

from src.envs.graph_coloring.gc_experimentation.problems import create_fixed_static_problem
from src.envs.graph_coloring.gc_baselines.simple_policies import random_policy


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


def test_random_policy():
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






