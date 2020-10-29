# basic imports
from pathlib import Path
from typing import Dict, Tuple
import time
# nn imports
import torch
# our imports
from src.agents.tg_ppo_agent import PPOAgent
from src.models.tg_models import PolicyGNN
from src.envs.graph_coloring.gc_simulation.simulator import Simulator
from src.envs.graph_coloring.gc_wrappers.gc_torch_geometric_wrappers import GraphWithColorsWrapper


class PPOPolicy:
    def __init__(self, agent_config_dict: Dict, model_config_dict: Dict, model_location: str, env: Simulator):
        model_folder = Path(model_location)
        # this is used only for the observation function, the real observation is not needed (only the function to
        # translate observation to graph
        # we save both normalized observation and torch observation converter so that we can create both conversions
        self.tg_env = GraphWithColorsWrapper(env)
        # choose model file as the last model saved
        model_file = max(list(model_folder.glob('model_ep_*')))
        print(f'Loading model file: {model_file}')
        cuda = torch.cuda.is_available()
        if cuda:
            gnn_model_params = torch.load(model_file)
        else:
            gnn_model_params = torch.load(model_file, map_location=torch.device('cpu'))
        # initialize policy with model parameters
        policy_model = PolicyGNN(model_config_dict, model_name='ppo_policy')
        policy_model.load_state_dict(gnn_model_params)
        # initialize agent with current policy model
        self.agent = PPOAgent(None, agent_config_dict, policy_model, [0], {})

    def __call__(self, state: Dict, env: GraphWithColorsWrapper) -> Tuple:
        """
        This function calls the policy network after converting the state to a torch geometric graph and returns the
        action chosen (after the action is converted into a simulation action)
        :param state: Dict of the current state
        :param env: Simulation wrapper, this is used only for converting the current state to the torch geometric state
        and also to save the dictionaries from edge to node and color chosen
        :return: (node_chosen, color_chosen) : (int, int) node and color id chosen
        """
        # make sure policy is in eval mode
        self.agent.policy.eval()
        # convert state to torch geometric graph
        obs_tg = self.tg_env.observation(state)
        # select action using the agent policy network
        policy_action = self.agent.select_action_and_log_prob(state=obs_tg)[0]
        # convert action from edge index to (node_id, color_chosen) so that simulation can accept the action
        (node_chosen, color_node_chosen) = self.tg_env.action_to_simulation_action_dict[policy_action]
        color_chosen = self.tg_env.node_to_color_dict[color_node_chosen]
        action = (node_chosen, color_chosen)
        return action

