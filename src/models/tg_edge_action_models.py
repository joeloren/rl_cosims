import numpy as np
# torch imports
import torch
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, BatchNorm1d
from torch_geometric.nn import MetaLayer
from torch.distributions import Categorical
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric import data as tg_data, utils as tg_utils
# our imports
from src.models.tg_core_models import EdgeModel, NodeModel, GlobalModel


class PolicyMultipleMPGNN(torch.nn.Module):
    def __init__(self, cfg, model_name):
        super(PolicyMultipleMPGNN, self).__init__()
        self.model_name = model_name
        self.n_passes = cfg['n_passes']
        self.use_value_critic = cfg['use_value_critic']
        self.use_batch_norm = cfg['use_batch_norm']
        self.logit_normalizer = cfg['logit_normalizer']
        if cfg['use_batch_norm']:
            self.edge_embedding_model = Seq(Lin(cfg['edge_feature_dim'], cfg['edge_embedding_dim']),
                                            LeakyReLU(),
                                            BatchNorm1d(cfg['edge_embedding_dim'])
                                            )

            self.node_embedding_model = Seq(Lin(cfg['node_feature_dim'], cfg['node_embedding_dim']),
                                            LeakyReLU(),
                                            BatchNorm1d(cfg['node_embedding_dim'])
                                            )
        else:
            self.edge_embedding_model = Seq(Lin(cfg['edge_feature_dim'], cfg['edge_embedding_dim']),
                                            LeakyReLU(),
                                            )

            self.node_embedding_model = Seq(Lin(cfg['node_feature_dim'], cfg['node_embedding_dim']),
                                            LeakyReLU(),
                                            )

        self.global_embedding_model = Seq(
            Lin(cfg['global_feature_dim'], cfg['global_embedding_dim']),
            LeakyReLU())

        # assume that after embedding the edges, nodes and globals have a new length
        mp_dict = [MetaLayer(EdgeModel(n_edge_features=cfg['edge_embedding_dim'],
                                       n_node_features=cfg['node_embedding_dim'],
                                       n_global_features=cfg['global_embedding_dim'],
                                       n_hiddens=cfg['edge_hidden_dim'],
                                       n_targets=cfg['edge_target_dim'],
                                       use_batch_norm=cfg['use_batch_norm']),

                             NodeModel(n_edge_features=cfg['edge_embedding_dim'],
                                       n_node_features=cfg['node_embedding_dim'],
                                       n_global_features=cfg['global_embedding_dim'],
                                       n_hiddens=cfg['node_hidden_dim'],
                                       n_targets=cfg['node_target_dim'],
                                       use_batch_norm=cfg['use_batch_norm']),

                             GlobalModel(n_node_features=cfg['node_embedding_dim'],
                                         n_global_features=cfg['global_embedding_dim'],
                                         n_hiddens=cfg['global_hidden_dim'],
                                         n_targets=cfg['global_target_dim'],
                                         use_batch_norm=cfg['use_batch_norm']))]
        for i in range(1, self.n_passes):
            mp_dict.append(MetaLayer(EdgeModel(n_edge_features=cfg['edge_target_dim'],
                                               n_node_features=cfg['node_target_dim'],
                                               n_global_features=cfg['global_target_dim'],
                                               n_hiddens=cfg['edge_hidden_dim'],
                                               n_targets=cfg['edge_target_dim']),

                                     NodeModel(n_edge_features=cfg['edge_target_dim'],
                                               n_node_features=cfg['node_target_dim'],
                                               n_global_features=cfg['global_target_dim'],
                                               n_hiddens=cfg['node_hidden_dim'],
                                               n_targets=cfg['node_target_dim']),

                                     GlobalModel(n_node_features=cfg['node_target_dim'],
                                                 n_global_features=cfg['global_target_dim'],
                                                 n_hiddens=cfg['global_hidden_dim'],
                                                 n_targets=cfg['global_target_dim'])))

        self.message_passing = torch.nn.ModuleList(mp_dict)

        self.edge_decoder_model = Lin(cfg['edge_target_dim'], cfg['edge_dim_out'])
        if self.use_value_critic:
            self.value_model = Seq(Lin(cfg['global_target_dim'], cfg['value_embedding_dim']),
                                   LeakyReLU(),
                                   Lin(cfg['value_embedding_dim'], 1))

        # in our case this is not needed since we don't use the nodes or global in our agent (if
        # needed in the output
        # this should be uncommented)

        # self.node_decoder_model = Lin(cfg['node_target_dim'], cfg['node_dim_out'])
        #
        # self.global_decoder_model = Lin(cfg['global_target_dim'], cfg['global_dim_out'])

    def forward(self, state: tg.data.Batch):
        """
        this function takes the graph edge,node and global features and returns the features after :
        embedding (fully connected layer to transfer edge,node and global features to the
        embedding size)
        message passing (graph network block, similar to the paper :
        https://arxiv.org/pdf/1806.01261.pdf
        “Relational Inductive Biases, Deep Learning, and Graph Networks”)
        decoder (fully connected layer to transfer edge,node and global features to the correct
        size)
        :param state: the current state of the simulation
        :return:
        """
        x, edge_index, edge_attr, u = state.x, state.edge_index, state.edge_attr, state.u
        batch = state.batch

        # create embedding to features of edges, nodes and globals
        edge_attr = self.edge_embedding_model(edge_attr)
        x = self.node_embedding_model(x)
        u = self.global_embedding_model(u)

        # run main message passing model n_passes times
        for mp in self.message_passing:
            x_new, edge_attr_new, u_new = mp(x, edge_index, edge_attr, u, batch)
            x, edge_attr, u = x + x_new, edge_attr + edge_attr_new, u + u_new

        edge_attr_out = self.edge_decoder_model(edge_attr)
        # x_out = self.node_decoder_model(x)
        # u_out = self.global_decoder_model(u)
        if self.use_value_critic:
            # return x_out, edge_attr_out, u_out
            value_out = self.value_model(u)
            return edge_attr_out, value_out
        else:
            return edge_attr_out

    def step(self, state: tg.data.Data, device):
        # PyTorch only accepts mini-batches and not individual observations so we have to add
        # a 'fake' dimension to our observation using un-squeeze
        self.eval()
        with torch.no_grad():
            # run policy in eval mode for running as a policy and not training
            action_values, state_value = self.policy.forward(tg_data.Batch.from_data_list([state.clone()]).to(device))
            action_values = action_values.squeeze()
        self.train()
        # normalize logit values
        action_values = torch.tanh(action_values) * self.logit_normalizer
        # mask out actions that are not possible
        action_values[state.illegal_actions] = -np.inf
        action_probabilities = F.softmax(action_values, dim=0)
        # this creates a distribution to sample from
        action_distribution = Categorical(action_probabilities)
        if ((action_probabilities < 0).any()) or (torch.isnan(action_probabilities).any()):
            print(f'action probs: {action_probabilities}, action values: {action_values}')
            print('Saving a checkpoint before crashing...')
            self.save_checkpoint('crash_checkpoint.pth.tar')
        action = action_distribution.sample()  # trying to sample for both train and test
        action_value = action_values[action]
        # gradients should be in log_prob, actions are without gradients
        if action.item() in state.illegal_actions.nonzero():
            print(f'Warning! Picked an illegal action: {action}')
        return action, action_distribution.log_prob(action), state_value

    def compute_probs_and_state_values(self, batch_states, batch_actions, device):
        """
        Calculates the loss for the current batch
        :return: (total loss, chosen log probabilities, mean approximate kl divergence

        """
        # Get action log probabilities
        state_batch = tg_data.Batch.from_data_list(batch_states).to(device="cpu")
        self.train()
        edge_rows, edge_cols = state_batch.edge_index
        edge_batch_indexes = state_batch.batch[edge_rows]
        batch_scores, batch_state_values = self.forward(state_batch.to(device=device).clone())
        # convert network outputs to cpu -
        batch_scores = batch_scores.to(device="cpu")
        batch_state_values = batch_state_values.to(device="cpu")
        batch_scores = torch.tanh(batch_scores) * self.logit_normalizer
        # in order to get the softmax on each batch separately the indexes for softmax are
        # the batch index for each row in edge index
        batch_scores[state_batch.illegal_actions] = -np.inf

        batch_probabilities = tg_utils.softmax(batch_scores, edge_batch_indexes)
        batch_graph_size = torch.tensor([torch.sum(edge_batch_indexes == b) for b in range(state_batch.num_graphs)]).to(
            "cpu")
        cumulative_batch_actions = torch.tensor(batch_actions).to(device="cpu")
        cumulative_batch_actions[1:] = (torch.cumsum(batch_graph_size, dim=0)[:-1] + cumulative_batch_actions[1:])
        chosen_probabilities = batch_probabilities.gather(dim=0, index=cumulative_batch_actions.view(-1, 1))
        # calculate log after choosing from probability for numerical reasons
        chosen_logprob = torch.log(chosen_probabilities).to(device="cpu").view(-1, 1)
        return chosen_logprob, batch_probabilities, batch_state_values


class PolicyGNN(torch.nn.Module):
    def __init__(self, cfg, model_name):
        super(PolicyGNN, self).__init__()
        self.use_value_critic = cfg['use_value_critic']
        self.use_batch_norm = cfg['use_batch_norm']
        self.logit_normalizer = cfg['logit_normalizer']
        self.model_name = model_name
        if cfg['use_batch_norm']:
            self.edge_embedding_model = Seq(Lin(cfg['edge_feature_dim'], cfg['edge_embedding_dim']),
                                            LeakyReLU(),
                                            BatchNorm1d(cfg['edge_embedding_dim'])
                                            )

            self.node_embedding_model = Seq(Lin(cfg['node_feature_dim'], cfg['node_embedding_dim']),
                                            LeakyReLU(),
                                            BatchNorm1d(cfg['node_embedding_dim'])
                                            )
        else:
            self.edge_embedding_model = Seq(Lin(cfg['edge_feature_dim'], cfg['edge_embedding_dim']),
                                            LeakyReLU(),
                                            )

            self.node_embedding_model = Seq(Lin(cfg['node_feature_dim'], cfg['node_embedding_dim']),
                                            LeakyReLU(),
                                            )

        self.global_embedding_model = Seq(
            Lin(cfg['global_feature_dim'], cfg['global_embedding_dim']),
            LeakyReLU())

        # assume that after embedding the edges, nodes and globals have a new length
        self.message_passing = MetaLayer(EdgeModel(n_edge_features=cfg['edge_embedding_dim'],
                                                   n_node_features=cfg['node_embedding_dim'],
                                                   n_global_features=cfg['global_embedding_dim'],
                                                   n_hiddens=cfg['edge_hidden_dim'],
                                                   n_targets=cfg['edge_target_dim'],
                                                   use_batch_norm=cfg['use_batch_norm']),

                                         NodeModel(n_edge_features=cfg['edge_embedding_dim'],
                                                   n_node_features=cfg['node_embedding_dim'],
                                                   n_global_features=cfg['global_embedding_dim'],
                                                   n_hiddens=cfg['node_hidden_dim'],
                                                   n_targets=cfg['node_target_dim'],
                                                   use_batch_norm=cfg['use_batch_norm']),

                                         GlobalModel(n_node_features=cfg['node_embedding_dim'],
                                                     n_global_features=cfg['global_embedding_dim'],
                                                     n_hiddens=cfg['global_hidden_dim'],
                                                     n_targets=cfg['global_target_dim'],
                                                     use_batch_norm=cfg['use_batch_norm']))

        self.edge_decoder_model = Lin(cfg['edge_target_dim'], cfg['edge_dim_out'])
        if self.use_value_critic:
            self.value_model = Seq(Lin(cfg['global_target_dim'], cfg['value_embedding_dim']),
                                   LeakyReLU(),
                                   Lin(cfg['value_embedding_dim'], 1))

        # in our case this is not needed since we don't use the nodes or global in our agent (if
        # needed in the output
        # this should be uncommented)

        # self.node_decoder_model = Lin(cfg['node_target_dim'], cfg['node_dim_out'])
        #
        # self.global_decoder_model = Lin(cfg['global_target_dim'], cfg['global_dim_out'])

        self.n_passes = cfg['n_passes']

    def forward(self, state: tg.data.Batch):
        """
        this function takes the graph edge,node and global features and returns the features after :
        embedding (fully connected layer to transfer edge,node and global features to the
        embedding size)
        message passing (graph network block, similar to the paper :
        https://arxiv.org/pdf/1806.01261.pdf
        “Relational Inductive Biases, Deep Learning, and Graph Networks”)
        decoder (fully connected layer to transfer edge,node and global features to the correct
        size)
        :param state: the current state of the simulation
        :return:
        """
        x, edge_index, edge_attr, u = state.x, state.edge_index, state.edge_attr, state.u
        batch = state.batch

        # create embedding to features of edges, nodes and globals
        edge_attr = self.edge_embedding_model(edge_attr)
        x = self.node_embedding_model(x)
        u = self.global_embedding_model(u)
        # run main message passing model n_passes times
        for i in range(self.n_passes):
            x, edge_attr, u = self.message_passing(x, edge_index, edge_attr, u, batch)

        edge_attr_out = self.edge_decoder_model(edge_attr)
        # x_out = self.node_decoder_model(x)
        # u_out = self.global_decoder_model(u)
        if self.use_value_critic:
            # return x_out, edge_attr_out, u_out
            value_out = self.value_model(u)
            return edge_attr_out, value_out
        else:
            return edge_attr_out

    def step(self, state: tg.data.Data, device):
        # PyTorch only accepts mini-batches and not individual observations so we have to add
        # a 'fake' dimension to our observation using un-squeeze
        self.eval()
        with torch.no_grad():
            # run policy in eval mode for running as a policy and not training
            action_values, state_value = self.forward(tg_data.Batch.from_data_list([state.clone()]).to(device))
            action_values = action_values.squeeze()
        self.train()
        # normalize logit values
        action_values = torch.tanh(action_values) * self.logit_normalizer
        # mask out actions that are not possible
        action_values[state.illegal_actions] = -np.inf
        action_probabilities = F.softmax(action_values, dim=0)
        # this creates a distribution to sample from
        action_distribution = Categorical(action_probabilities)
        if ((action_probabilities < 0).any()) or (torch.isnan(action_probabilities).any()):
            print(f'action probs: {action_probabilities}, action values: {action_values}')
        action = action_distribution.sample()  # trying to sample for both train and test
        action_value = action_values[action]
        # gradients should be in log_prob, actions are without gradients
        if action.item() in state.illegal_actions.nonzero():
            print(f'Warning! Picked an illegal action: {action}')
        return action, action_distribution.log_prob(action), state_value

    def compute_probs_and_state_values(self, batch_states, batch_actions, device):
        """
        Calculates the loss for the current batch
        :return: (total loss, chosen log probabilities, mean approximate kl divergence

        """
        # Get action log probabilities
        state_batch = tg_data.Batch.from_data_list(batch_states).to(device="cpu")
        self.train()
        edge_rows, edge_cols = state_batch.edge_index
        edge_batch_indexes = state_batch.batch[edge_rows]
        batch_scores, batch_state_values = self.forward(state_batch.to(device=device).clone())
        # convert network outputs to cpu -
        batch_scores = batch_scores.to(device="cpu")
        batch_state_values = batch_state_values.to(device="cpu")
        batch_scores = torch.tanh(batch_scores) * self.logit_normalizer
        # in order to get the softmax on each batch separately the indexes for softmax are
        # the batch index for each row in edge index
        batch_scores[state_batch.illegal_actions] = -np.inf

        batch_probabilities = tg_utils.softmax(batch_scores, edge_batch_indexes)
        batch_graph_size = torch.tensor([torch.sum(edge_batch_indexes == b) for b in range(state_batch.num_graphs)]).to(
            "cpu")
        cumulative_batch_actions = torch.tensor(batch_actions).to(device="cpu")
        cumulative_batch_actions[1:] = (torch.cumsum(batch_graph_size, dim=0)[:-1] + cumulative_batch_actions[1:])
        chosen_probabilities = batch_probabilities.gather(dim=0, index=cumulative_batch_actions.view(-1, 1))
        # calculate log after choosing from probability for numerical reasons
        chosen_logprob = torch.log(chosen_probabilities).to(device="cpu").view(-1, 1)
        return chosen_logprob, batch_probabilities, batch_state_values
