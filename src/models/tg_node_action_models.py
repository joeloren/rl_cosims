import math

import numpy as np
# torch imports
import torch
import torch.nn.functional as F
from torch_geometric.nn import MetaLayer
import torch_geometric as tg
from torch.distributions import Categorical
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, ReLU, BatchNorm1d
from torch_geometric import data as tg_data, utils as tg_utils
from torch_geometric.nn import GATConv
from torch_geometric.nn.norm import BatchNorm
from torch_scatter import scatter_mean
# our imports
from src.models.tg_core_models import EdgeModel, NodeModel, GlobalModel


class PolicyFullyConnectedGAT(torch.nn.Module):
    def __init__(self, cfg: dict, model_name) -> torch.tensor:
        """
        this class is an encoder - decoder model similar to the attention model implemented in attention, learn to route!
        it returns a probability vector
        :param env: gym environment simulating the vrp problem
        :param cfg: dict of model configuration
        keys:
            * num_features: number of features in graph
            * embedding_dim: output dimension of embedding layer
            * value_embedding_dim dim: dimension used for value embedding layer
        """
        super(PolicyFullyConnectedGAT, self).__init__()
        self.cfg = cfg
        self.use_value_critic = cfg['use_value_critic']
        self.dropout = 0
        self.decode_type = None
        self.num_layers = 1
        self.logit_normalizer = cfg['logit_normalizer']

        self.embedding = Seq(Lin(self.cfg['num_features'], self.cfg['embedding_dim']),
                             LeakyReLU())
        # encoder is the following equation for each node:
        #   g_i = BN(h_i + MHA(h_0, h_1, ... h_n))
        #   h_i = BN(g_i + FF(g_i))
        # this is done l times
        # we also compute the global node which is:
        # H = 1/n * sum(h_i)  (for i = 0, 1, ... n)
        self.encoder1 = GATConv(self.cfg['embedding_dim'], self.cfg['embedding_dim'], heads=8,
                                dropout=self.dropout, bias=True, concat=False)
        self.ff_encoder1 = Seq(Lin(self.cfg['embedding_dim'], self.cfg['embedding_dim']),
                               LeakyReLU())
        self.batch_norm1 = BatchNorm(self.cfg['embedding_dim'])

        self.encoder2 = GATConv(self.cfg['embedding_dim'], self.cfg['embedding_dim'], heads=8,
                                dropout=self.dropout, bias=True, concat=False)
        self.ff_encoder2 = Seq(Lin(self.cfg['embedding_dim'], self.cfg['embedding_dim']),
                               ReLU())
        self.batch_norm2 = BatchNorm(self.cfg['embedding_dim'])

        self.encoder3 = GATConv(self.cfg['embedding_dim'], self.cfg['embedding_dim'], heads=8,
                                dropout=self.dropout, bias=True, concat=False)
        self.ff_encoder3 = Seq(Lin(self.cfg['embedding_dim'], self.cfg['embedding_dim']),
                               ReLU())
        self.batch_norm3 = BatchNorm(self.cfg['embedding_dim'])
        # the decoder is done once on the following vector
        # concat([H, h_prev_node, vehicle_capacity])
        self.decoder = GATConv(self.cfg['embedding_dim'], 1, heads=1, dropout=self.dropout, bias=True)
        if self.use_value_critic:
            self.value_model = Seq(Lin(cfg['embedding_dim'], cfg['value_embedding_dim'] * 2),
                                   ReLU(),
                                   Lin(cfg['value_embedding_dim'] * 2, 1))

        self.init_parameters()

    def init_parameters(self):
        """
        this function initializes all the tensors in the model
        """
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def forward(self, state: tg.data.Batch) -> torch.tensor:
        """
        here the forward of the model is calculated:
            embedding -> [num_features, embedding_dim]
                encoder (attention layer) -> [embedding_dim, encoder_dim]
                    decoder (attention for each vehicle) -> [encoder_dim, num_vehicles]
                        fc layer (combined stats) ->  [num_vehicles*num_nodes, num_actions]
                            probs
        """
        x_in = state.x
        edge_index = state.edge_index
        x = x_in.clone()
        x_out = self.embedding(x)
        for i in range(self.num_layers):
            x_out = self.encoder1(x_out, edge_index)
            x_encoder_1 = self.batch_norm1(self.ff_encoder1(self.encoder1(x_out, edge_index) +
                                                            x_out) + x_out)
            x_encoder_2 = self.batch_norm2(self.ff_encoder2(self.encoder2(x_encoder_1, edge_index) +
                                                            x_encoder_1) + x_encoder_1)
            x_out = self.batch_norm3(self.ff_encoder3(self.encoder3(x_encoder_2, edge_index) +
                                                      x_encoder_2) + x_encoder_2)
        # run model decoder and find next action
        output_network = self.decoder(x_out, edge_index)
        if self.use_value_critic:
            # takes the output of the network and returns the value
            value_input = scatter_mean(x_out, dim=0, index=state.batch)
            value = self.value_model(value_input)
            return output_network, value
        else:
            return output_network

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
            print(f'action probs are negative, action probs: {action_probabilities}, action values: {action_values}')
        action = action_distribution.sample()  # trying to sample for both train and test
        # gradients should be in log_prob, actions are without gradients
        if action.item() in state.illegal_actions.nonzero():
            print(f'Warning! Picked an illegal action: {action}')
        return action, action_distribution.log_prob(action), state_value

    def compute_probs_and_state_values(self, batch_states_list, batch_actions, device):
        """
        Calculates the loss for the current batch
        :return: (total loss, chosen log probabilities, mean approximate kl divergence

        """
        # Get action log probabilities
        state_batch = tg_data.Batch.from_data_list(batch_states_list).to(device="cpu")
        self.train()
        # get batch scores and state values from network -
        batch_scores, batch_state_values = self.forward(state_batch.to(device=device).clone())
        # convert network outputs to cpu -
        batch_scores = batch_scores.to(device="cpu")
        batch_state_values = batch_state_values.to(device="cpu")
        # normalize score with tanh and logit_normalizer
        batch_scores = torch.tanh(batch_scores) * self.logit_normalizer
        # in order to get the softmax on each batch separately the indexes for softmax are
        # the batch node indexes (since the actions are in the nodes)
        batch_scores[state_batch.illegal_actions] = -np.inf
        batch_probabilities = tg_utils.softmax(batch_scores, state_batch.batch.to(device="cpu"))
        cumulative_batch_actions = state_batch.action_chosen_index.to(device="cpu")
        chosen_probabilities = batch_probabilities.gather(dim=0, index=cumulative_batch_actions.view(-1, 1))
        # calculate log after choosing from probability for numerical reasons
        chosen_logprob = torch.log(chosen_probabilities).to(device="cpu").view(-1, 1)
        return chosen_logprob, batch_probabilities, batch_state_values


class PolicyFullyConnectedMessagePassing(torch.nn.Module):
    def __init__(self, cfg, model_name):
        super(PolicyFullyConnectedMessagePassing, self).__init__()
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

        self.node_decoder_model = Lin(cfg['node_target_dim'], cfg['node_dim_out'])
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

        x_out = self.node_decoder_model(x)
        # edge_attr_out = self.edge_decoder_model(edge_attr)
        # u_out = self.global_decoder_model(u)
        if self.use_value_critic:
            # return x_out, edge_attr_out, u_out
            value_out = self.value_model(u)
            return x_out, value_out
        else:
            return x_out

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
            print(f'action probs are negative, action probs: {action_probabilities}, action values: {action_values}')
        action = action_distribution.sample()  # trying to sample for both train and test
        # gradients should be in log_prob, actions are without gradients
        if action.item() in state.illegal_actions.nonzero():
            print(f'Warning! Picked an illegal action: {action}')
        return action, action_distribution.log_prob(action), state_value

    def compute_probs_and_state_values(self, batch_states_list, batch_actions, device):
        """
        Calculates the loss for the current batch
        :return: (total loss, chosen log probabilities, mean approximate kl divergence

        """
        # Get action log probabilities
        state_batch = tg_data.Batch.from_data_list(batch_states_list).to(device="cpu")
        self.train()
        # get batch scores and state values from network -
        batch_scores, batch_state_values = self.forward(state_batch.to(device=device).clone())
        # convert network outputs to cpu -
        batch_scores = batch_scores.to(device="cpu")
        batch_state_values = batch_state_values.to(device="cpu")
        # normalize score with tanh and logit_normalizer
        batch_scores = torch.tanh(batch_scores) * self.logit_normalizer
        # in order to get the softmax on each batch separately the indexes for softmax are
        # the batch node indexes (since the actions are in the nodes)
        batch_scores[state_batch.illegal_actions] = -np.inf
        batch_probabilities = tg_utils.softmax(batch_scores, state_batch.batch.to(device="cpu"))
        cumulative_batch_actions = state_batch.action_chosen_index
        chosen_probabilities = batch_probabilities.gather(dim=0, index=cumulative_batch_actions.view(-1, 1))
        # calculate log after choosing from probability for numerical reasons
        chosen_logprob = torch.log(chosen_probabilities).to(device="cpu").view(-1, 1)
        return chosen_logprob, batch_probabilities, batch_state_values
