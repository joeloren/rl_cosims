import math

import numpy as np
# torch imports
import torch
import torch.nn.functional as F
import torch_geometric as tg
from torch.distributions import Categorical
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, ReLU
from torch_geometric import data as tg_data, utils as tg_utils
from torch_geometric.nn import GATConv
from torch_geometric.nn.norm import BatchNorm


class PolicyFullyConnectedGAT(torch.nn.Module):
    def __init__(self, cfg: dict) -> torch.tensor:
        """
        this class is an encoder - decoder model similar to the attention model implemented in attention, learn to route!
        it returns a probability vector
        :param env: gym environment simulating the vrp problem
        :param cfg: dict of model configuration
        keys:
            * num_vehicles: number of vehicles in problem
            * num_actions: number of possible actions
            * num_customers: number of customers in problem
            * num_nodes: number of nodes in graph
            * num_features: number of features in graph
            * embedding_dim: output dimension of embedding layer
            * encoder dim: dimension used for attention encoder layer
        """
        super(PolicyFullyConnectedGAT, self).__init__()
        self.cfg = cfg
        self.num_depots = cfg['num_depots']
        self.num_customers = cfg['num_customers']
        self.use_value_critic = cfg['use_value_critic']
        # if graph is only customers and depot then: num_nodes = num_customers + num_depots
        self.num_nodes = self.num_customers + self.num_depots
        self.dropout = 0
        self.decode_type = None
        self.num_layers = 3

        self.embedding = Seq(Lin(self.cfg['num_features'], self.cfg['embedding_dim'] * 5),
                             LeakyReLU(),
                             Lin(self.cfg['embedding_dim'] * 5, self.cfg['embedding_dim']))
        # encoder is the following equation for each node:
        #   g_i = BN(h_i + MHA(h_0, h_1, ... h_n))
        #   h_i = BN(g_i + FF(g_i))
        # this is done l times
        # we also compute the global node which is:
        # H = 1/n * sum(h_i)  (for i = 0, 1, ... n)
        self.encoder1 = GATConv(self.cfg['embedding_dim'], self.cfg['embedding_dim'], heads=8,
                                dropout=self.dropout, bias=True, concat=False)
        self.ff_encoder1 = Seq(Lin(self.cfg['embedding_dim'], self.cfg['embedding_dim'] * 5),
                               LeakyReLU(),
                               Lin(self.cfg['embedding_dim'] * 5, self.cfg['embedding_dim']))
        self.batch_norm1 = BatchNorm(self.cfg['embedding_dim'])

        self.encoder2 = GATConv(self.cfg['embedding_dim'], self.cfg['embedding_dim'], heads=8,
                                dropout=self.dropout, bias=True, concat=False)
        self.ff_encoder2 = Seq(Lin(self.cfg['embedding_dim'], self.cfg['embedding_dim'] * 5),
                               ReLU(),
                               Lin(self.cfg['embedding_dim'] * 5, self.cfg['embedding_dim']))
        self.batch_norm2 = BatchNorm(self.cfg['embedding_dim'])

        self.encoder3 = GATConv(self.cfg['embedding_dim'], self.cfg['embedding_dim'], heads=8,
                                dropout=self.dropout, bias=True, concat=False)
        self.ff_encoder3 = Seq(Lin(self.cfg['embedding_dim'], self.cfg['embedding_dim'] * 5),
                               ReLU(),
                               Lin(self.cfg['embedding_dim'] * 5, self.cfg['embedding_dim']))
        self.batch_norm3 = BatchNorm(self.cfg['embedding_dim'])
        # the decoder is done once on the following vector
        # concat([H, h_prev_node, vehicle_capacity])
        self.decoder = GATConv(self.cfg['embedding_dim'], 1, heads=1, dropout=self.dropout, bias=True)
        if self.use_value_critic:
            self.value_model = Seq(Lin(self.num_nodes, cfg['value_embedding_dim'] * 3),
                                   ReLU(),
                                   Lin(cfg['value_embedding_dim'] * 3, 1))

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
        if isinstance(state, tg.data.Batch):
            batch_size = state.num_graphs
        else:
            batch_size = 1
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
        x_out_mean = tg_utils.mean_iou(x_out, )

        output_network = self.decoder(x_out, edge_index)
        if self.use_value_critic:
            # takes the output of the network and returns the value
            value = self.value_model(output_network)
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

    def compute_probs_and_state_values(self, batch_states, batch_actions, device):
        """
        Calculates the loss for the current batch
        :return: (total loss, chosen log probabilities, mean approximate kl divergence

        """
        # Get action log probabilities
        state_batch = tg_data.Batch.from_data_list(batch_states).to(device="cpu")
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
        batch_probabilities = tg_utils.softmax(batch_scores, state_batch.batch)
        cumulative_batch_actions = batch_states.chosen_action_index
        chosen_probabilities = batch_probabilities.gather(dim=0, index=cumulative_batch_actions.view(-1, 1))
        # calculate log after choosing from probability for numerical reasons
        chosen_logprob = torch.log(chosen_probabilities).to(device="cpu").view(-1, 1)
        return chosen_logprob, batch_probabilities, batch_state_values
