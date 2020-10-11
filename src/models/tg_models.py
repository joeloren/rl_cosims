import torch
import torch_geometric as tg
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, BatchNorm1d
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_add


class EdgeModel(torch.nn.Module):
    def __init__(self, n_node_features, n_edge_features, n_global_features, n_hiddens, n_targets,
                 use_batch_norm=False):
        super().__init__()
        if use_batch_norm:
            self.edge_mlp = Seq(
                Lin(2 * n_node_features + n_edge_features + n_global_features, n_hiddens),
                LeakyReLU(),
                Lin(n_hiddens, n_hiddens),
                LeakyReLU(),
                Lin(n_hiddens, n_targets),
                BatchNorm1d(n_targets)
            )
        else:
            self.edge_mlp = Seq(
                Lin(2 * n_node_features + n_edge_features + n_global_features, n_hiddens),
                LeakyReLU(),
                Lin(n_hiddens, n_hiddens),
                LeakyReLU(),
                Lin(n_hiddens, n_targets),
            )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.edge_mlp(out)
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, n_node_features, n_edge_features, n_global_features, n_hiddens, n_targets,
                 use_batch_norm=False):
        super(NodeModel, self).__init__()
        if use_batch_norm:
            self.node_mlp = Seq(
                Lin(n_node_features + n_edge_features + n_global_features, n_hiddens),
                LeakyReLU(),
                Lin(n_hiddens, n_hiddens),
                LeakyReLU(),
                Lin(n_hiddens, n_targets),
                BatchNorm1d(n_targets)
            )
        else:
            self.node_mlp = Seq(
                Lin(n_node_features + n_edge_features + n_global_features, n_hiddens),
                LeakyReLU(),
                Lin(n_hiddens, n_hiddens),
                LeakyReLU(),
                Lin(n_hiddens, n_targets),
            )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        out = self.node_mlp(out)
        return out


class GlobalModel(torch.nn.Module):
    def __init__(self, n_node_features, n_global_features, n_hiddens, n_targets,
                 use_batch_norm=False):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(
            Lin(n_global_features + n_node_features, n_hiddens),
            LeakyReLU(),
            Lin(n_hiddens, n_hiddens),
            LeakyReLU(),
            Lin(n_hiddens, n_targets),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        # batch = batch or torch.zeros_like(x, dtype=torch.long)
        out = torch.cat([u, scatter_add(x, batch, dim=0)], dim=1)
        result = self.global_mlp(out)
        return result


class PolicyMultipleMPGNN(torch.nn.Module):
    def __init__(self, cfg, model_name):
        super(PolicyMultipleMPGNN, self).__init__()
        self.model_name = model_name
        self.n_passes = cfg['n_passes']
        self.use_value_critic = cfg['use_value_critic']
        self.use_batch_norm = cfg['use_batch_norm']
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
        mp_dict = []
        mp_dict.append(MetaLayer(EdgeModel(n_edge_features=cfg['edge_embedding_dim'],
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
                                             use_batch_norm=cfg['use_batch_norm'])))
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


class PolicyGNN(torch.nn.Module):
    def __init__(self, cfg, model_name):
        super(PolicyGNN, self).__init__()
        self.use_value_critic = cfg['use_value_critic']
        self.use_batch_norm = cfg['use_batch_norm']
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
        row, col = edge_index
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
