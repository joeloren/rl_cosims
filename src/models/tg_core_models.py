import torch
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, BatchNorm1d
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