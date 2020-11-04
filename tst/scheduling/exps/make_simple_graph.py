import torch
from torch_geometric.data import Data

print("finish import")
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(f"{data}")
# Data(edge_index=[2, 4], x=[3, 1])
print(f"{data.keys}")