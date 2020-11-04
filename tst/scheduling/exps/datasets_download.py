from torch_geometric.datasets import TUDataset
from definitions_rl_cosims import ROOT_DIR

dataset = TUDataset(root=f"{ROOT_DIR}/datasets/ENZYMES", name='ENZYMES')