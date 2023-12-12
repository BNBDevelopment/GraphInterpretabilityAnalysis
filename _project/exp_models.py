import torch
from torch_geometric.nn import GATConv, global_mean_pool


class Model_BinClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes, hdim=64, n_heads=8):
        super().__init__()
        self.conv1 = GATConv(num_features, hdim//n_heads, heads=n_heads)
        self.conv2 = GATConv(hdim, num_classes, heads=1, concat=False)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index).relu()
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        if not batch is None:
            x = global_mean_pool(x, batch)
        return torch.nn.functional.log_softmax(x, dim=1)


class Model_Regressor(torch.nn.Module):
    def __init__(self, num_features, num_classes, hdim=64, n_heads=8):
        super().__init__()
        self.conv1 = GATConv(num_features, hdim//n_heads, heads=n_heads)
        self.conv2 = GATConv(hdim, 1, heads=1, concat=False)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x.to(torch.float32), edge_index).relu()
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        if not batch is None:
            x = global_mean_pool(x, batch)
        else:
            x = global_mean_pool(x, batch=torch.zeros(x.shape[0]).to(x.device).to(torch.int64))
        return x