import copy
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential

import torch_geometric.transforms as T
from torch_geometric.datasets import QM7b
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm

target = 0
dim = 64


class MyTransform:
    def __call__(self, data):
        data = copy.copy(data)
        data.y = data.y[:, target]  # Specify target.
        return data


class Complete:
    def __call__(self, data):
        data = copy.copy(data)
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


path = osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 'data', 'QM7b')
transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
dataset = QM7b(path, transform=transform).shuffle()

# Normalize targets to mean = 0 and std = 1.
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean[:, target].item(), std[:, target].item()

# Split datasets.
n_train = round(len(dataset) * 0.8)
n_val = round(len(dataset) * 0.1)

test_dataset = dataset[:n_train]
val_dataset = dataset[n_train:n_train+n_val]
train_dataset = dataset[n_train+n_val:]

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self, num_features, dim):
        super().__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(num_features=dataset.num_features, dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)


def train():
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data.x, data.edge_index, data.batch, data.edge_attr), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += (model(data.x, data.edge_index, data.batch, data.edge_attr) * std - data.y * std).abs().sum().item()  # MAE
    return error / len(loader.dataset)


best_val_error = None
for epoch in range(1, 2):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error = test(val_loader)
    scheduler.step(val_error)

    if best_val_error is None or val_error <= best_val_error:
        test_error = test(test_loader)
        best_val_error = val_error

    print(f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '
          f'Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}')

    explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=5),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='log_probs',
            ),
            threshold_config=dict(threshold_type='topk', value=5))


    # explainer = Explainer(
    #     model=model,
    #     algorithm=AttentionExplainer(),
    #     explanation_type='model',
    #     node_mask_type=None,
    #     edge_mask_type='object',
    #     model_config=dict(
    #         mode='multiclass_classification',
    #         task_level='graph',
    #         return_type='log_probs',
    #     ),
    #     threshold_config=dict(threshold_type='topk', value=5))


    for data in tqdm(val_loader, unit="batch", total=len(val_loader)):
        explanation = explainer(x=data.x, edge_index=data.edge_index)
        sub_graph = explanation.get_explanation_subgraph()
        sub_graph.y = data.y
        datapoint = {'subgraph': sub_graph}